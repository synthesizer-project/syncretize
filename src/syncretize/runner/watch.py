"""A live, full-terminal view of a run: ``syncretize watch [run]``.

A Textual app that redraws every few seconds (``-n``, default 2) from the
files a run writes, without changing any of them, so it is safe to leave
running anywhere the work directory is visible:

    state.yaml       the run's stage, its start job and its Cloudy sections
    run.log          as it grows
    squeue, sacct    the run's jobs (every 30 s, when Slurm is there)

and, for each Cloudy section (in <work_dir>/<section>):

    state.yaml       next stage and job, rounds, jobs, failed models
    queue/*          how many models are waiting, running, done and failed,
                     and how long each running model has been going
    state/*.jsonl    the current round's worker records (read as they grow)
    grid_points.json each model's grid point (from the setup)

One grid (Cloudy section) is shown at a time: a card with its run, name,
stage and progress, then its panels, whose log shows the run.log lines about
that section and the run as a whole. g opens the all grids page (progress
across every grid, then a row per grid; Enter shows one), and 1-9 and 0
show the grid at that place. Every
panel scrolls with the mouse; run.log follows new lines unless scrolled up.
Keys: l run.log and r the running models full screen (again to go back), s
your Slurm queue (Esc to go back), q quits.

Given a directory holding runs instead (e.g. many runs' directories, each
with its run.yaml), every grid of every run under way there is watched. Runs
appear once submitted and drop off once done (the directory is searched
every 30 s). Only the shown grid's run is read in full; the others' state
files and queues are read every QUEUE_EVERY seconds, for the all grids page.
"""

import functools
import getpass
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone

import yaml
from rich import box
from rich.cells import cell_len
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    ContentSwitcher,
    DataTable,
    Footer,
    RichLog,
    Static,
)

from syncretize.runner.cloudy import iteration, model_path
from syncretize.runner.perturb import is_perturbed
from syncretize.runner.queue import STATES

#: How often squeue is asked about the run's jobs (s)
SLURM_EVERY = 30

#: How often the Slurm queue view asks squeue for all your jobs (s)
SQUEUE_EVERY = 10

#: The Slurm queue view's columns, and squeue's format for them
SQUEUE_COLUMNS = (
    "job",
    "partition",
    "name",
    "state",
    "time",
    "limit",
    "nodes",
    "nodelist (reason)",
)
SQUEUE_FORMAT = "%i|%P|%j|%T|%M|%l|%D|%R"

#: Above this many models, the queue is counted every QUEUE_EVERY seconds
#: rather than every redraw (listing big directories is slow on Lustre)
BIG_QUEUE = 20000
QUEUE_EVERY = 10


def find_work_dir(path=None):
    """The work directory of a run, from a run config or a directory.

    Args:
        path (str)
            A run config, a work directory (holding state.yaml) or a
            directory holding run.yaml; None for the current directory.

    Returns:
        str
            The work directory.
    """
    path = os.path.abspath(path or os.getcwd())
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "state.yaml")):
            return path
        if os.path.exists(os.path.join(path, "run.yaml")):
            path = os.path.join(path, "run.yaml")
        else:
            raise ValueError(
                f"{path} holds neither a run's state.yaml nor a run.yaml"
            )
    # Only work_dir is needed, so the config isn't checked (which would
    # import synthesizer and take seconds)
    with open(path) as file:
        work_dir = os.path.expanduser(
            (yaml.safe_load(file) or {})["Run"]["work_dir"]
        )
    return os.path.normpath(os.path.join(os.path.dirname(path), work_dir))


def find_runs(directory):
    """The runs under a directory, by where they are.

    A run is a directory holding a run.yaml; the directories inside a run
    aren't searched (they hold its work, not more runs).

    Args:
        directory (str)
            The directory to search.

    Returns:
        dict
            The work directories of the runs "active" (submitted, not done),
            "done" and "waiting" (not yet submitted), in path order.
    """
    found = {"active": [], "done": [], "waiting": []}
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        if "run.yaml" not in files:
            continue
        dirs.clear()
        try:
            work_dir = find_work_dir(os.path.join(root, "run.yaml"))
        except (KeyError, TypeError, yaml.YAMLError):
            continue  # not a Runner config
        state = _read_yaml(os.path.join(work_dir, "state.yaml"))
        if not state:
            found["waiting"].append(work_dir)
        elif state.get("next") == "done":
            found["done"].append(work_dir)
        else:
            found["active"].append(work_dir)
    return found


class Watcher:
    """Reads a run's files, keeping what is slow to read between redraws.

    Attributes:
        work_dir (str)
            The run's work directory.
        show_squeue (bool)
            Whether to ask squeue for all your jobs (while the Slurm queue
            view is open).
    """

    def __init__(self, work_dir):
        """Watch a run.

        Args:
            work_dir (str)
                The run's work directory.
        """
        self.work_dir = work_dir
        self.show_squeue = False
        self._sections = {}  # each Cloudy section's reader, by its label
        self._jobs, self._asked = {}, 0.0
        self._ended = {}  # job ID: how its tasks ended, from sacct
        self._log_offset = None  # how much of run.log has been read
        self._squeue, self._squeue_at = None, 0.0
        self._brief, self._briefed_at = None, -math.inf

    def read(self):
        """Everything the view shows.

        Returns:
            dict
                "run" (the run's state), "sections" (a snapshot of each
                Cloudy section, see ``_SectionReader.read``, or for a run
                without Cloudy one of the run itself), "squeue" (your Slurm
                queue, when asked for) and "now".
        """
        run = _read_yaml(os.path.join(self.work_dir, "state.yaml"))
        readers = self._readers(run)
        states = {reader.label: reader.state() for reader in readers}

        # The run's start job, then each section's own
        jobs = list(run.get("jobs", []))
        for label, state in states.items():
            if label is not None:
                jobs += state.get("jobs", [])
        slurm = self._slurm(run, jobs)
        squeue = self._all_jobs() if self.show_squeue else None
        log, log_new = self._log_tail(), self._log_new()
        now = datetime.now(timezone.utc)

        sections = []
        for reader in readers:
            label, state = reader.label, states[reader.label]
            own = state.get("jobs", []) if label is not None else []
            shared = {
                "run": run,
                "jobs": [*run.get("jobs", []), *own],
                "slurm": {
                    job: entry
                    for job, entry in slurm.items()
                    if _about(entry["name"], label)
                },
                "ended": self._ended,
                "squeue": squeue,
                "log": [line for line in log if _about_line(line, label)],
                "log_new": [
                    line for line in log_new if _about_line(line, label)
                ],
                "now": now,
            }
            sections.append(reader.read(state, shared))
        return {"run": run, "sections": sections, "squeue": squeue, "now": now}

    def brief(self):
        """Only the run's state, and each section's with its queue's counts
        (enough for its tab and the overview). Read again only every
        QUEUE_EVERY seconds, as many runs are briefed at once.

        Returns:
            dict
                "run" (the run's state) and "sections" (each section's
                "label", "state" and "counts", or for a run without Cloudy
                sections yet one of the run itself, labelled None).
        """
        if time.monotonic() - self._briefed_at < QUEUE_EVERY:
            return self._brief
        run = _read_yaml(self._path("state.yaml"))
        sections = [
            {
                "label": reader.label,
                "run": run,
                "state": reader.state() if reader.label else run,
                "counts": reader._queue_counts() if reader.label else {},
            }
            for reader in self._readers(run)
        ]
        self._brief = {"run": run, "sections": sections}
        self._briefed_at = time.monotonic()
        return self._brief

    def _readers(self, run):
        """Each Cloudy section's reader, or one of the run without Cloudy.

        Args:
            run (dict)
                The run's state.
        """
        for label, name in (run.get("sections") or {}).items():
            if label not in self._sections:
                self._sections[label] = _SectionReader(
                    os.path.join(self.work_dir, name), label
                )
        return list(self._sections.values()) or [
            _SectionReader(self.work_dir, None)
        ]

    def _path(self, *names):
        """A path in the run's work directory."""
        return os.path.join(self.work_dir, *names)

    def _log_tail(self, size=65536):
        """run.log's last lines (its last 64 KB)."""
        try:
            with open(self._path("run.log"), "rb") as file:
                file.seek(0, os.SEEK_END)
                file.seek(max(0, file.tell() - size))
                return file.read().decode(errors="replace").splitlines()[1:]
        except FileNotFoundError:
            return []

    def _log_new(self, first=65536):
        """run.log's lines since the last read (at first, its last 64 KB)."""
        partial = False
        try:
            with open(self._path("run.log"), "rb") as file:
                file.seek(0, os.SEEK_END)
                end = file.tell()
                if self._log_offset is None or self._log_offset > end:
                    self._log_offset = max(0, end - first)
                    partial = self._log_offset > 0  # mid-line
                file.seek(self._log_offset)
                data = file.read()
        except FileNotFoundError:
            return []
        # A partly written last line waits for the next read
        done = data.rfind(b"\n") + 1
        self._log_offset += done
        lines = data[:done].decode(errors="replace").splitlines()
        return lines[1:] if partial else lines

    def _all_jobs(self):
        """All your jobs in squeue (SQUEUE_COLUMNS), or None without it."""
        if shutil.which("squeue") is None:
            return None
        if time.monotonic() - self._squeue_at < SQUEUE_EVERY:
            return self._squeue
        try:
            result = subprocess.run(
                ["squeue", "-u", getpass.getuser(), "-h", "-o", SQUEUE_FORMAT],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            return self._squeue
        self._squeue_at = time.monotonic()
        width = len(SQUEUE_COLUMNS)
        self._squeue = [
            (line.split("|", width - 1) + [""] * width)[:width]
            for line in result.stdout.splitlines()
        ]
        return self._squeue

    def _slurm(self, run, jobs):
        """The run's queued and running jobs, by job ID (every 30 s).

        Also asks sacct how the run's other jobs ended, once each (a job that
        has ended doesn't change), into ``self._ended``.

        Args:
            run (dict)
                The run's state.
            jobs (list of dict)
                Every job the run and its sections have submitted.

        Returns:
            dict
                Each queued or running job's name and how many of its tasks
                are running and pending.
        """
        name = run.get("name")
        if name is None or shutil.which("squeue") is None:
            return {}
        if time.monotonic() - self._asked < SLURM_EVERY:
            return self._jobs
        self._asked = time.monotonic()
        try:
            result = subprocess.run(
                ["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%j|%T|%M"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            return self._jobs
        live = {}
        for line in result.stdout.splitlines():
            job, job_name, status, elapsed = (line.split("|") + [""] * 4)[:4]
            if f"-{name}-" not in f"-{job_name}-":
                continue
            base = job.split("_")[0]
            entry = live.setdefault(
                base, {"name": job_name, "running": 0, "pending": 0}
            )
            tasks = _count_tasks(job)
            if status == "RUNNING":
                entry["running"] += tasks
                entry["elapsed"] = elapsed
            else:
                entry["pending"] += tasks
        self._jobs = live

        ask = [
            str(entry["job"])
            for entry in jobs
            if str(entry["job"]) not in live
            and str(entry["job"]) not in self._ended
        ]
        if ask and shutil.which("sacct") is not None:
            self._ended |= _sacct(ask)
        return live


class _SectionReader:
    """Reads one Cloudy section's files (or a run without Cloudy's).

    Attributes:
        directory (str)
            The section's directory (the work directory for a run without
            Cloudy).
        label (str)
            The section's label, e.g. "C1" (None for a run without Cloudy).
    """

    def __init__(self, directory, label):
        """Watch a section.

        Args:
            directory (str)
                Its directory.
            label (str)
                Its label, or None.
        """
        self.directory = directory
        self.label = label
        self._records = {}  # worker record file: (offset, records)
        self._counts, self._counted = None, 0.0
        self._running_cache, self._running_at = [], 0.0
        self._models, self._models_mtime = {}, None
        self._finished = []  # every finished attempt, from models.json
        self._names = {state: [] for state in STATES}  # the queue's models
        self._grid_points = None

    def state(self):
        """The section's state file."""
        return _read_yaml(self._path("state.yaml"))

    def read(self, state, shared):
        """A snapshot of the section, for the view.

        Args:
            state (dict)
                Its state file.
            shared (dict)
                What it shares with the run: "run" (the run's state),
                "jobs" (the start job and the section's own), "slurm" (its
                live jobs), "ended", "squeue", "log" and "log_new" (the
                run.log lines about it or the whole run) and "now".

        Returns:
            dict
                The shared values and the section's "label", "state",
                "counts", "records", "attempts", "finished", "running",
                "names", "points" and "cloudy_dir".
        """
        return {
            **shared,
            "label": self.label,
            "state": state,
            "counts": self._queue_counts(),
            "records": self._round_records(),
            "attempts": self._attempts(),
            "finished": self._finished,
            "running": self._running(),
            "names": self._names,
            "points": self._points(state.get("cloudy_dir")),
            "cloudy_dir": state.get("cloudy_dir"),
        }

    def _path(self, *names):
        """A path in the section's directory."""
        return os.path.join(self.directory, *names)

    def _queue_counts(self):
        """How many models are in each queue state (and which models)."""
        total = sum(self._counts.values()) if self._counts else 0
        if (
            total > BIG_QUEUE
            and time.monotonic() - self._counted < QUEUE_EVERY
        ):
            return self._counts
        counts = {}
        for state in STATES:
            try:
                names = os.listdir(self._path("queue", state))
            except FileNotFoundError:
                names = []
            self._names[state] = [name.partition("@")[0] for name in names]
            counts[state] = len(names)
        self._counts, self._counted = counts, time.monotonic()
        return counts

    def _points(self, cloudy_dir):
        """Each model's grid point (grid_points.json, from the setup)."""
        if self._grid_points is None and cloudy_dir is not None:
            try:
                with open(
                    os.path.join(cloudy_dir, "grid_points.json")
                ) as file:
                    self._grid_points = json.load(file)
            except (OSError, json.JSONDecodeError):
                return None
        return self._grid_points

    def _running(self):
        """The claimed models and how long they've run, longest first.

        A claim is a rename into queue/claimed, which sets the file's ctime,
        so that is when the model started.
        """
        directory = self._path("queue", "claimed")
        busy = len(self._running_cache) > BIG_QUEUE // 10
        if busy and time.monotonic() - self._running_at < QUEUE_EVERY:
            return self._running_cache
        now, running = time.time(), []
        try:
            names = os.listdir(directory)
        except FileNotFoundError:
            names = []
        for name in names:
            try:
                started = os.stat(os.path.join(directory, name)).st_ctime
            except FileNotFoundError:
                continue  # finished since the listing
            model, _, owner = name.rpartition("@")
            running.append((model, owner, now - started))
        running.sort(key=lambda item: -item[2])
        self._running_cache, self._running_at = running, time.monotonic()
        return running

    def _round_records(self):
        """The workers' records not yet gathered by a finish job."""
        directory = self._path("state")
        try:
            names = [n for n in os.listdir(directory) if n.endswith(".jsonl")]
        except FileNotFoundError:
            names = []
        # Files gathered by a finish job are gone; forget them
        self._records = {n: r for n, r in self._records.items() if n in names}
        for name in names:
            offset, records = self._records.get(name, (0, []))
            try:
                with open(os.path.join(directory, name)) as file:
                    file.seek(offset)
                    for line in file:
                        if not line.endswith("\n"):
                            break  # still being written
                        offset += len(line.encode())
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            except FileNotFoundError:
                continue
            self._records[name] = (offset, records)
        return [r for _, records in self._records.values() for r in records]

    def _attempts(self):
        """How many times each model has run, from models.json."""
        try:
            mtime = os.path.getmtime(self._path("models.json"))
        except FileNotFoundError:
            return {}
        if mtime != self._models_mtime:
            try:
                with open(self._path("models.json")) as file:
                    models = json.load(file)
            except (OSError, json.JSONDecodeError):
                return self._models
            self._models = {m: len(e["attempts"]) for m, e in models.items()}
            self._finished = [
                attempt
                for entry in models.values()
                for attempt in entry["attempts"]
                if attempt.get("result") in ("done", "failed")
            ]
            self._models_mtime = mtime
        return self._models


def _read_yaml(path):
    """A YAML file's contents (empty if it is missing or half written)."""
    try:
        with open(path) as file:
            return yaml.safe_load(file) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


#: A Cloudy section's label, e.g. C2
SECTION_LABEL = re.compile(r"C\d+")


def _about(job_name, label):
    """Whether a job is the start job or one of a section's.

    Args:
        job_name (str)
            The job's name, e.g. "R2-bc03-C1-Cloudy".
        label (str)
            The section's label (None for a run without Cloudy).

    Returns:
        bool
    """
    parts = job_name.split("-")
    labels = [part for part in parts if SECTION_LABEL.fullmatch(part)]
    return not labels or label in labels


def _about_line(line, label):
    """Whether a run.log line is about a section or the whole run.

    Lines about a section start their tag with its label, e.g.
    "2026-11-20 13:06:20 [C2 finish 123] ...".

    Args:
        line (str)
            The line.
        label (str)
            The section's label (None for a run without Cloudy).

    Returns:
        bool
    """
    match = re.match(r"\S+ \S+ \[(\S+)", line)
    if match is None or not SECTION_LABEL.fullmatch(match.group(1)):
        return True
    return match.group(1) == label


def _sacct(jobs):
    """How the tasks of ended jobs ended, e.g. {"12": {"COMPLETED": 249}}."""
    try:
        result = subprocess.run(
            [
                "sacct",
                "-n",
                "-X",
                "-P",
                "-o",
                "JobID,State",
                "-j",
                ",".join(jobs),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    ended = {}
    for line in result.stdout.splitlines():
        job, _, status = line.partition("|")
        status = status.split(" ")[0]  # e.g. "CANCELLED by 123"
        if status in ("PENDING", "RUNNING", "REQUEUED", "COMPLETING"):
            continue
        tasks = ended.setdefault(job.split("_")[0], {})
        tasks[status] = tasks.get(status, 0) + _count_tasks(job)
    return ended


def _count_tasks(job):
    """How many array tasks a squeue job ID stands for, e.g. 1_[0-9,12]."""
    if "[" not in job:
        return 1
    spec = job.split("[", 1)[1].rstrip("]").split("%")[0]
    count = 0
    for part in spec.split(","):
        start, _, end = part.partition("-")
        count += int(end) - int(start) + 1 if end else 1
    return count


def _duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _grid_points(snapshot):
    """How many grid points are in each queue state, and interpolated.

    Runs perturbed around failed models aren't points of the grid, and the
    failed models interpolated from them are filled, not failed.

    Returns:
        (dict, int, int)
            The grid points in each state ("failed" without those
            interpolated), how many were interpolated, and how many perturbed
            runs there are.
    """
    points = {
        state: sum(not is_perturbed(model) for model in names)
        for state, names in snapshot["names"].items()
    }
    perturbed = sum(
        is_perturbed(model)
        for names in snapshot["names"].values()
        for model in names
    )
    interpolated = len(snapshot["state"].get("interpolated") or [])
    points["failed"] -= interpolated
    return points, interpolated, perturbed


#: The stage badge in the header: its label and style
STAGE_BADGES = {
    "start": ("STARTING", "bold black on bright_blue"),
    "cloudy": ("CLOUDY RUNNING", "bold black on bright_cyan"),
    "finish": ("ROUND {round} RUNNING", "bold black on bright_cyan"),
    "assemble": ("ASSEMBLING", "bold black on yellow"),
    "done": ("DONE", "bold black on green"),
}

#: The badge while failed models' perturbed runs are under way
PERTURB_BADGE = ("PERTURBING (ROUND {round})", "bold black on bright_magenta")

#: The badges of a round whose workers haven't started (its models all
#: waiting), and of one with no models left (waiting for its finish job)
QUEUED_BADGE = ("ROUND {round} QUEUED", "bold black on grey70")
FINISHING_BADGE = ("ROUND {round} FINISHING", "bold black on cyan")


def _header(view, width=118):
    """The header: the run and its stage, then its progress across its grids.

    For a run with one Cloudy section (or none), the stage is the section's
    (e.g. its round); with several, how many of their grids are done. Built
    as two lines of the given width (Textual lays nested rich tables out
    unreliably).

    Args:
        view (dict)
            What ``Watcher.read`` returns.
        width (int)
            The header's width.

    Returns:
        Text
            The two lines.
    """
    run, sections = view["run"], view["sections"]
    if len(sections) == 1:
        label, style = _badge(sections[0]["state"], sections[0]["counts"])
        details = _section_details(sections[0]["state"])
    else:
        done = sum(s["state"].get("next") == "done" for s in sections)
        stage = run.get("next", "?")
        label, style = STAGE_BADGES.get(stage, (stage.upper(), "bold reverse"))
        if stage == "cloudy":
            label, style = (
                f"{len(sections) - done} GRIDS RUNNING",
                "bold black on bright_cyan",
            )
        details = [("grids done", f"{done}/{len(sections)}")]
    left = Text.assemble(
        (run.get("name", "?"), "bold"), "   ", (f" {label} ", style)
    )
    return Text("\n").join(
        [_spread(left, _details(details), width), _progress(sections, width)]
    )


def _badge(state, counts=None):
    """A section's stage badge (or a run's without Cloudy), from its state.

    Args:
        state (dict)
            Its state file.
        counts (dict)
            How many of its models are in each queue state, to tell a round
            under way from one queued or finishing (None not to).

    Returns:
        (str, str)
            Its label, e.g. "ROUND 3 RUNNING", and style.
    """
    stage = state.get("next", "?")
    label, style = STAGE_BADGES.get(stage, (stage.upper(), "bold reverse"))
    if stage == "finish" and state.get("perturbing"):
        label, style = PERTURB_BADGE
    elif stage == "finish" and counts and not counts.get("claimed"):
        label, style = QUEUED_BADGE if counts.get("todo") else FINISHING_BADGE
    return label.format(round=state.get("round", "?")), style


def _section_details(state):
    """A section's round, its workers' time limit and its next job."""
    return [
        ("round", state.get("round", "-")),
        ("limit", _limit(state)),
        ("next job", state.get("next_job", "-")),
    ]


def _details(details):
    """Name-value pairs as one line, e.g. "round 3 · limit 2:00:00"."""
    text = Text()
    for n, (name, value) in enumerate(details):
        if n:
            text.append("  ·  ", style="grey50")
        text.append(f"{name} ", style="dim")
        text.append(str(value))
    return text


def _counts(counts):
    """Counts as one line, each before what it counts, e.g. "10 runs under
    way · 18 done"."""
    text = Text()
    for n, (count, what) in enumerate(counts):
        if n:
            text.append("  ·  ", style="grey50")
        text.append(f"{count:,} ", style="bold")
        text.append(what, style="dim")
    return text


def _spread(left, right, width):
    """Two texts on one line of the given width, at its two ends."""
    return left + " " * max(width - left.cell_len - right.cell_len, 1) + right


def _bar(fraction, width, style="green"):
    """A progress bar of the given width, filled to a fraction."""
    filled = round(max(width, 0) * min(max(fraction, 0.0), 1.0))
    return Text("━" * filled, style=style) + Text(
        "━" * (max(width, 0) - filled), style="grey30"
    )


def _progress(snapshots, width):
    """Progress across sections' grid points: a bar, then the counts.

    Args:
        snapshots (list of dict)
            The sections' snapshots (see ``_SectionReader.read``).
        width (int)
            The line's width.

    Returns:
        Text
            One line, e.g. "━━━━━━━  656/657 grid points (99.8%) · 1 failed".
    """
    done = failed = total = interpolated = left_to_run = 0
    rate = 0.0
    for snapshot in snapshots:
        points, filled, _ = _grid_points(snapshot)
        done += points["done"]
        failed += points["failed"]
        interpolated += filled
        total += sum(points.values()) + filled
        counts = snapshot["counts"]
        left_to_run += counts["todo"] + counts["claimed"]
        rate += _rate(snapshot["records"])
    summary = Text.assemble(
        "  ",
        (f"{done:,}", "bold green"),
        f"/{total:,} grid points ",
        (f"({100 * done / max(total, 1):.1f}%)", "dim"),
        ("  ·  ", "grey50"),
        (f"{failed:,} failed", "bold red" if failed else "dim"),
    )
    if interpolated:
        summary.append("  ·  ", style="grey50")
        summary.append(f"{interpolated:,} interpolated", style="bold yellow")
    if rate and left_to_run:
        summary.append("  ·  ", style="grey50")
        summary.append(f"{rate:.1f}/min", style="dim")
        summary.append(f"  ETA ~{_duration(left_to_run / rate * 60)}")
    fraction = (done + failed + interpolated) / max(total, 1)
    return _bar(fraction, max(width - summary.cell_len, 10)) + summary


def _card(snapshot, width=118):
    """A grid's card, atop its tab: its run, name and stage, then progress.

    Args:
        snapshot (dict)
            The section's snapshot (see ``_SectionReader.read``).
        width (int)
            The card's width.

    Returns:
        Text
            The two lines.
    """
    state = snapshot["state"]
    title = Text.assemble((snapshot["run"].get("name", "?"), "bold"))
    if snapshot["label"] is not None:
        title.append("  ·  ", style="grey50")
        title.append(f"{snapshot['label']} ", style="bold cyan")
        title.append(state.get("name") or "", style="bold cyan")
    label, style = _badge(state, snapshot["counts"])
    title.append("   ")
    title.append(f" {label} ", style=style)
    details = _details(_section_details(state) if snapshot["label"] else [])
    return Text("\n").join(
        [_spread(title, details, width), _progress([snapshot], width)]
    )


def _overview(views, found, width=118):
    """Progress across every grid being run under a directory.

    Args:
        views (dict)
            Each run under way's read (``Watcher.read`` or ``brief``).
        found (dict)
            The runs by where they are (see ``find_runs``).
        width (int)
            The overview's width.

    Returns:
        Text
            Two lines: a bar across every grid's models, then how many runs
            and grids are where.
    """
    counts = {state: 0 for state in STATES}
    grids = {"done": 0, "running": 0, "starting": 0}
    for view in views.values():
        for snapshot in view["sections"]:
            for state, n in (snapshot.get("counts") or {}).items():
                counts[state] = counts.get(state, 0) + n
            stage = snapshot["state"].get("next")
            if snapshot["label"] is None or stage is None:
                grids["starting"] += 1
            else:
                grids["done" if stage == "done" else "running"] += 1
    total = sum(counts.values())
    summary = Text.assemble(
        "  ",
        (f"{counts['done']:,}", "bold green"),
        f"/{total:,} models ",
        (f"({100 * counts['done'] / max(total, 1):.1f}%)", "dim"),
        ("  ·  ", "grey50"),
        (f"{counts['claimed']:,} running", "cyan"),
        ("  ·  ", "grey50"),
        (
            f"{counts['failed']:,} failed",
            "bold red" if counts["failed"] else "dim",
        ),
    )
    label = Text.assemble((" ALL GRIDS ", "bold black on magenta"), "  ")
    fraction = (counts["done"] + counts["failed"]) / max(total, 1)
    bar = _bar(fraction, width - label.cell_len - summary.cell_len)
    runs = _counts(
        [
            (len(found.get(key, [])), where)
            for where, key in (
                ("runs under way", "active"),
                ("done", "done"),
                ("waiting", "waiting"),
            )
            if found.get(key) or key == "active"
        ]
    )
    # e.g. "2 grids done · 19 running"
    grids = [(n, where) for where, n in grids.items() if n]
    if grids:
        grids[0] = (grids[0][0], f"grids {grids[0][1]}")
    sections = _counts(grids)
    indent = Text(" " * label.cell_len)
    return Text("\n").join(
        [label + bar + summary, _spread(indent + runs, sections, width)]
    )


def _tab_title(snapshot):
    """A section's tab title: its label and name, and where it is.

    Args:
        snapshot (dict)
            The section's snapshot.

    Returns:
        str
            e.g. "1 c23.01-sps ▸ R3".
    """
    state = snapshot["state"]
    number = snapshot["label"].removeprefix("C")
    return f"{number} {state.get('name') or snapshot['label']} {_where(state)}"


def _where(state):
    """Where a section is, from its state, e.g. "▸ R3" or "✓"."""
    stage = state.get("next")
    if stage == "done":
        return "✓"
    if stage == "finish":
        where = f"▸ R{state.get('round', 1)}"
        return where + (" perturb" if state.get("perturbing") else "")
    if stage == "assemble":
        return "▸ assembling"
    return "…"


def _limit(state):
    """The current round's worker time limit."""
    rounds = state.get("rounds") or [{}]
    return state.get("walltime") or rounds[-1].get("walltime", "-")


def _rate(records, window=600):
    """Models finished per minute, over the last ten minutes of the round.

    Early in a round the rate is taken over the time since its first model
    started, so it isn't diluted by minutes before anything ran.
    """
    now = datetime.now(timezone.utc)
    ages = []  # how long ago each model started and ended (s)
    for record in records:
        try:
            started = now - datetime.fromisoformat(record["started"])
        except (KeyError, ValueError):
            continue
        started = started.total_seconds()
        ages.append((started, started - record.get("seconds", 0)))
    if not ages:
        return 0.0
    span = min(window, max(started for started, _ in ages))
    recent = sum(ended <= span for _, ended in ages)
    return recent / max(span / 60, 1)


def _queue_table(snapshot):
    """The grid points in each state, with their share of the grid."""
    points, interpolated, perturbed = _grid_points(snapshot)
    total = max(sum(points.values()) + interpolated, 1)
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    table.add_column(justify="right", no_wrap=True, style="dim")
    table.add_column(ratio=1)
    rows = [
        ("todo", points["todo"], "bright_white"),
        ("running", points["claimed"], "cyan"),
        ("done", points["done"], "green"),
        ("failed", points["failed"], "red"),
    ]
    if interpolated:
        rows.append(("interpolated", interpolated, "yellow"))
    for label, n, colour in rows:
        table.add_row(
            Text(label, style=colour),
            Text(f"{n:,}", style=f"bold {colour}" if n else "dim"),
            f"{100 * n / total:.1f}%",
            ProgressBar(
                total=total,
                completed=n,
                complete_style=colour,
                finished_style=colour,
            ),
        )
    if perturbed:
        table.add_row(
            Text("perturbed runs", style="dim"),
            Text(f"{perturbed:,}", style="dim"),
            "",
            "",
        )
    return table


def _round_table(snapshot):
    """The round under way: its workers and the models they've run."""
    records = snapshot["records"]
    results = [r["result"] for r in records]
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(no_wrap=True)
    if snapshot["state"].get("next") != "finish" and not records:
        # Between rounds (or done): the run's Cloudy time so far
        cpu = sum(r.get("seconds", 0) for r in snapshot["finished"]) / 3600
        table.add_row("", Text("no round running", style="dim italic"))
        table.add_row("Cloudy runs", f"{len(snapshot['finished']):,}")
        table.add_row("CPU time", f"{cpu:,.1f} h")
        return table
    workers = [
        job
        for job in snapshot["slurm"].values()
        # -Cloudy and -PerturbCloudy
        if job["name"].endswith("Cloudy")
    ]
    if workers:
        running = sum(job["running"] for job in workers)
        pending = sum(job["pending"] for job in workers)
        table.add_row(
            "workers",
            Text.assemble(
                (f"{running} running", "bold cyan"),
                (f"  {pending} queued", "dim"),
            ),
        )
    table.add_row(
        "finished",
        Text.assemble(
            (f"{len(records):,}", "bold green" if records else "dim"),
            (f"  {_rate(records):.1f}/min" if records else "", "dim"),
        ),
    )
    # Runs that errored or hit the time limit (and how many did), those put
    # back to run again, and those given up on
    errored = len(records) - results.count("done")
    timed_out = results.count("stopped")
    failed = sum(_given_up(r) for r in records)
    table.add_row(
        "errored",
        Text.assemble(
            (f"{errored}", "bold red" if errored else "dim"),
            (
                f"  {timed_out} timed out · {errored - failed} retried"
                f" · {failed} failed"
                if errored
                else "",
                "dim",
            ),
        ),
    )
    times = [
        r["seconds"] for r in records if r["result"] in ("done", "failed")
    ]
    if times:
        table.add_row(
            "run times",
            f"median {_duration(statistics.median(times))}, "
            f"longest {_duration(max(times))}",
        )
        table.add_row(
            "CPU time",
            f"{sum(r.get('seconds', 0) for r in records) / 3600:.1f} h",
        )
    return table


def _rounds(state, records, counts):
    """The finished rounds, and the one under way from its workers' records."""
    rounds = [r for r in state.get("rounds", []) if "queue" in r]
    current = state.get("round")
    if state.get("next") == "finish" and current is not None:
        submitted = next(
            (r for r in state.get("rounds", []) if r["round"] == current), {}
        )
        live = {"round": f"{current} ▸", "queue": counts}
        live["perturb"] = state.get("perturbing", False)
        live["walltime"] = submitted.get("walltime", _limit(state))
        live["workers"] = submitted.get("workers", "here")
        for result in ("done", "failed", "stopped", "lost"):
            live[result] = sum(r["result"] == result for r in records)
        live["given_up"] = sum(_given_up(r) for r in records)
        rounds.append(live)
    return rounds


def _given_up(record):
    """Whether a worker's record gave up on its model."""
    return record.get("attempt") is not None and not record.get("requeued")


def _failures(snapshot):
    """The models that failed every attempt, with why the last failed.

    Those a finish job has gathered, and those failed since.
    """
    failures = dict(snapshot["state"].get("failures") or {})
    for record in snapshot["records"]:
        if _given_up(record):
            failures[record["model"]] = record.get("reason")
    return failures


#: How grid axes are shown: their label, and whether on a log scale
AXIS_LABELS = {
    "ages": ("log age", True),
    "metallicities": ("Z", False),
    "ionisation_parameter": ("log U", True),
    "reference_ionisation_parameter": ("log U_ref", True),
    "hydrogen_density": ("log nH", True),
    "abundance_scalings.carbon_to_oxygen": ("C/O", False),
    "abundance_scalings.nitrogen_to_oxygen": ("N/O", False),
    "depletion_scale": ("depl", False),
    "gas_metallicity": ("Z gas", False),
    "alpha_enhancement": ("alpha", False),
}


def _point_columns(points):
    """The grid's axes, incident then photoionisation, as columns.

    Returns:
        list of (label, log, part, position)
    """
    if points is None:
        return []
    columns = []
    for part in ("incident", "photoionisation"):
        for position, axis in enumerate(points[part]["axes"]):
            label, log = AXIS_LABELS.get(axis, (axis.split(".")[-1], False))
            columns.append((label, log, part, position))
    return columns


def _format(value, log):
    if log and value > 0:
        return f"{math.log10(value):.2f}"
    return f"{value:.3g}"


def _point(points, model, columns):
    """A model's grid point, as text for each column.

    A perturbed run ("<i>_<j>p...") shows the grid point it was run around.
    """
    i, _, j = model.partition("_")
    i, j = int(i), int(j.partition("p")[0])
    out = []
    for _, log, part, position in columns:
        try:
            values = points[part]["values"][i if part == "incident" else j]
            out.append(_format(values[position], log))
        except (IndexError, TypeError, KeyError):
            out.append("?")
    return out


@functools.lru_cache(maxsize=4096)
def _ionisation_parameter(cloudy_dir, model):
    """A model's ionisation parameter, as written into its input."""
    try:
        with open(model_path(cloudy_dir, model) + ".yaml") as file:
            value = (yaml.safe_load(file) or {}).get("ionisation_parameter")
        return None if value is None else float(value)
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return None


def _iteration_text(cloudy_dir, model):
    if cloudy_dir is None:
        return "?"
    now, most = iteration(model_path(cloudy_dir, model))
    if now is None:
        return Text("–", style="yellow")
    return f"{now}/{most}" if most else str(now)


#: The columns of the rounds table
ROUND_COLUMNS = (
    "round",
    "limit",
    "workers",
    "done",
    "errored",
    "timed out",
    "retried",
    "failed",
    "left",
)


def _errored(r):
    """A round's runs that errored, hit the time limit or were lost."""
    return r.get("failed", 0) + r.get("stopped", 0) + r.get("lost", 0)


def _round_rows(rounds):
    """The rounds table's rows, newest first."""
    return [
        (
            str(r["round"]),
            [
                str(v)
                for v in (
                    f"{r['round']} perturb"
                    if r.get("perturb")
                    else r["round"],
                    r.get("walltime", ""),
                    r.get("workers", "here"),
                    r.get("done", 0),
                    _errored(r),
                    r.get("stopped", 0),
                    max(_errored(r) - r.get("given_up", 0), 0),
                    r.get("given_up", 0),
                    r["queue"]["todo"],
                )
            ],
        )
        for r in reversed(rounds)
    ]


def _job_status(snapshot, entry):
    """How a job is doing: live from squeue, else how it ended."""
    job, live = str(entry.get("job")), snapshot["slurm"]
    if job in live:
        now = live[job]
        if now["running"]:
            status = Text(f"{now['running']} running", style="cyan")
            if now["pending"]:
                status.append(f" · {now['pending']} queued", style="dim")
            return status
        return Text(f"{now['pending']} queued", style="yellow")
    if job in snapshot["ended"]:
        ended = snapshot["ended"][job]
        bad = set(ended) - {"COMPLETED"}
        return Text(
            " · ".join(
                f"{n} {s.lower()}" if sum(ended.values()) > 1 else s.lower()
                for s, n in sorted(ended.items())
            ),
            style="red"
            if bad - {"CANCELLED"}
            else ("yellow" if bad else "green"),
        )
    if "outcome" in entry:
        failed = entry["outcome"] != "completed"
        return Text(
            "failed" if failed else "done", style="red" if failed else "green"
        )
    return Text("submitted", style="dim")


def _job_rows(snapshot):
    """The jobs table's rows, newest first, each job named without its run's
    name (e.g. "R3 C1 Cloudy")."""
    run = f"-{snapshot['run'].get('name')}-"
    return [
        (
            str(entry.get("job")),
            [
                entry.get("name", entry.get("stage", ""))
                .replace(run, "-")
                .replace("-", " "),
                Text(str(entry.get("job")), style="dim"),
                _job_status(snapshot, entry),
            ],
        )
        for entry in reversed(snapshot["jobs"])
    ]


def _failure_rows(snapshot):
    """The failed models table's rows."""
    attempts = snapshot["attempts"]
    return [
        (
            model,
            [
                Text(model, style="red"),
                str(attempts.get(model, "?")),
                why or "",
            ],
        )
        for model, why in _failures(snapshot).items()
    ]


def _model_columns(snapshot, extra):
    """The columns of a table of models: model, grid point, log U, extra.

    Returns:
        (list of str, list, bool): the headers, the grid's columns, and
        whether a "log U" column (the reference model's U, not an axis) is
        shown.
    """
    columns = _point_columns(snapshot["points"])
    show_u = "log U" not in [c[0] for c in columns]
    headers = ["model", *(c[0] for c in columns)]
    headers += ["log U"] if show_u else []
    return headers + list(extra), columns, show_u


#: How many of the longest running models have their .out read for their
#: iteration each refresh
ITERATIONS_READ = 50


def _running_rows(snapshot):
    """The running models, longest first: grid point, iteration, worker, time.

    Those running more than twice this round's median are yellow, to spot
    one that hangs.

    Returns:
        (headers, rows, title)
    """
    running, cloudy_dir = snapshot["running"], snapshot["cloudy_dir"]
    times = [
        r["seconds"]
        for r in snapshot["records"]
        if r["result"] in ("done", "failed")
    ]
    slow = 2 * statistics.median(times) if times else None
    headers, columns, show_u = _model_columns(
        snapshot, ["iter", "worker", "time"]
    )
    rows = []
    for n, (model, owner, seconds) in enumerate(running):
        u = _ionisation_parameter(cloudy_dir, model) if cloudy_dir else None
        rows.append(
            (
                f"{model}@{owner}",
                [
                    model,
                    *_point(snapshot["points"], model, columns),
                    *([_format(u, True) if u else "?"] if show_u else []),
                    _iteration_text(cloudy_dir, model)
                    if n < ITERATIONS_READ
                    else "",
                    Text(owner, style="dim"),
                    Text(
                        _duration(seconds),
                        style="yellow" if slow and seconds > slow else "",
                    ),
                ],
            )
        )
    title = f"Running now ({len(running)}"
    title += f", yellow > 2 x median {_duration(slow / 2)})" if slow else ")"
    return headers, rows, title


def _squeue_rows(snapshot):
    """The Slurm queue view's rows: this run's jobs bold, states coloured.

    Returns:
        (rows, title)
    """
    jobs = snapshot["squeue"]
    if jobs is None:
        return [], "Slurm queue (no squeue here)"
    run = f"-{snapshot['run'].get('name')}-"
    colours = {"RUNNING": "cyan", "PENDING": "yellow", "COMPLETING": "green"}
    rows = []
    for n, job in enumerate(jobs):
        bold = " bold" if run in job[2] else ""
        cells = [Text(value, style=bold.strip()) for value in job]
        cells[3] = Text(job[3], style=colours.get(job[3], "red") + bold)
        rows.append((str(n), cells))
    running = sum(job[3] == "RUNNING" for job in jobs)
    title = (
        f"Slurm queue: {len(jobs)} jobs, {running} running "
        "(this run's in bold)"
    )
    return rows, title


#: The colour of each stage's tag in run.log
STAGE_COLOURS = {
    "run": "magenta",
    "resume": "magenta",
    "start": "bright_blue",
    "work": "cyan",
    "perturb": "bright_magenta",
    "finish": "green",
    "assemble": "yellow",
}


def _log_text(line):
    """A run.log line for the log panel: time dim, stage tag coloured."""
    # The date is today's; the time is enough
    if len(line) > 20 and line[4] == "-" and line[10] == " ":
        line = line[11:]
    match = re.match(r"(\d\d:\d\d:\d\d) \[([^\]]*)\] ?(.*)", line)
    if match is None:
        return Text(line)
    when, tag, message = match.groups()
    # The stage follows a section's label (e.g. "C2 finish 123")
    words = tag.split(" ")
    if SECTION_LABEL.fullmatch(words[0]) and len(words) > 1:
        words = words[1:]
    colour = STAGE_COLOURS.get(words[0], "white")
    if "ERROR" in message:
        message_style = "bold red"
    elif "WARNING" in message:
        message_style = "yellow"
    else:
        message_style = ""
    text = Text.assemble(
        (when, "dim"), " ", (f"[{tag}]", colour), " ", (message, message_style)
    )
    # Tables (the rounds in run.log) are cropped rather than wrapped
    if message.lstrip()[:1] in ("+", "|"):
        text.no_wrap, text.overflow = True, "crop"
    return text


def _plain(cell):
    return cell.plain if isinstance(cell, Text) else str(cell)


def _fill(table, headers, rows, right=(), stretch="even"):
    """Refill a DataTable across its width, keeping its scroll and cursor.

    Columns are as wide as their contents, and the room left over goes to
    the stretch columns (their indices, or "even" for every column), so the
    table fills its panel; with too little room, the stretch columns (if
    given) give it up, down to 8 characters, cropping what they hold.

    Args:
        headers (list of str)
            The column titles.
        rows (list of (key, cells))
            The rows.
        right (iterable of int)
            The columns to right align (numbers).
        stretch (list of int or "even")
            The columns given the room left over.
    """
    right = set(right)
    padding = table.cell_padding
    widths = [
        max([cell_len(str(h))] + [cell_len(_plain(c[i])) for _, c in rows])
        for i, h in enumerate(headers)
    ]
    # The room inside the border, less the scrollbar
    room = table.content_size.width - 1
    spare = room - sum(width + 2 * padding for width in widths)
    targets = list(range(len(widths)) if stretch == "even" else stretch)
    if table.content_size.width and spare > 0:
        for n, i in enumerate(targets):
            widths[i] += spare // len(targets) + (n < spare % len(targets))
    elif table.content_size.width and spare < 0 and stretch != "even":
        for i in targets:
            give = min(-spare, max(widths[i] - 8, 0))
            widths[i] -= give
            spare += give
    columns = list(table.columns.values())
    if [str(c.label) for c in columns] != list(headers) or [
        c.width for c in columns
    ] != widths:
        table.clear(columns=True)
        for i, (header, width) in enumerate(zip(headers, widths)):
            table.add_column(
                Text(header, justify="right" if i in right else "left"),
                width=width,
            )
    cursor, scroll = table.cursor_row, table.scroll_y
    table.clear()
    for key, cells in rows:
        cells = [
            Text(
                _plain(c),
                style=c.style if isinstance(c, Text) else "",
                justify="right",
            )
            if i in right
            else c
            for i, c in enumerate(cells)
        ]
        table.add_row(*cells, key=key)
    if rows:
        table.move_cursor(row=min(cursor, len(rows) - 1), scroll=False)
    table.scroll_to(y=scroll, animate=False)


def _table(name):
    """A table panel, by its name (its class, e.g. "jobs").

    The cursor row shows only while the table is focused (see WatchApp).
    """
    return DataTable(
        classes=name, zebra_stripes=True, cursor_type="row", show_cursor=False
    )


class _SlurmQueue(Screen):
    """Your Slurm queue, full screen; s or Esc goes back."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("s", "app.pop_screen", "Back"),
        ("q", "app.quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """The run's header and the queue."""
        yield Static(id="header")
        yield _table("view")
        yield Footer()

    def on_mount(self):
        """Ask for the queue while shown, and now rather than next redraw."""
        self.app.watcher.show_squeue = True
        self.app.read()
        if self.app.view is not None:
            self.show(self.app.view)

    def on_unmount(self):
        """Stop asking for the queue."""
        self.app.watcher.show_squeue = False

    def show(self, view):
        """Draw the header and the queue.

        Args:
            view (dict)
                What ``Watcher.read`` returns.
        """
        header = self.query_one("#header", Static)
        header.update(_header(view, header.content_size.width or 118))
        table = self.query_one(".view", DataTable)
        rows, title = _squeue_rows(view["sections"][0])
        table.border_title = title
        _fill(table, list(SQUEUE_COLUMNS), rows, right=[4, 5, 6])


#: The all grids page's columns (the names last, as they may not fit)
GRID_COLUMNS = (
    "#",
    "stage",
    "progress",
    "done",
    "running",
    "failed",
    "run",
    "grid",
)


#: The order of a section's stages, to sort the all grids page by stage
STAGE_ORDER = ("start", "finish", "assemble", "done")

#: The columns the all grids page sorts by most first (the rest A to Z)
MOST_FIRST = ("progress", "done", "running", "failed")


def _grid_row(snapshot):
    """A grid's cells in the all grids page (after its place), and what
    each sorts by.

    Args:
        snapshot (dict)
            The section's snapshot (or, from ``Watcher.brief``, its "label",
            "run", "state" and "counts").

    Returns:
        (list, list)
            Its stage, progress bar, models done (of all), running and
            failed, its run's name and its own; then each one's sort key.
    """
    state, label = snapshot["state"], snapshot["label"]
    counts = snapshot.get("counts") or {}
    total = sum(counts.values())
    stage, style = _badge(state, counts) if label else ("STARTING", "blue")
    done, running, failed = (
        counts.get(name, 0) for name in ("done", "claimed", "failed")
    )
    name = f"{label} {state.get('name', '')}" if label else ""
    run = snapshot["run"].get("name", "?")
    next_stage = state.get("next") if label else None
    cells = [
        Text(stage.lower(), style=style.split(" on ")[-1]),
        _bar(done / max(total, 1), 20) if total else Text(""),
        f"{done:,}/{total:,}",
        Text(f"{running:,}", style="cyan" if running else "dim"),
        Text(f"{failed:,}", style="bold red" if failed else "dim"),
        run,
        name,
    ]
    keys = [
        (
            STAGE_ORDER.index(next_stage) if next_stage in STAGE_ORDER else -1,
            state.get("round", 0),
        ),
        done / max(total, 1),
        done,
        running,
        failed,
        run,
        name,
    ]
    return cells, keys


class _AllGrids(Screen):
    """Every grid being run, full screen: the overview, then a row per grid,
    sorted by any column (o picks the next, O reverses, or click a column's
    title). Enter shows the grid; g or Esc goes back."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("g", "app.pop_screen", "Back"),
        ("o", "sort(1)", "Sort by next column"),
        ("O", "reverse", "Reverse"),
        ("q", "app.quit", "Quit"),
    ]

    def __init__(self):
        """The page, in place order at first."""
        super().__init__()
        self.sort = 0  # the column sorted by
        self.reverse = False  # whether the sort is reversed

    def compose(self) -> ComposeResult:
        """The title bar, the overview and the grids."""
        yield Static(id="title")
        yield Static(id="overview")
        yield _table("view")
        yield Footer()

    def on_mount(self):
        """Draw now rather than at the next redraw, and pick with the keys."""
        self.show()
        self.query_one(".view", DataTable).focus()

    def show(self):
        """Draw the title, the overview and every grid's row, sorted."""
        title = self.query_one("#title", Static)
        title.update(self.app.title_bar(title.content_size.width or 118))
        overview = self.query_one("#overview", Static)
        overview.update(self.app.overview(overview.content_size.width or 116))
        rows = self.app.grid_rows()
        column = GRID_COLUMNS[self.sort]
        rows.sort(
            key=lambda row: row[2][self.sort],
            reverse=(column in MOST_FIRST) != self.reverse,
        )
        table = self.query_one(".view", DataTable)
        order = "in order" if column == "#" else f"sorted by {column}"
        if self.reverse:
            order += ", reversed"
        table.border_title = (
            f"All grids ({len(rows)})  ·  {order}  ·  Enter shows one"
        )
        _fill(
            table,
            list(GRID_COLUMNS),
            [(key, cells) for key, cells, _ in rows],
            right=[0, 3, 4, 5],
            stretch=[7],
        )

    def action_sort(self, step):
        """Sort by the next column (from the first, its place, again).

        Args:
            step (int)
                How many columns along.
        """
        self.sort, self.reverse = (self.sort + step) % len(GRID_COLUMNS), False
        self.show()

    def action_reverse(self):
        """Reverse the sort."""
        self.reverse = not self.reverse
        self.show()

    def on_data_table_header_selected(self, event):
        """Sort by the column whose title was clicked (again to reverse)."""
        if event.column_index == self.sort:
            self.reverse = not self.reverse
        else:
            self.sort, self.reverse = event.column_index, False
        self.show()

    def on_data_table_row_selected(self, event):
        """Show the grid picked."""
        self.app.pop_screen()
        self.app.select(event.row_key.value)


class _SectionView(Vertical):
    """One Cloudy section's panels (the body of its tab)."""

    #: Each panel's title
    TITLES = {
        "queue": "Grid points",
        "round": "This round",
        "jobs": "Jobs",
        "rounds": "Rounds",
        "log": "run.log",
        "failed": "Failed models",
        "running": "Running now",
    }

    def compose(self) -> ComposeResult:
        """The section's card, then its panels: its queue, round and jobs,
        its rounds, then its log, failed and running models."""
        yield Static(classes="card")
        with Horizontal(classes="top"):
            yield Static(classes="queue")
            yield Static(classes="round")
            yield _table("jobs")
        yield _table("rounds")
        with Horizontal(classes="bottom"):
            yield RichLog(classes="log", wrap=True, max_lines=20000)
            with Vertical(classes="right"):
                yield _table("failed")
                yield _table("running")

    def on_mount(self):
        """Title the panels."""
        for name, title in self.TITLES.items():
            self.panel(name).border_title = title

    def panel(self, name):
        """One of the section's panels, by its name (e.g. "log")."""
        return self.query_one(f".{name}")

    def show(self, snapshot):
        """Draw a snapshot of the section on every panel.

        Args:
            snapshot (dict)
                The section's snapshot (see ``_SectionReader.read``).
        """
        counts = snapshot["counts"]
        card = self.panel("card")
        card.update(_card(snapshot, card.content_size.width or 118))
        self.panel("queue").update(_queue_table(snapshot))
        self.panel("round").update(_round_table(snapshot))
        _fill(
            self.panel("jobs"),
            ["job", "id", "status"],
            _job_rows(snapshot),
            right=[1],
            stretch=[0, 2],
        )
        rounds = _rounds(snapshot["state"], snapshot["records"], counts)
        _fill(
            self.panel("rounds"),
            list(ROUND_COLUMNS),
            _round_rows(rounds),
            right=range(len(ROUND_COLUMNS)),
        )
        failed = self.panel("failed")
        rows = _failure_rows(snapshot)
        failed.border_title = f"Failed models ({len(rows)})"
        _fill(
            failed,
            ["model", "attempts", "last reason"],
            rows,
            right=[1],
            stretch=[2],
        )
        running = self.panel("running")
        headers, rows, title = _running_rows(snapshot)
        running.border_title = title
        _fill(running, headers, rows, right=range(1, len(headers) - 2))

        # The log follows new lines, unless scrolled up to read
        log = self.panel("log")
        following = log.scroll_y >= log.max_scroll_y
        for line in snapshot["log_new"]:
            log.write(_log_text(line), scroll_end=following)


class WatchApp(App):
    """The live view of a run."""

    TITLE = "syncretize watch"
    # No panel is focused (and so shows a cursor row) until clicked
    AUTO_FOCUS = None
    CSS = """
    /* The title bar, then the tabs, then the overview */
    #title {
        height: 1;
        padding: 0 1;
        background: $boost;
    }
    #header, #overview {
        height: 4;
        border: heavy $accent;
        padding: 0 1;
    }
    .card {
        height: 4;
        border: round $secondary;
        padding: 0 1;
    }
    ContentSwitcher, _SectionView { height: 1fr; }
    .top { height: 9; }
    /* Every panel in a row fills it, so their borders line up */
    .top > *, .bottom > * { height: 100%; }
    .queue { width: 3fr; }
    .round { width: 3fr; }
    .jobs { width: 4fr; }
    .rounds { height: auto; max-height: 12; }
    .bottom { height: 1fr; }
    .log { width: 3fr; }
    .right { width: 2fr; }
    .failed { height: auto; max-height: 12; }
    .running { height: 1fr; }
    .view { height: 1fr; }
    .queue, .round, .jobs, .rounds, .log, .failed, .running, .view {
        border: round $primary 60%;
        border-title-align: left;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 0 1;
    }
    .jobs, .rounds, .failed, .running, .view { padding: 0; }
    DataTable:focus, RichLog:focus {
        border: round $accent;
    }
    /* Thin, quiet scrollbars, brighter when used */
    DataTable, RichLog {
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-color: $secondary 30%;
        scrollbar-color-hover: $secondary 60%;
        scrollbar-color-active: $accent;
        scrollbar-corner-color: $surface;
    }
    /* run.log wraps, so never needs to scroll sideways */
    RichLog { overflow-x: hidden; }
    /* Column titles: bold on a soft band, the same under the mouse */
    DataTable > .datatable--header,
    DataTable > .datatable--header-hover,
    DataTable > .datatable--header-cursor {
        background: $primary 25%;
        color: $text;
        text-style: bold;
    }
    /* Key bindings: each key a bright chip, its action plain text, on
       the screen's own background */
    Footer { background: $background; }
    FooterKey { background: transparent; }
    FooterKey .footer-key--key {
        background: $accent;
        color: $background;
        text-style: bold;
        padding: 0 1;
    }
    FooterKey .footer-key--description {
        background: transparent;
        color: $text;
        padding: 0 1;
    }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("l", "toggle_max('log')", "Log"),
        ("r", "toggle_max('running')", "Running"),
        ("s", "squeue", "Slurm queue"),
        ("g", "grids", "All grids"),
        *(
            Binding(str(n % 10), f"pick({n})", show=False)
            for n in range(1, 11)
        ),
    ]

    def __init__(self, work_dir=None, interval=2.0, directory=None):
        """Watch a run, or the runs under way under a directory.

        Args:
            work_dir (str)
                The run's work directory.
            interval (float)
                Seconds between redraws.
            directory (str)
                A directory holding runs, to watch those under way (see
                ``find_runs``) instead of one run.
        """
        super().__init__()
        self.interval, self.directory = interval, directory
        # Each run's watcher, by its work directory (the runs under way,
        # watching a directory)
        self.watchers = {} if directory else {work_dir: Watcher(work_dir)}
        self.views = {}  # each run's last read, by its work directory
        self.found = {}  # the runs by where they are (watching a directory)
        self.shown = None  # the work directory of the run shown
        # Each section's number (naming its view, "view-<n>"), by its run's
        # work directory and its label
        self._pane_ids = {}
        self._made = 0  # how many views have been made
        self._order = []  # the views' IDs, in order
        self._found_at = -math.inf  # when the directory was last searched
        self._drawing = False  # whether a read is being drawn
        self.sub_title = directory or work_dir

    def compose(self) -> ComposeResult:
        """The title bar, the shown grid's view, and the keys."""
        yield Static(id="title")
        yield ContentSwitcher(id="sections")
        yield Footer()

    def on_mount(self):
        """Keep the main page's widgets (other pages open over it), then read
        the runs now and every interval."""
        self._title = self.query_one("#title", Static)
        self._switcher = self.query_one("#sections", ContentSwitcher)
        self.read()
        self.set_interval(self.interval, self.read)

    @property
    def watcher(self):
        """The shown run's watcher (None while no run is under way)."""
        return self.watchers.get(self._shown())

    @property
    def view(self):
        """The shown run's last full read (None before the first)."""
        view = self.views.get(self._shown())
        return view if view is not None and "now" in view else None

    def _shown(self):
        """The shown run's work directory (the first run's, at first)."""
        if self.shown in self.watchers:
            return self.shown
        return next(iter(self.watchers), None)

    @work(thread=True, exclusive=True)
    def read(self):
        """Read the runs' files (away from the screen, which stays live): the
        shown run's in full, the others' state files and queues only."""
        if (
            self.directory is not None
            and time.monotonic() - self._found_at >= SLURM_EVERY
        ):
            self.found = find_runs(self.directory)
            self._found_at = time.monotonic()
            self.watchers = {
                work_dir: self.watchers.get(work_dir) or Watcher(work_dir)
                for work_dir in self.found["active"]
            }
        shown = self._shown()
        views = {
            work_dir: watcher.read() if work_dir == shown else watcher.brief()
            for work_dir, watcher in self.watchers.items()
        }
        self.call_from_thread(self.show, views)

    async def show(self, views):
        """Draw what was read: the title, then the shown grid's view (and the
        page open over it).

        Args:
            views (dict)
                Each run's read, by its work directory (see ``read``).
        """
        # A read that arrives while the last is being drawn (while new views
        # mount) is dropped; the next will follow it
        if self._drawing:
            return
        self._drawing = True
        try:
            await self._draw(views)
        finally:
            self._drawing = False

    async def _draw(self, views):
        """Draw what was read (see ``show``)."""
        self.views = views

        # Every run's sections, in order, each with a view (made as it
        # appears, dropped once its run is done or has started its sections)
        sections = [
            (work_dir, snapshot)
            for work_dir, view in views.items()
            for snapshot in view["sections"]
        ]
        keys = [(work_dir, s["label"]) for work_dir, s in sections]
        for key in list(self._pane_ids):
            if key not in keys:
                n = self._pane_ids.pop(key)
                await self._switcher.get_child_by_id(f"view-{n}").remove()
        for work_dir, snapshot in sections:
            key = (work_dir, snapshot["label"])
            if key not in self._pane_ids:
                self._made += 1
                self._pane_ids[key] = self._made
                await self._switcher.add_content(
                    _SectionView(id=f"view-{self._made}")
                )
            if "now" in snapshot:
                n = self._pane_ids[key]
                self._switcher.get_child_by_id(f"view-{n}").show(snapshot)
        self._order = [f"view-{self._pane_ids[key]}" for key in keys]
        if self._switcher.current not in self._order and self._order:
            self.select(self._order[0])

        self._title.update(
            self.title_bar(self._title.content_size.width or 118)
        )
        if isinstance(self.screen, _SlurmQueue) and self.view is not None:
            self.screen.show(self.view)
        if isinstance(self.screen, _AllGrids):
            self.screen.show()
        self.refresh_bindings()

    def title_bar(self, width):
        """The title bar: what is watched, which grid is shown, and the time.

        Args:
            width (int)
                Its width.

        Returns:
            Text
        """
        right = Text()
        if len(self._order) > 1 and self._switcher.current in self._order:
            place = self._order.index(self._switcher.current) + 1
            right.append(f"grid {place} of {len(self._order)}", style="bold")
            right.append("  ·  ", style="grey50")
        right.append(
            datetime.now(timezone.utc).strftime("%H:%M:%S UTC"), style="dim"
        )
        subject = self.directory
        if subject is None:
            subject = (self.view or {}).get("run", {}).get("name", "")
        left = Text.assemble(
            ("◆ syncretize watch", "bold magenta"), "   ", (subject, "bold")
        )
        return _spread(left, right, width)

    def overview(self, width):
        """Progress across every grid under the directory, or the run's.

        Args:
            width (int)
                Its width.

        Returns:
            Text
        """
        if self.directory is not None:
            return _overview(self.views, self.found, width)
        if self.view is not None:
            return _header(self.view, width)
        return Text()

    def select(self, view_id):
        """Show a grid's view, reading its run in full.

        Args:
            view_id (str)
                The view's ID, "view-<n>".
        """
        self._switcher.current = view_id
        n = int(view_id.removeprefix("view-"))
        for (work_dir, _), number in self._pane_ids.items():
            if number == n and work_dir != self.shown:
                self.shown = work_dir
                self.read()

    def grid_rows(self):
        """Every grid's row in the all grids page, in order.

        Returns:
            list of (str, list, list)
                Each grid's view ID, its cells and what each sorts by.
        """
        rows = []
        for work_dir, view in self.views.items():
            for snapshot in view["sections"]:
                n = self._pane_ids.get((work_dir, snapshot["label"]))
                if n is not None:
                    place = len(rows) + 1
                    cells, keys = _grid_row(snapshot)
                    rows.append(
                        (f"view-{n}", [str(place), *cells], [place, *keys])
                    )
        return rows

    def _current(self):
        """The section view shown, or None before the first read."""
        current = self._switcher.visible_content
        return current if isinstance(current, _SectionView) else None

    def on_descendant_focus(self, event):
        """Show the cursor row only in the table being scrolled."""
        if isinstance(event.widget, DataTable):
            event.widget.show_cursor = True

    def on_descendant_blur(self, event):
        """Hide the cursor row of a table no longer being scrolled."""
        if isinstance(event.widget, DataTable):
            event.widget.show_cursor = False

    def check_action(self, action, parameters):
        """Hide the main page's keys while another page is open, the grid
        keys with one grid (or none), and the Slurm queue's key with no
        run."""
        if isinstance(self.screen, (_SlurmQueue, _AllGrids)):
            return action not in ("toggle_max", "pick", "grids")
        if action in ("pick", "grids"):
            return len(self._pane_ids) > 1
        if action == "squeue":
            return self.watcher is not None
        return True

    def action_toggle_max(self, name):
        """Show a panel of the grid shown full screen (or go back if it is).

        Args:
            name (str)
                The panel's name, e.g. "log".
        """
        if self.screen.maximized is not None:
            self.screen.minimize()
        elif self._current() is not None:
            self.screen.maximize(self._current().panel(name), container=False)

    def action_pick(self, number):
        """Show a grid by its place (1 to 10, as in the all grids page).

        Args:
            number (int)
                The grid's place.
        """
        if number <= len(self._order):
            self.select(self._order[number - 1])

    def action_squeue(self):
        """Open the Slurm queue."""
        self.push_screen(_SlurmQueue())

    def action_grids(self):
        """Open the all grids page."""
        self.push_screen(_AllGrids())


def summary(view, log_lines=15):
    """A one-off view of the run (``syncretize watch --once``).

    Args:
        view (dict)
            What ``Watcher.read`` returns.
        log_lines (int)
            How many of run.log's last lines to show.

    Returns:
        Group
            The run's header, then each section's panels.
    """
    parts = [Panel(_header(view), box=box.HEAVY)]
    for snapshot in view["sections"]:
        parts += _section_summary(snapshot, log_lines)
    return Group(*parts)


def _section_summary(snapshot, log_lines):
    """A one-off view of one section, as panels.

    Args:
        snapshot (dict)
            The section's snapshot.
        log_lines (int)
            How many of its run.log lines to show.

    Returns:
        list of Panel
    """
    counts = snapshot["counts"]
    rounds = Table(box=box.SIMPLE_HEAD)
    for header in ROUND_COLUMNS:
        rounds.add_column(header, justify="right")
    for _, cells in _round_rows(
        _rounds(snapshot["state"], snapshot["records"], counts)
    ):
        rounds.add_row(*cells)
    failed = Table(box=box.SIMPLE_HEAD)
    for header in ("model", "attempts", "last reason"):
        failed.add_column(header)
    for _, cells in _failure_rows(snapshot):
        failed.add_row(*cells)
    headers, rows, title = _running_rows(snapshot)
    running = Table(box=box.SIMPLE_HEAD)
    for header in headers:
        running.add_column(header)
    for _, cells in rows[:10]:
        running.add_row(*cells)
    log = Text("\n").join(
        _log_text(line) for line in snapshot["log"][-log_lines:]
    )
    name = _tab_title(snapshot) if snapshot["label"] else "run"
    return [
        Columns(
            [
                Panel(_queue_table(snapshot), title=f"{name}: grid points"),
                Panel(_round_table(snapshot), title="This round"),
            ]
        ),
        Panel(rounds, title="Rounds"),
        Panel(failed, title=f"Failed models ({len(_failures(snapshot))})"),
        Panel(running, title=title),
        Panel(log, title="run.log"),
    ]


def watch(path=None, interval=2.0, once=False):
    """Show a run's state, redrawn every interval seconds until q.

    Args:
        path (str)
            The run (see ``find_work_dir``), or a directory holding runs to
            watch every one under way (see ``find_runs``).
        interval (float)
            Seconds between redraws.
        once (bool)
            Print a summary once and return (e.g. in scripts).
    """
    try:
        work_dir, directory = find_work_dir(path), None
    except ValueError:
        # Not a run, so a directory holding runs
        work_dir, directory = None, os.path.abspath(path or os.getcwd())
    if once:
        runs = (
            [work_dir] if directory is None else find_runs(directory)["active"]
        )
        for run in runs:
            Console().print(summary(Watcher(run).read()))
        return
    WatchApp(work_dir, interval, directory=directory).run()
