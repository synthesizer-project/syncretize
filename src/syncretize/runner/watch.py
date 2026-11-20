"""A live, full-terminal view of a run: ``syncretize watch [run]``.

A Textual app that redraws every few seconds (``-n``, default 2) from the
files a run writes, without changing any of them, so it is safe to leave
running anywhere the work directory is visible:

    state.yaml       next stage and job, rounds, jobs, failed models
    queue/*          how many models are waiting, running, done and failed,
                     and how long each running model has been going
    state/*.jsonl    the current round's worker records (read as they grow)
    grid_points.json each model's grid point (from the setup)
    run.log          as it grows
    squeue, sacct    the run's jobs (every 30 s, when Slurm is there)

Every panel scrolls (keys or mouse); Tab moves between them. run.log follows
new lines unless scrolled up. Keys: f the failed models and l run.log full
screen (again to go back), m a map of the grid coloured by each point's
state, s the slowest models so far (Esc to go back), q quits.
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
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, RichLog, Static

from syncretize.runner.cloudy import iteration, model_path
from syncretize.runner.perturb import is_perturbed
from syncretize.runner.queue import STATES

#: How often squeue is asked about the run's jobs (s)
SLURM_EVERY = 30

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


class Watcher:
    """Reads a run's files, keeping what is slow to read between redraws.

    Attributes:
        work_dir (str)
            The run's work directory.
    """

    def __init__(self, work_dir):
        self.work_dir = work_dir
        self._records = {}  # worker record file: (offset, records)
        self._counts, self._counted = None, 0.0
        self._jobs, self._asked = {}, 0.0
        self._ended = {}  # job ID: how its tasks ended, from sacct
        self._running_cache, self._running_at = [], 0.0
        self._models, self._models_mtime = {}, None
        self._finished = []  # every finished attempt, from models.json
        self._names = {state: [] for state in STATES}  # the queue's models
        self._grid_points = None
        self._log_offset = None  # how much of run.log has been read

    def read(self):
        """Everything the view shows, as a dict."""
        state = self._yaml("state.yaml")
        return {
            "state": state,
            "counts": self._queue_counts(),
            "records": self._round_records(),
            "log": self._log_tail(),
            "log_new": self._log_new(),
            "slurm": self._slurm(state),
            "ended": self._ended,
            "attempts": self._attempts(),
            "finished": self._finished,
            "running": self._running(),
            "names": self._names,
            "points": self._points(state.get("cloudy_dir")),
            "cloudy_dir": state.get("cloudy_dir"),
            "now": datetime.now(timezone.utc),
        }

    def _path(self, *names):
        return os.path.join(self.work_dir, *names)

    def _yaml(self, name):
        try:
            with open(self._path(name)) as file:
                return yaml.safe_load(file) or {}
        except (FileNotFoundError, yaml.YAMLError):
            return {}

    def _queue_counts(self):
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

    def _log_tail(self, size=65536):
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

    def _slurm(self, state):
        """The run's queued and running jobs, by job ID (every 30 s).

        Also asks sacct how the run's other jobs ended, once each (a job that
        has ended doesn't change), into ``self._ended``.
        """
        name = state.get("name")
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
        jobs = {}
        for line in result.stdout.splitlines():
            job, job_name, status, elapsed = (line.split("|") + [""] * 4)[:4]
            if f"-{name}-" not in f"-{job_name}-":
                continue
            base = job.split("_")[0]
            entry = jobs.setdefault(
                base, {"name": job_name, "running": 0, "pending": 0}
            )
            tasks = _count_tasks(job)
            if status == "RUNNING":
                entry["running"] += tasks
                entry["elapsed"] = elapsed
            else:
                entry["pending"] += tasks
        self._jobs = jobs

        ask = [
            str(entry["job"])
            for entry in state.get("jobs", [])
            if str(entry["job"]) not in jobs
            and str(entry["job"]) not in self._ended
        ]
        if ask and shutil.which("sacct") is not None:
            self._ended |= _sacct(ask)
        return jobs


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
    "finish": ("ROUND {round} RUNNING", "bold black on bright_cyan"),
    "assemble": ("ASSEMBLING", "bold black on yellow"),
    "done": ("DONE", "bold black on green"),
}

#: The badge while failed models' perturbed runs are under way
PERTURB_BADGE = ("PERTURBING (ROUND {round})", "bold black on bright_magenta")


def _header(snapshot, width=118):
    """The header: the run and its stage, then its progress across the grid.

    Built as two lines of the given width (Textual lays nested rich tables
    out unreliably).
    """
    state, counts = snapshot["state"], snapshot["counts"]
    points, interpolated, _ = _grid_points(snapshot)
    total = sum(points.values()) + interpolated
    done, failed = points["done"], points["failed"]
    stage = state.get("next", "?")
    label, style = STAGE_BADGES.get(stage, (stage.upper(), "bold reverse"))
    if stage == "finish" and state.get("perturbing"):
        label, style = PERTURB_BADGE
    label = label.format(round=state.get("round", "?"))

    left = Text.assemble(
        ("◆ syncretize ", "bold magenta"),
        (state.get("name", "?"), "bold"),
        "   ",
        (f" {label} ", style),
    )
    right = Text.assemble(
        ("round ", "dim"),
        str(state.get("round", "-")),
        ("   limit ", "dim"),
        str(_limit(state)),
        ("   next job ", "dim"),
        str(state.get("next_job", "-")),
        "   ",
        (snapshot["now"].strftime("%H:%M:%S UTC"), "dim"),
    )
    top = left + " " * max(width - left.cell_len - right.cell_len, 1) + right

    summary = Text.assemble(
        " ",
        (f"{done:,}", "bold green"),
        f"/{total:,} grid points ",
        (f"({100 * done / max(total, 1):.1f}%)", "dim"),
        ("  ·  ", "dim"),
        (f"{failed:,} failed", "bold red" if failed else "dim"),
    )
    if interpolated:
        summary.append("  ·  ", style="dim")
        summary.append(f"{interpolated:,} interpolated", style="bold yellow")
    rate = _rate(snapshot["records"])
    left_to_run = counts["todo"] + counts["claimed"]
    if rate and left_to_run:
        summary.append("  ·  ", style="dim")
        summary.append(
            f"{rate:.1f}/min, ETA ~{_duration(left_to_run / rate * 60)}"
        )
    bar_width = max(width - summary.cell_len, 10)
    filled = round(bar_width * (done + failed + interpolated) / max(total, 1))
    bar = Text("━" * filled, style="green") + Text(
        "━" * (bar_width - filled), style="grey30"
    )
    return Text("\n").join([top, bar + summary])


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
    retried = sum(
        r["result"] == "failed" and r.get("requeued") for r in records
    )
    errors = results.count("failed")
    table.add_row(
        "errors",
        Text.assemble(
            (f"{errors}", "bold red" if errors else "dim"),
            (f"  {retried} retried" if errors else "", "dim"),
        ),
    )
    stopped = results.count("stopped")
    table.add_row(
        "stopped",
        Text(f"{stopped}", style="bold yellow" if stopped else "dim"),
    )
    times = [
        r["seconds"] for r in records if r["result"] in ("done", "failed")
    ]
    if times:
        table.add_row(
            "run times",
            Text.assemble(
                ("median ", "dim"),
                _duration(statistics.median(times)),
                ("  max ", "dim"),
                _duration(max(times)),
            ),
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
        live["retried"] = sum(
            r["result"] == "failed" and r.get("requeued") for r in records
        )
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


#: The map's shades, by the fraction of a cell's models done
SHADES = " ░▒▓█"


def _map_text(snapshot, width, height):
    """A map of the grid's first two incident axes, coloured by state.

    Each cell holds every model at that point (all the other axes): its
    shade is the fraction done, red if any failed, cyan if any running.
    """
    points = snapshot["points"]
    if points is None:
        return Text(
            "No grid_points.json: the run was set up before the setup wrote "
            "it",
            style="dim",
        )
    incident = points["incident"]
    axes, indices = incident["axes"], incident["indices"]
    nx = max(index[0] for index in indices) + 1
    ny = max(index[1] for index in indices) + 1 if len(axes) > 1 else 1

    # Values along each axis, for the labels
    along = [{}, {}]
    for index, values in zip(indices, incident["values"]):
        for k in range(min(2, len(axes))):
            along[k][index[k]] = values[k]
    labels = [AXIS_LABELS.get(axis, (axis, False)) for axis in axes[:2]]

    label_width = 8
    columns = max(width - label_width - 2, 1)
    bx = max(1, math.ceil(nx / columns))  # grid points per cell, across
    by = max(1, math.ceil(ny / max(height - 6, 1)))  # and down
    # As wide as fits (a small grid gets big cells)
    cell = max(1, min(columns // math.ceil(nx / bx), 6))

    cells = {}
    for state, names in snapshot["names"].items():
        for model in names:
            try:
                i = int(model.split("_")[0])
                index = indices[i]
            except (ValueError, IndexError):
                continue
            key = (index[0] // bx, (index[1] if ny > 1 else 0) // by)
            tally = cells.setdefault(key, dict.fromkeys(STATES, 0))
            tally[state] += 1

    lines = Text()
    for row in range(math.ceil(ny / by)):
        value = along[1].get(row * by) if ny > 1 else None
        label = (
            _format(value, labels[1][1]) if value is not None else ""
        ).rjust(label_width)
        lines.append(label + " ", style="dim")
        for col in range(math.ceil(nx / bx)):
            tally = cells.get((col, row))
            if not tally or not sum(tally.values()):
                lines.append(" " * cell)
                continue
            total = sum(tally.values())
            shade = SHADES[round(4 * tally["done"] / total)]
            if shade == " ":
                shade = "·"
            style = (
                "red"
                if tally["failed"]
                else "cyan"
                if tally["claimed"]
                else "green"
                if tally["done"] == total
                else "white"
            )
            lines.append(shade * cell, style=style)
        lines.append("\n")
    first, last = along[0].get(0), along[0].get(nx - 1)
    axis_line = (
        f"{_format(first, labels[0][1])} … {_format(last, labels[0][1])}"
        if first is not None
        else ""
    )
    lines.append(" " * (label_width + 1) + axis_line + "\n", style="dim")
    lines.append(
        f"across: {labels[0][0]}"
        + (f" · down: {labels[1][0]}" if ny > 1 else "")
        + f" · {bx * by} point(s) a cell · shade: fraction done"
        " · green all done · cyan running · red failed\n",
        style="dim",
    )
    return lines


#: The columns of the rounds table
ROUND_COLUMNS = (
    "round",
    "limit",
    "workers",
    "done",
    "errors",
    "retried",
    "stopped",
    "lost",
    "failed",
    "left",
)


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
                    r.get("failed", 0),
                    r.get("retried", 0),
                    r.get("stopped", 0),
                    r.get("lost", 0),
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
    """The jobs table's rows, newest first."""
    return [
        (
            str(entry.get("job")),
            [
                entry.get("name", entry.get("stage", "")),
                Text(str(entry.get("job")), style="dim"),
                _job_status(snapshot, entry),
            ],
        )
        for entry in reversed(snapshot["state"].get("jobs", []))
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


#: The most rows the slowest view lists
SLOWEST_ROWS = 2000


def _slowest_rows(snapshot):
    """The slowest Cloudy runs so far, with their grid points.

    Returns:
        (headers, rows, title)
    """
    finished = snapshot["finished"] + [
        r for r in snapshot["records"] if r["result"] in ("done", "failed")
    ]
    finished.sort(key=lambda r: -r.get("seconds", 0))
    headers, columns, show_u = _model_columns(
        snapshot, ["iterations", "result", "worker", "time"]
    )
    rows = []
    for n, record in enumerate(finished[:SLOWEST_ROWS]):
        u = record.get("ionisation_parameter")
        iterations = record.get("iterations")
        rows.append(
            (
                str(n),
                [
                    record["model"],
                    *_point(snapshot["points"], record["model"], columns),
                    *([_format(u, True) if u else "?"] if show_u else []),
                    "?" if iterations is None else str(iterations),
                    Text(
                        record["result"],
                        style="green" if record["result"] == "done" else "red",
                    ),
                    Text(record.get("owner", ""), style="dim"),
                    _duration(record.get("seconds", 0)),
                ],
            )
        )
    title = f"Slowest Cloudy runs ({len(finished)} finished)"
    return headers, rows, title


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
    colour = STAGE_COLOURS.get(tag.split(" ")[0], "white")
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
    table fills its panel.

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
    if table.content_size.width and spare > 0:
        targets = list(range(len(widths)) if stretch == "even" else stretch)
        for n, i in enumerate(targets):
            widths[i] += spare // len(targets) + (n < spare % len(targets))
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


def _table(id):
    # The cursor shows only while the table is focused (see WatchApp)
    return DataTable(
        id=id, zebra_stripes=True, cursor_type="row", show_cursor=False
    )


class _FullScreen(Screen):
    """A full-screen view (the map or the slowest runs); Esc goes back."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("q", "app.quit", "Quit"),
    ]

    def __init__(self, key):
        super().__init__()
        self.key = key

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        if self.key == "map":
            with VerticalScroll(id="view"):
                yield Static(id="map")
        else:
            yield _table("view")
        yield Footer()

    def on_mount(self):
        if self.app.snapshot is not None:
            self.show(self.app.snapshot)

    def show(self, snapshot):
        header = self.query_one("#header", Static)
        header.update(_header(snapshot, header.content_size.width or 118))
        view = self.query_one("#view")
        if self.key == "map":
            view.border_title = "Grid map"
            self.query_one("#map", Static).update(
                _map_text(snapshot, self.size.width - 4, self.size.height - 8)
            )
        else:
            headers, rows, title = _slowest_rows(snapshot)
            view.border_title = title
            _fill(view, headers, rows, right=range(1, len(headers) - 3))


class WatchApp(App):
    """The live view of a run."""

    TITLE = "syncretize watch"
    # No panel is focused (and so shows a cursor row) until Tab or a click
    AUTO_FOCUS = None
    CSS = """
    #header {
        height: 4;
        border: heavy $accent;
        padding: 0 1;
    }
    #top { height: 9; }
    /* Every panel in a row fills it, so their borders line up */
    #top > *, #bottom > * { height: 100%; }
    #queue { width: 3fr; }
    #round { width: 3fr; }
    #jobs { width: 4fr; }
    #rounds { height: auto; max-height: 12; }
    #bottom { height: 1fr; }
    #log { width: 3fr; }
    #right { width: 2fr; }
    #failed { height: auto; max-height: 12; }
    #running { height: 1fr; }
    #view { height: 1fr; }
    #queue, #round, #jobs, #rounds, #log, #failed, #running, #view {
        border: round $primary 60%;
        border-title-align: left;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 0 1;
    }
    #jobs, #rounds, #failed, #running, #view { padding: 0; }
    DataTable:focus, RichLog:focus, VerticalScroll:focus {
        border: round $accent;
    }
    /* Thin, quiet scrollbars, brighter when used */
    DataTable, RichLog, VerticalScroll {
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
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("f", "toggle_max('failed')", "Failed"),
        ("l", "toggle_max('log')", "Log"),
        ("m", "view('map')", "Map"),
        ("s", "view('slowest')", "Slowest"),
    ]

    def __init__(self, work_dir, interval=2.0):
        super().__init__()
        self.work_dir, self.interval = work_dir, interval
        self.watcher = Watcher(work_dir)
        self.snapshot = None
        self.sub_title = work_dir

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with Horizontal(id="top"):
            yield Static(id="queue")
            yield Static(id="round")
            yield _table("jobs")
        yield _table("rounds")
        with Horizontal(id="bottom"):
            yield RichLog(id="log", wrap=True, max_lines=20000)
            with Vertical(id="right"):
                yield _table("failed")
                yield _table("running")
        yield Footer()

    def on_mount(self):
        titles = {
            "queue": "Grid points",
            "round": "This round",
            "jobs": "Jobs",
            "rounds": "Rounds",
            "log": "run.log",
            "failed": "Failed models",
            "running": "Running now",
        }
        for id, title in titles.items():
            self.query_one(f"#{id}").border_title = title
        self.read()
        self.set_interval(self.interval, self.read)

    @work(thread=True, exclusive=True)
    def read(self):
        """Read the run's files (away from the screen, which stays live)."""
        snapshot = self.watcher.read()
        self.call_from_thread(self.show, snapshot)

    def show(self, snapshot):
        """Draw a snapshot of the run on every panel."""
        self.snapshot = snapshot
        counts = snapshot["counts"]
        header = self.query_one("#header", Static)
        header.update(_header(snapshot, header.content_size.width or 118))
        self.query_one("#queue", Static).update(_queue_table(snapshot))
        self.query_one("#round", Static).update(_round_table(snapshot))
        _fill(
            self.query_one("#jobs", DataTable),
            ["job", "id", "status"],
            _job_rows(snapshot),
            right=[1],
            stretch=[0, 2],
        )
        rounds = _rounds(snapshot["state"], snapshot["records"], counts)
        _fill(
            self.query_one("#rounds", DataTable),
            list(ROUND_COLUMNS),
            _round_rows(rounds),
            right=range(len(ROUND_COLUMNS)),
        )
        failed = self.query_one("#failed", DataTable)
        rows = _failure_rows(snapshot)
        failed.border_title = f"Failed models ({len(rows)})"
        _fill(
            failed,
            ["model", "attempts", "last reason"],
            rows,
            right=[1],
            stretch=[2],
        )
        running = self.query_one("#running", DataTable)
        headers, rows, title = _running_rows(snapshot)
        running.border_title = title
        _fill(running, headers, rows, right=range(1, len(headers) - 2))

        # The log follows new lines, unless scrolled up to read
        log = self.query_one("#log", RichLog)
        following = log.scroll_y >= log.max_scroll_y
        for line in snapshot["log_new"]:
            log.write(_log_text(line), scroll_end=following)

        if isinstance(self.screen, _FullScreen):
            self.screen.show(snapshot)

    def on_descendant_focus(self, event):
        """Show the cursor row only in the table being scrolled."""
        if isinstance(event.widget, DataTable):
            event.widget.show_cursor = True

    def on_descendant_blur(self, event):
        if isinstance(event.widget, DataTable):
            event.widget.show_cursor = False

    def check_action(self, action, parameters):
        """Hide the main view's keys while the map or slowest view is open."""
        if action == "toggle_max" and isinstance(self.screen, _FullScreen):
            return False
        return True

    def action_toggle_max(self, id):
        """Show a panel full screen (or go back if it is)."""
        if isinstance(self.screen, _FullScreen):
            return
        if self.screen.maximized is not None:
            self.screen.minimize()
        else:
            self.screen.maximize(self.query_one(f"#{id}"))

    def action_view(self, key):
        """Open the map or slowest view (or go back if it is open)."""
        current = getattr(self.screen, "key", None)
        if current is not None:
            self.pop_screen()
        if current != key:
            self.push_screen(_FullScreen(key))


def summary(snapshot, log_lines=15):
    """A one-off view of the run (``syncretize watch --once``)."""
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
    return Group(
        Panel(_header(snapshot), box=box.HEAVY),
        Columns(
            [
                Panel(_queue_table(snapshot), title="Grid points"),
                Panel(_round_table(snapshot), title="This round"),
            ]
        ),
        Panel(rounds, title="Rounds"),
        Panel(failed, title=f"Failed models ({len(_failures(snapshot))})"),
        Panel(running, title=title),
        Panel(log, title="run.log"),
    )


def watch(path=None, interval=2.0, once=False):
    """Show a run's state, redrawn every interval seconds until q.

    Args:
        path (str)
            The run (see ``find_work_dir``).
        interval (float)
            Seconds between redraws.
        once (bool)
            Print a summary once and return (e.g. in scripts).
    """
    work_dir = find_work_dir(path)
    if once:
        Console().print(summary(Watcher(work_dir).read()))
        return
    WatchApp(work_dir, interval).run()
