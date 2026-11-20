"""A live, full-terminal view of a run: ``syncretize watch [run]``.

Like ``watch``, it redraws every few seconds (``-n``, default 2) from the files
a run writes, without changing any of them, so it is safe to leave running
anywhere the work directory is visible:

    state.yaml       next stage and job, rounds, jobs, failed models
    queue/*          how many models are waiting, running, done and failed,
                     and how long each running model has been going
    state/*.jsonl    the current round's worker records (read as they grow)
    run.log          its last lines
    squeue           the run's jobs (every 30 s, when Slurm is there)

Keys: q quits, and each of these toggles a full-screen view: f the failed
models (those that failed every attempt), l run.log, m a map of the grid
coloured by each point's state, s the slowest models so far.
"""

import functools
import getpass
import json
import math
import os
import select
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml
from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from syncretize.runner.cloudy import iteration, model_path
from syncretize.runner.queue import STATES

#: The header's height (a panel of two lines)
HEADER = 4

#: Measures how many rows a wrapped line takes
_MEASURE = Console(width=1000)

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

    def read(self):
        """Everything the view shows, as a dict."""
        state = self._yaml("state.yaml")
        return {
            "state": state,
            "counts": self._queue_counts(),
            "records": self._round_records(),
            "log": self._log_tail(),
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


def render(snapshot, width, height, view="main"):
    """The whole screen for a snapshot of the run.

    Args:
        snapshot (dict)
            From ``Watcher.read``.
        width, height (int)
            The terminal's size, which the view fills.
        view (str)
            "main", or a full-screen "failures", "log", "map" or "slowest".

    Returns:
        rich.layout.Layout
    """
    state, counts = snapshot["state"], snapshot["counts"]
    layout = Layout()
    header = _header(snapshot)
    footer = Text(
        f" q quit · f failed · l run.log · m map · s slowest · every "
        f"{snapshot['interval']:g}s"
        f" · {snapshot['work_dir']}",
        style="dim",
        no_wrap=True,
        overflow="ellipsis",
    )

    if view != "main":
        rows = height - HEADER - 1 - 2
        body = {
            "log": lambda: _log_panel(snapshot["log"], rows, width - 4),
            "failures": lambda: _failures_panel(snapshot, rows, full=True),
            "map": lambda: _map_panel(snapshot, width - 4, rows),
            "slowest": lambda: _slowest_panel(snapshot, rows - 1),
        }[view]()
        layout.split_column(
            Layout(header, size=HEADER), Layout(body), Layout(footer, size=1)
        )
        return layout

    rounds = _rounds(state, snapshot["records"], counts)
    rounds_height = min(len(rounds), 8) + 4 if rounds else 3
    middle_height = 8
    bottom_height = height - HEADER - middle_height - rounds_height - 1
    middle = Layout()
    middle.split_row(
        Layout(_queue_panel(counts), ratio=2),
        Layout(_round_panel(snapshot), ratio=3),
        Layout(_jobs_panel(snapshot, middle_height - 2), ratio=4),
    )
    # run.log on the left; on the right, the failed models (as many rows as
    # they need, up to 8) above the models running longest
    failures = _failures(snapshot)
    failures_height = min(max(len(failures), 1), 8) + 2
    if len(failures) > 8:
        failures_height += 1  # "and N more"
    right = Layout()
    right.split_column(
        Layout(
            _failures_panel(snapshot, failures_height - 2),
            size=failures_height,
        ),
        Layout(_running_panel(snapshot, bottom_height - failures_height - 2)),
    )
    bottom = Layout()
    bottom.split_row(
        Layout(
            _log_panel(snapshot["log"], bottom_height - 2, width * 3 // 5 - 4),
            ratio=3,
        ),
        Layout(right, ratio=2),
    )
    layout.split_column(
        Layout(header, size=HEADER),
        Layout(middle, size=middle_height),
        Layout(_rounds_panel(rounds), size=rounds_height),
        Layout(bottom),
        Layout(footer, size=1),
    )
    return layout


def _header(snapshot):
    state, counts = snapshot["state"], snapshot["counts"]
    total = sum(counts.values())
    done, failed = counts["done"], counts["failed"]
    stage = state.get("next", "?")
    title = Text.assemble(
        (" syncretize ", "bold white on blue"),
        f" {state.get('name', '?')} · ",
        f"round {state.get('round', '-')} (limit {_limit(state)}) · ",
        ("next: ", "dim"),
        (f"{stage}", "bold green" if stage == "done" else "bold yellow"),
        f" {state.get('next_job', '')} · ",
        snapshot["now"].strftime("%H:%M:%S UTC"),
    )
    rate = _rate(snapshot["records"])
    left = counts["todo"] + counts["claimed"]
    eta = (
        f" · {rate:.1f} done/min · ETA ~{_duration(left / rate * 60)}"
        if rate and left
        else ""
    )
    progress = Table.grid(expand=True)
    progress.add_column(ratio=1)
    progress.add_column(no_wrap=True)
    progress.add_row(
        ProgressBar(total=max(total, 1), completed=done + failed),
        f" {done}/{total} done ({100 * done / max(total, 1):.1f}%) · "
        f"{failed} failed{eta}",
    )
    return Panel(Group(title, progress), box=box.HEAVY, padding=(0, 1))


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


def _queue_panel(counts):
    table = Table.grid(padding=(0, 2))
    table.add_column()
    table.add_column(justify="right")
    styles = {"todo": "", "claimed": "cyan", "done": "green", "failed": "red"}
    for state in STATES:
        table.add_row(
            Text(state, style=styles[state]), f"{counts.get(state, 0):,}"
        )
    return Panel(table, title="Queue", title_align="left")


def _round_panel(snapshot):
    records = snapshot["records"]
    results = [r["result"] for r in records]
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    workers = [
        job
        for job in snapshot["slurm"].values()
        if job["name"].endswith("-Cloudy")
    ]
    if workers:
        running = sum(job["running"] for job in workers)
        pending = sum(job["pending"] for job in workers)
        table.add_row("workers", f"{running} running · {pending} queued")
    table.add_row(
        "finished",
        f"{len(records)} ({_rate(records):.1f}/min)" if records else "0",
    )
    retried = sum(
        r["result"] == "failed" and r.get("requeued") for r in records
    )
    table.add_row("errors", f"{results.count('failed')} ({retried} retried)")
    table.add_row("stopped", str(results.count("stopped")))
    times = [
        r["seconds"] for r in records if r["result"] in ("done", "failed")
    ]
    if times:
        table.add_row(
            "run times",
            f"median {_duration(statistics.median(times))} · "
            f"max {_duration(max(times))}",
        )
    return Panel(table, title="This round", title_align="left")


def _jobs_panel(snapshot, rows):
    state, live = snapshot["state"], snapshot["slurm"]
    table = Table(box=None, padding=(0, 1), expand=True, show_header=False)
    table.add_column(no_wrap=True, overflow="ellipsis")
    table.add_column(justify="right", style="dim")
    table.add_column(no_wrap=True)
    if not state.get("jobs"):
        table.add_row(Text("none (a local run)", style="dim"), "", "")
    for entry in reversed(state.get("jobs", [])[-rows:]):
        job = str(entry.get("job"))
        name = entry.get("name", entry.get("stage", ""))
        if job in live:
            now = live[job]
            if now["running"]:
                status = Text(f"{now['running']} running", style="cyan")
                if now["pending"]:
                    status.append(f" · {now['pending']} queued", style="dim")
            else:
                status = Text(f"{now['pending']} queued", style="yellow")
        elif job in snapshot["ended"]:
            ended = snapshot["ended"][job]
            bad = set(ended) - {"COMPLETED"}
            status = Text(
                " · ".join(
                    f"{n} {s.lower()}"
                    if sum(ended.values()) > 1
                    else s.lower()
                    for s, n in sorted(ended.items())
                ),
                style="red"
                if bad - {"CANCELLED"}
                else ("yellow" if bad else "green"),
            )
        elif "outcome" in entry:
            failed = entry["outcome"] != "completed"
            status = Text(
                "failed" if failed else "done",
                style="red" if failed else "green",
            )
        else:
            status = Text("submitted", style="dim")
        table.add_row(name, job, status)
    return Panel(table, title="Jobs", title_align="left")


def _rounds(state, records, counts):
    """The finished rounds, and the one under way from its workers' records."""
    rounds = [r for r in state.get("rounds", []) if "queue" in r]
    current = state.get("round")
    if state.get("next") == "finish" and current is not None:
        submitted = next(
            (r for r in state.get("rounds", []) if r["round"] == current), {}
        )
        live = {"round": f"{current} ▸", "queue": counts}
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


def _rounds_panel(rounds):
    table = Table(
        box=box.SIMPLE_HEAD, expand=True, padding=(0, 1), show_edge=False
    )
    for header in (
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
    ):
        table.add_column(header, justify="right")
    for r in rounds[-8:]:
        table.add_row(
            *(
                str(v)
                for v in (
                    r["round"],
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
            )
        )
    return Panel(table, title="Rounds", title_align="left", padding=0)


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
    """A model's grid point, as text for each column."""
    i, j = (int(index) for index in model.split("_"))
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


def _models_table(snapshot, rows, extra_headers):
    """A table of models with their grid points, and columns to add.

    Returns:
        (Table, columns, show_u): the table, the grid's columns, and whether
        a "log U" column (the reference model's U, not an axis) is shown.
    """
    points = snapshot["points"]
    columns = _point_columns(points)
    show_u = "log U" not in [c[0] for c in columns]
    # Columns as wide as their contents, so none is squeezed
    table = Table(box=None, padding=(0, 1))
    table.add_column("model", no_wrap=True, min_width=5)
    for label, *_ in columns:
        table.add_column(label, justify="right", no_wrap=True)
    if show_u:
        table.add_column("log U", justify="right", no_wrap=True)
    for header in extra_headers:
        table.add_column(header, justify="right", no_wrap=True)
    return table, columns, show_u


def _iteration_text(cloudy_dir, model):
    if cloudy_dir is None:
        return "?"
    now, most = iteration(model_path(cloudy_dir, model))
    if now is None:
        return Text("–", style="yellow")
    return f"{now}/{most}" if most else str(now)


def _running_panel(snapshot, rows):
    """The models running longest: grid point, iteration, worker, time.

    Those running more than twice this round's median are yellow, to spot
    one that hangs. Only the rows shown read their .out for the iteration.
    """
    running, cloudy_dir = snapshot["running"], snapshot["cloudy_dir"]
    times = [
        r["seconds"]
        for r in snapshot["records"]
        if r["result"] in ("done", "failed")
    ]
    slow = 2 * statistics.median(times) if times else None
    table, columns, show_u = _models_table(
        snapshot, rows, ["iter", "worker", "time"]
    )
    for model, owner, seconds in running[: max(rows - 1, 0)]:
        style = "yellow" if slow is not None and seconds > slow else ""
        u = _ionisation_parameter(cloudy_dir, model) if cloudy_dir else None
        table.add_row(
            model,
            *(_point(snapshot["points"], model, columns) if columns else []),
            *([_format(u, True) if u else "?"] if show_u else []),
            _iteration_text(cloudy_dir, model),
            Text(owner, style="dim"),
            Text(_duration(seconds), style=style),
        )
    title = f"Running now ({len(running)}"
    title += f", yellow > 2 x median {_duration(slow / 2)})" if slow else ")"
    if not running:
        return Panel(
            Text("nothing running", style="dim"),
            title=title,
            title_align="left",
        )
    return Panel(table, title=title, title_align="left")


def _slowest_panel(snapshot, rows):
    """The slowest Cloudy runs so far, with their grid points."""
    finished = snapshot["finished"] + [
        r for r in snapshot["records"] if r["result"] in ("done", "failed")
    ]
    finished.sort(key=lambda r: -r.get("seconds", 0))
    table, columns, show_u = _models_table(
        snapshot, rows, ["iterations", "result", "worker", "time"]
    )
    for record in finished[: max(rows, 0)]:
        u = record.get("ionisation_parameter")
        iterations = record.get("iterations")
        table.add_row(
            record["model"],
            *(
                _point(snapshot["points"], record["model"], columns)
                if columns
                else []
            ),
            *([_format(u, True) if u else "?"] if show_u else []),
            "?" if iterations is None else str(iterations),
            Text(
                record["result"],
                style="green" if record["result"] == "done" else "red",
            ),
            Text(record.get("owner", ""), style="dim"),
            _duration(record.get("seconds", 0)),
        )
    return Panel(
        table,
        title=f"Slowest Cloudy runs ({len(finished)} finished)",
        title_align="left",
    )


#: The map's shades, by the fraction of a cell's models done
SHADES = " ░▒▓█"


def _map_panel(snapshot, width, height):
    """A map of the grid's first two incident axes, coloured by state.

    Each cell holds every model at that point (all the other axes): its
    shade is the fraction done, red if any failed, cyan if any running.
    """
    points = snapshot["points"]
    if points is None:
        return Panel(
            Text(
                "No grid_points.json: the run was set up before the setup "
                "wrote it",
                style="dim",
            ),
            title="Grid map",
            title_align="left",
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
    return Panel(lines, title="Grid map", title_align="left")


def _failures_panel(snapshot, rows, full=False):
    failures = _failures(snapshot)
    attempts = snapshot["attempts"]
    table = Table(box=None, padding=(0, 1), expand=True, show_header=False)
    items = list(failures.items())
    # The names and attempts are never squeezed; the reasons take the rest
    table.add_column(
        no_wrap=True,
        style="red",
        min_width=max((len(m) for m, _ in items), default=4),
    )
    table.add_column(justify="right", style="dim", no_wrap=True, min_width=3)
    table.add_column(
        no_wrap=not full,
        overflow="ellipsis" if not full else "fold",
        ratio=1,
    )
    # All that fit, or room for "and N more" below the rest
    shown = items if full or len(items) <= rows else items[: max(rows - 1, 0)]
    for model, why in shown:
        table.add_row(model, f"({attempts.get(model, '?')})", why or "")
    if len(items) > len(shown):
        table.add_row("", "", Text(f"and {len(items) - len(shown)} more (f)"))
    if not items:
        table.add_row(Text("none", style="dim"), "", "")
    return Panel(
        table,
        title=f"Failed models ({len(items)}, attempts, last reason)",
        title_align="left",
    )


def _log_panel(lines, rows, width, title="run.log"):
    """The end of run.log, its long lines wrapped, as many as fit.

    Args:
        lines (list of str)
            run.log's last lines.
        rows, width (int)
            The space inside the panel.
    """
    # The date is today's; the time is enough
    lines = [
        line[11:]
        if len(line) > 20 and line[4] == "-" and line[10] == " "
        else line
        for line in lines
    ]
    # Wrap the lines into rows and keep the last rows that fit, so the
    # newest stay in view and the panel stays full
    wrapped = []
    for line in lines[-max(rows, 0) :] if rows > 0 else []:
        style = (
            "red"
            if "ERROR" in line
            else ("yellow" if "WARNING" in line else "")
        )
        text = Text(line, style=style)
        # Tables (the rounds in run.log) are cropped rather than wrapped
        if line.partition("] ")[2].lstrip()[:1] in ("+", "|"):
            text.truncate(max(width, 1))
            wrapped.append(text)
        else:
            wrapped.extend(text.wrap(_MEASURE, max(width, 1)))
    text = Text("\n").join(wrapped[-max(rows, 0) :] if rows > 0 else [])
    text.no_wrap, text.overflow = True, "crop"
    return Panel(text, title=title, title_align="left")


def watch(path=None, interval=2.0, once=False):
    """Show a run's state, redrawn every interval seconds until q.

    Args:
        path (str)
            The run (see ``find_work_dir``).
        interval (float)
            Seconds between redraws.
        once (bool)
            Print one view and return (e.g. in scripts).
    """
    work_dir = find_work_dir(path)
    watcher = Watcher(work_dir)
    console = Console()
    extra = {"work_dir": work_dir, "interval": interval}
    if once or not sys.stdin.isatty():
        snapshot = watcher.read() | extra
        width, height = console.size
        console.print(render(snapshot, width, height))
        return

    import termios
    import tty

    view = "main"
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                snapshot = watcher.read() | extra
                width, height = console.size
                live.update(
                    render(snapshot, width, height, view), refresh=True
                )
                ready, _, _ = select.select([sys.stdin], [], [], interval)
                if ready:
                    key = sys.stdin.read(1).lower()
                    if key == "q":
                        return
                    views = {
                        "f": "failures",
                        "l": "log",
                        "m": "map",
                        "s": "slowest",
                    }
                    if key in views:
                        new = views[key]
                        view = "main" if view == new else new
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
