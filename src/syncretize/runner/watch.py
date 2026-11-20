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

Keys: q quits, f toggles a full-screen list of the failed models (those that
failed every attempt), l a full-screen run.log.
"""

import getpass
import json
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
    from syncretize.runner.config import load_run

    return load_run(path).run["work_dir"]


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
            "running": self._running(),
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
                counts[state] = len(os.listdir(self._path("queue", state)))
            except FileNotFoundError:
                counts[state] = 0
        self._counts, self._counted = counts, time.monotonic()
        return counts

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
            "main", "failures" or "log".

    Returns:
        rich.layout.Layout
    """
    state, counts = snapshot["state"], snapshot["counts"]
    layout = Layout()
    header = _header(snapshot)
    footer = Text(
        f" q quit · f failed models · l run.log · every "
        f"{snapshot['interval']:g}s"
        f" · {snapshot['work_dir']}",
        style="dim",
        no_wrap=True,
        overflow="ellipsis",
    )

    if view in ("log", "failures"):
        rows = height - HEADER - 1 - 2
        body = (
            _log_panel(snapshot["log"], rows, width - 4)
            if view == "log"
            else _failures_panel(snapshot, rows, full=True)
        )
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


def _running_panel(snapshot, rows):
    """The models running longest, flagging those slow for this round."""
    running = snapshot["running"]
    times = [
        r["seconds"]
        for r in snapshot["records"]
        if r["result"] in ("done", "failed")
    ]
    slow = 2 * statistics.median(times) if times else None
    table = Table(box=None, padding=(0, 1), expand=True, show_header=False)
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True, style="dim", overflow="ellipsis")
    table.add_column(justify="right", no_wrap=True)
    for model, owner, seconds in running[: max(rows, 0)]:
        style = "yellow" if slow is not None and seconds > slow else ""
        table.add_row(model, owner, Text(_duration(seconds), style=style))
    if not running:
        table.add_row(Text("nothing running", style="dim"), "", "")
    title = f"Running longest ({len(running)} running"
    title += f", yellow > 2 x median {_duration(slow / 2)})" if slow else ")"
    return Panel(table, title=title, title_align="left")


def _failures_panel(snapshot, rows, full=False):
    failures = _failures(snapshot)
    attempts = snapshot["attempts"]
    table = Table(box=None, padding=(0, 1), expand=True, show_header=False)
    table.add_column(no_wrap=True, style="red")
    table.add_column(justify="right", style="dim", no_wrap=True)
    table.add_column(
        no_wrap=not full, overflow="ellipsis" if not full else "fold"
    )
    items = list(failures.items())
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
                    if key in ("f", "l"):
                        new = {"f": "failures", "l": "log"}[key]
                        view = "main" if view == new else new
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
