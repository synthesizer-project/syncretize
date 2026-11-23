"""The Runner's logs.

Every job logs to its own Slurm log (stdout), each line prefixed with the
time since the job started:

    [00012.34]: Wrote 1547 Cloudy inputs in 18 s

The start, finish and assembly jobs, and ``run`` and ``resume``, also add
the run's story to ``<work_dir>/run.log``, each line prefixed with the date
and the job:

    2026-11-20 13:06:20 [C1 start 12379695] Wrote 1547 Cloudy inputs in 18 s

where a line about one Cloudy section starts with its label (C1, C2, ...).
The workers, which run many at once, only log to their own job logs (the
finish jobs sum them up in run.log); the other jobs of different sections
may write to run.log at once, a line at a time.

``log.info`` goes to both logs, ``log.debug`` only to the job log.
"""

import contextlib
import logging
import os
import sys
import time

#: The Runner's logger
log = logging.getLogger("syncretize.runner")

#: Printed as a run starts: its grid of points, named
BANNER = """
⡏⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⢹
⡇⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⢸
⡇⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⢸
⡇⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⢸
⡇⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⢸
⡇⠀+-+-+-+-+-+-+-+-+-+-+⠀⢸
⡇⠀|S|Y|N|C|R|E|T|I|Z|E|⠀⢸
⡇⠀+-+-+-+-+-+-+-+-+-+-+⠀⢸
⡇⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⢸
⡇⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⢸
⡇⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⢸
⡇⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⠶⠀⢸
⣇⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣸
"""

#: What run.log lines say made them (e.g. "finish 12379722"), and which
#: work directory's run.log is written to
_context = {"tag": "", "work_dir": None}


def setup(work_dir, stage, job=None, run_log=True, section=None):
    """Log this job to stdout and (unless run_log is False) to run.log.

    Args:
        work_dir (str)
            The run's work directory, holding run.log.
        stage (str)
            The stage this job runs.
        job (str)
            The job's ID, if it has one.
        run_log (bool)
            Also add to run.log (not for the workers).
        section (str)
            The label of the Cloudy section the job runs (e.g. "C2"), if
            any.
    """
    for handler in log.handlers:
        handler.close()
    log.handlers.clear()
    _context["work_dir"] = work_dir
    log.setLevel(logging.DEBUG)
    log.propagate = False
    set_stage(stage, job, section)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.DEBUG)
    stdout.setFormatter(_Formatter(_elapsed))
    log.addHandler(stdout)
    if run_log:
        os.makedirs(work_dir, exist_ok=True)
        run = logging.FileHandler(
            os.path.join(work_dir, "run.log"), encoding="utf-8"
        )
        run.setLevel(logging.INFO)
        run.setFormatter(_Formatter(_dated))
        log.addHandler(run)


def logging_to(work_dir):
    """Whether the logs are set up for this work directory's run.

    Args:
        work_dir (str)
            The run's work directory.

    Returns:
        bool
            True if the logs have handlers, set up for ``work_dir``.
    """
    return bool(log.handlers) and _context["work_dir"] == work_dir


def set_stage(stage, job=None, section=None):
    """Say which stage, job and Cloudy section run.log lines come from.

    Args:
        stage (str)
            The stage.
        job (str)
            The job's ID, if it has one.
        section (str)
            The Cloudy section's label (e.g. "C2"), if the lines are about
            one.
    """
    _context["tag"] = " ".join(x for x in (section, stage, job) if x)


def took(start):
    """The time since start (a ``time.monotonic()``), in sensible units.

    Args:
        start (float)
            The ``time.monotonic()`` to count from.

    Returns:
        str
            The time since, e.g. "2.5 min" (see ``duration``).
    """
    return duration(time.monotonic() - start)


def duration(seconds):
    """A number of seconds in sensible units.

    Args:
        seconds (float)
            The time.

    Returns:
        str
            e.g. "12.0 s", "2.5 min" or "1.25 h".
    """
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


#: The columns of a table of rounds (run.log's after each round, and watch's)
ROUND_COLUMNS = (
    "round",
    "time limit",
    "workers",
    "done",
    "errored",
    "timed out",
    "retried",
    "failed",
    "left",
)


def round_row(summary):
    """A round's row of a table of rounds (see ROUND_COLUMNS).

    Its runs that errored are those that failed, hit the time limit
    (stopped) or were lost with their job; those not left failed were
    retried, and those left failed ("given_up") have failed every attempt.

    Args:
        summary (dict)
            The round's summary, as a finish job saves it in the state file.

    Returns:
        list
            Its cells.
    """
    errored = sum(summary.get(key, 0) for key in ("failed", "stopped", "lost"))
    given_up = summary.get("given_up", 0)
    round_ = summary["round"]
    return [
        f"{round_} perturb" if summary.get("perturb") else round_,
        summary.get("walltime", "-"),
        summary.get("workers", "here"),
        summary.get("done", 0),
        errored,
        summary.get("stopped", 0),
        max(errored - given_up, 0),
        given_up,
        summary["queue"]["todo"],
    ]


def table(headers, rows):
    """A bordered table, as Synthesizer's Pipeline prints.

    Args:
        headers (list of str)
            The column titles.
        rows (list of list)
            The rows' values.

    Returns:
        str
            The table's lines.
    """
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *rows)
    ]
    rule = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def line(values):
        """One row of the table, its values right-aligned in their columns.

        Args:
            values (list)
                The row's values.

        Returns:
            str
                The row, between "|" borders.
        """
        cells = (f" {str(v):>{w}} " for v, w in zip(values, widths))
        return "|" + "|".join(cells) + "|"

    return "\n".join(
        [rule, line(headers), rule, *(line(row) for row in rows), rule]
    )


@contextlib.contextmanager
def quiet(path):
    """Send what the scripts print to a file instead of the job log.

    Their warnings (lines holding "WARNING") are logged as well.

    Args:
        path (str)
            The file to add their output (stdout and stderr) to.

    Yields:
        None
            The body runs with its output going to ``path``.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as file:
        start = file.tell()
        with contextlib.redirect_stdout(file), contextlib.redirect_stderr(
            file
        ):
            yield
    with open(path) as file:
        file.seek(start)
        for text in file:
            if "WARNING" in text:
                log.warning(text.strip().removeprefix("WARNING:").strip())


def _elapsed(record):
    """A job log line's prefix: the time since logging started.

    Args:
        record (logging.LogRecord)
            The line's record.

    Returns:
        str
            e.g. "[00012.34]:" (seconds).
    """
    return f"[{record.relativeCreated / 1000:08.2f}]:"


def _dated(record):
    """A run.log line's prefix: when (UTC) and where it comes from.

    Args:
        record (logging.LogRecord)
            The line's record.

    Returns:
        str
            e.g. "2025-01-02 03:04:05 [C2 finish 123]" (see ``set_stage``).
    """
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(record.created))
    return f"{when} [{_context['tag']}]"


class _Formatter(logging.Formatter):
    """Prefix every line of a message (tables span several)."""

    def __init__(self, prefix):
        """Make a formatter that starts every line with ``prefix(record)``.

        Args:
            prefix (callable)
                Takes a ``logging.LogRecord`` and returns its prefix.
        """
        super().__init__()
        self.prefix = prefix

    def format(self, record):
        """Format a record, prefixing each of its lines.

        Warnings and errors also get their level after the prefix, and an
        exception's traceback follows the message.

        Args:
            record (logging.LogRecord)
                The record.

        Returns:
            str
                Its lines, each with the prefix.
        """
        prefix = self.prefix(record)
        if record.levelno >= logging.WARNING:
            prefix += f" {record.levelname}:"
        text = record.getMessage()
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return "\n".join(f"{prefix} {line}" for line in text.splitlines())
