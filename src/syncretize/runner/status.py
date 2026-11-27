"""Where a run, and each of its Cloudy sections, is.

One status for the watcher, scripts topping runs up, and anyone asking:

    waiting   not submitted yet
    running   its next job is queued or running (or it runs here)
    stalled   its next job has gone without carrying on: it failed (e.g.
              sbatch refused its next job), or Slurm ended it before it ran
              (e.g. NODE_FAIL); resume the run to carry on
    walltime  its models ran out of time at max_walltime
    review    it paused for a look at its failures
    done      its grids are made

A stalled job can't record that it stalled (one that never ran writes
nothing), so the status is read from the state files against your jobs in
Slurm (see ``live_jobs``).
"""

import getpass
import os
import shutil
import subprocess
from datetime import datetime, timezone

import yaml

#: A job submitted this recently (s) counts as running even when the jobs
#: listed in Slurm don't hold it (they may have been listed just before)
GRACE = 300

#: The statuses, the one needing a look most first (a run takes the first
#: of its sections')
STATUSES = ("stalled", "review", "walltime", "running", "waiting", "done")


def live_jobs():
    """Your jobs queued or running in Slurm.

    Returns:
        set of str
            Their IDs (an array's, not its tasks'), or None when squeue
            can't be asked (no Slurm here, or it failed).
    """
    if shutil.which("squeue") is None:
        return None
    try:
        result = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%F"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return set(result.stdout.split())


def section_status(state, live):
    """A section's status (or a run's before its sections are set up).

    Args:
        state (dict)
            Its state file.
        live (set of str)
            Your jobs in Slurm (see ``live_jobs``); None if unknown, when a
            job is taken to be running.

    Returns:
        str
            One of STATUSES.
    """
    stage = state.get("next")
    if stage is None:
        return "waiting"
    if stage in ("done", "walltime", "review"):
        return stage
    # A run here (no job), or Slurm not asked, is taken to be running
    job = state.get("next_job")
    if live is None or job is None or not str(job).isdigit():
        return "running"
    if str(job) in live or _age(state, job) < GRACE:
        return "running"
    return "stalled"


def _age(state, job):
    """How long ago a job was submitted (s; infinite if not known).

    Args:
        state (dict)
            The state file listing it.
        job (str)
            The job's ID.

    Returns:
        float
            Seconds since it was submitted.
    """
    for entry in state.get("jobs") or []:
        if str(entry.get("job")) == str(job) and entry.get("submitted"):
            submitted = datetime.fromisoformat(entry["submitted"])
            return (datetime.now(timezone.utc) - submitted).total_seconds()
    return float("inf")


def stall_reason(state):
    """Why a stalled section (or run) stalled, from its next job's entry.

    Args:
        state (dict)
            Its state file.

    Returns:
        str
            e.g. "finish job 123 failed: RuntimeError(...)" or "finish job
            123 ended without running".
    """
    stage, job = state.get("next"), state.get("next_job")
    for entry in state.get("jobs") or []:
        if str(entry.get("job")) == str(job):
            outcome = entry.get("outcome", "")
            if outcome.startswith("failed"):
                return f"{stage} job {job} {outcome}"
            if entry.get("started"):
                return f"{stage} job {job} stopped without carrying on"
    return f"{stage} job {job} ended without running"


def read_yaml(path):
    """A YAML file (empty if it's missing or being rewritten).

    Args:
        path (str)
            The file.

    Returns:
        dict
            What it holds.
    """
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    try:
        with open(path) as file:
            return yaml.load(file, Loader=loader) or {}
    except (OSError, yaml.YAMLError):
        return {}


def run_status(work_dir, live, read=read_yaml):
    """A run's status, and its sections'.

    Args:
        work_dir (str)
            The run's work directory.
        live (set of str)
            Your jobs in Slurm (see ``live_jobs``), or None.
        read (callable)
            Reads a state file (path to dict), e.g. one keeping what it read.

    Returns:
        (str, dict, dict)
            The run's status, its state, and each section's state and
            status by its directory's name: {name: (state, status)}.
    """
    run = read(os.path.join(work_dir, "state.yaml"))
    sections = {}
    for name in (run.get("sections") or {}).values():
        state = read(os.path.join(work_dir, name, "state.yaml"))
        sections[name] = (state, section_status(state, live))
    if run.get("next") in (None, "start", "done") or not sections:
        return section_status(run, live), run, sections
    statuses = {status for _, status in sections.values()}
    # Every section done but not the run: its last assembly died before it
    # could mark it (resuming marks it)
    if statuses == {"done"}:
        return "stalled", run, sections
    status = next(s for s in STATUSES if s in statuses)
    return status, run, sections
