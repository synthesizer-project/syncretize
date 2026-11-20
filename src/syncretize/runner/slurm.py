"""Submit the Runner's jobs to Slurm and ask Slurm which are running.

Every job runs one stage, ``python -m syncretize.runner <stage> run.yaml``,
from a script kept in ``<work_dir>/jobs`` with its log in ``<work_dir>/logs``.
"""

import os
import shlex
import subprocess
import sys

from syncretize.runner.log import log

#: Job states (sacct) in which a job may still be running
RUNNING_STATES = {
    "PENDING",
    "CONFIGURING",
    "RUNNING",
    "COMPLETING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
}


def submit(
    run,
    run_file,
    stage,
    script_name,
    walltime,
    cores=1,
    memory=None,
    array=None,
    after=None,
    args=(),
):
    """Write a job script running one stage and submit it.

    Args:
        run (dict)
            The run config's Run section.
        run_file (str)
            The run config.
        stage (str)
            The stage the job runs.
        script_name (str)
            The script's name in <work_dir>/jobs, without ".sh".
        walltime (str)
            The job's time limit.
        cores (int)
            Cores for the job (each array task).
        memory (str)
            The job's memory, or None for Slurm's default.
        array (int)
            Submit an array of this many tasks.
        after (str)
            A job ID to start after, however it ends.
        args (sequence of str)
            More arguments for the stage.

    Returns:
        str
            The job ID.
    """
    work_dir = run["work_dir"]
    for directory in ("jobs", "logs"):
        os.makedirs(os.path.join(work_dir, directory), exist_ok=True)
    log_name = "%x-%A_%a.out" if array else "%x-%j.out"
    options = {
        "job-name": f"{run['name']}-{stage}",
        "output": os.path.join(work_dir, "logs", log_name),
        "chdir": work_dir,
        "time": walltime,
        "cpus-per-task": cores,
        "mem": memory,
        "partition": run["slurm"]["partition"],
        "account": run["slurm"]["account"],
        "array": f"0-{array - 1}" if array else None,
        "dependency": f"afterany:{after}" if after else None,
    }
    command = [sys.executable, "-m", "syncretize.runner", stage, run_file]
    lines = ["#!/bin/bash -l"]
    lines += [
        f"#SBATCH --{key}={value}"
        for key, value in options.items()
        if value is not None
    ]
    lines += [run["environment"], shlex.join([*command, *args])]
    script = os.path.join(work_dir, "jobs", f"{script_name}.sh")
    with open(script, "w") as file:
        file.write("\n".join(lines) + "\n")

    result = subprocess.run(
        ["sbatch", "--parsable", script],
        capture_output=True,
        text=True,
        check=True,
    )
    # --parsable prints "<id>" or "<id>;<cluster>"
    job = result.stdout.strip().split(";")[0]
    log.debug(f"Submitted {script} as job {job}")
    return job


def is_running():
    """A function saying whether a queue claim's owner is still running.

    Owners are "<array job>_<task>.<slot>" (see ``Runner.submit_round``);
    sacct is asked once per array job. Any other owner (e.g. "local.0") is
    taken to have stopped.

    Returns:
        callable
            Takes an owner and returns a bool.
    """
    running = {}

    def check(owner):
        job = owner.rpartition(".")[0]
        array = job.partition("_")[0]
        if not array.isdigit():
            return False
        if array not in running:
            running[array] = _running_tasks(array)
        return job in running[array]

    return check


def job_running(job):
    """Whether a job (or any task of an array job) is queued or running."""
    return bool(_running_tasks(job))


def _running_tasks(array):
    """The tasks of an array job that may still be running."""
    result = subprocess.run(
        ["sacct", "-n", "-X", "-P", "-o", "JobID,State", "-j", array],
        capture_output=True,
        text=True,
        check=True,
    )
    running = set()
    for line in result.stdout.splitlines():
        if "|" not in line:
            continue
        job, state = line.split("|")[:2]
        # e.g. "CANCELLED by 123"
        if state.split(" ")[0] in RUNNING_STATES:
            running.add(job)
    return running
