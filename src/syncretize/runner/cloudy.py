"""The Cloudy workers: claim models from the queue and run them.

A model ``<i>_<j>`` is the Cloudy input ``<cloudy_dir>/<i>/<j>.in``. A worker
runs a few models at once (Cloudy is single-threaded), each in its own loop
that claims models, runs them and records how they went, until the queue is
empty or the worker's time is nearly up.

Each worker appends a line to its record file (JSON lines) for every model
it runs; the finish stage gathers these into the run's models.json.
"""

import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import yaml

from syncretize.runner.log import duration, log
from syncretize.runner.queue import Queue

#: A worker claims a model only if, before it stops its runs, it has this
#: many times its longest run so far
CLAIM_TIME_FACTOR = 1.5


def work(
    queue_dir,
    cloudy_dir,
    executable,
    job,
    slots,
    claim,
    extensions,
    max_retries,
    record_file,
    deadline=None,
    stop_at=None,
    can_grow=True,
):
    """Run models from the queue until it is empty or the deadline passes.

    No model is claimed after the deadline, nor when the time left before
    ``stop_at`` is under ``CLAIM_TIME_FACTOR`` times the longest run this
    worker has had, so models that likely won't finish aren't started. A
    model still running at ``stop_at`` is stopped: put back in the queue
    without counting an attempt when the next round's time limit can grow,
    else recorded as failed. A failed model goes straight back in the queue
    while it has retries left, so it can run again in this round.

    Args:
        queue_dir (str)
            The queue's directory.
        cloudy_dir (str)
            The directory holding the Cloudy inputs.
        executable (str)
            The Cloudy directory, holding source/cloudy.exe and data/.
        job (str)
            This worker's name; loop n claims models as "<job>.<n>".
        slots (int)
            Models to run at once.
        claim (int)
            Models each loop takes from the queue at a time.
        extensions (list of str)
            The output files a model must write (see
            ``create_synthesizer_grid.required_extensions``).
        max_retries (int)
            How many times a failed model is run again.
        record_file (str)
            The JSON lines file this worker records its models in.
        deadline (float)
            The ``time.monotonic()`` after which no model is claimed, or
            None to run until the queue is empty.
        stop_at (float)
            The ``time.monotonic()`` at which running models are stopped, or
            None.
        can_grow (bool)
            Whether the next round's time limit is longer than this one's.

    Returns:
        list of dict
            The records of the models this worker ran.
    """
    queue = Queue(queue_dir)
    os.makedirs(os.path.dirname(record_file), exist_ok=True)
    lock = threading.Lock()
    records = []

    def time_to_claim():
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            return False
        longest = max((r["seconds"] for r in records), default=0)
        if stop_at is not None and stop_at - now < CLAIM_TIME_FACTOR * longest:
            log.debug(
                f"Not claiming: {duration(stop_at - now)} left, under "
                f"{CLAIM_TIME_FACTOR} x the longest run ({duration(longest)})"
            )
            return False
        return True

    def loop(slot):
        owner = f"{job}.{slot}"
        while time_to_claim():
            models = queue.claim(owner, claim)
            if not models:
                return
            for model in models:
                started = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                start = time.monotonic()
                result, reason = run_model(
                    cloudy_dir, model, executable, extensions, stop_at
                )
                attempts, requeued = None, False
                if result == "done":
                    queue.done(model, owner)
                elif result == "stopped" and can_grow:
                    queue.release(model, owner)
                    requeued = True
                else:
                    attempts = queue.fail(model, owner, max_retries)
                    requeued = attempts <= max_retries
                record = {
                    "model": model,
                    "owner": owner,
                    "result": result,
                    "reason": reason,
                    "attempt": attempts,
                    "requeued": requeued,
                    "started": started,
                    "seconds": round(time.monotonic() - start, 1),
                    **model_details(cloudy_dir, model),
                }
                with lock:
                    with open(record_file, "a") as file:
                        file.write(json.dumps(record) + "\n")
                    records.append(record)
                    # The job log only: run.log gets the round's summary
                    log.debug(_describe(record))

    with ThreadPoolExecutor(slots) as pool:
        # result() raises any loop's error here
        for future in [pool.submit(loop, slot) for slot in range(slots)]:
            future.result()
    return records


def model_details(cloudy_dir, model):
    """What a model's files say about it, for its record.

    Returns:
        dict
            "iterations" (how many Cloudy iterations it reached, or None)
            and "ionisation_parameter" (as written into its input, or None).
    """
    path = model_path(cloudy_dir, model)
    details = {"iterations": iteration(path)[0], "ionisation_parameter": None}
    try:
        with open(path + ".yaml") as file:
            value = (yaml.safe_load(file) or {}).get("ionisation_parameter")
        details["ionisation_parameter"] = (
            None if value is None else float(value)
        )
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        pass
    return details


def model_path(cloudy_dir, model):
    """A model's files, <cloudy_dir>/<i>/<j> (add .in, .out, ...)."""
    incident_index, photoionisation_index = model.split("_")
    return os.path.join(cloudy_dir, incident_index, photoionisation_index)


#: Cloudy's iteration markers in its .out, e.g. "Start Iteration Number 3"
#: and "Iteration 2 of 10"
ITERATION = re.compile(rb"Iteration (?:Number )?(\d+)(?: of (\d+))?")


def iteration(path, size=65536):
    """The iteration a Cloudy run (path without .out) has reached.

    Returns:
        (int, int)
            The iteration and the most it may take, either None when the
            .out doesn't say (e.g. it has none yet).
    """
    try:
        with open(path + ".out", "rb") as file:
            file.seek(0, os.SEEK_END)
            file.seek(max(0, file.tell() - size))
            matches = ITERATION.findall(file.read())
    except OSError:
        return None, None
    if not matches:
        return None, None
    most = next((int(m[1]) for m in reversed(matches) if m[1]), None)
    return int(matches[-1][0]), most


def run_model(cloudy_dir, model, executable, extensions, stop_at=None):
    """Run one Cloudy model.

    Returns:
        (str, str)
            "done", "failed" or "stopped" (still running at ``stop_at``), and
            why it didn't succeed (None when done).
    """
    incident_index, photoionisation_index = model.split("_")
    directory = os.path.join(cloudy_dir, incident_index)
    process = subprocess.Popen(
        [
            os.path.join(executable, "source", "cloudy.exe"),
            "-r",
            photoionisation_index,
        ],
        cwd=directory,
        env={**os.environ, "CLOUDY_DATA_PATH": f"{executable}/data/:./"},
    )
    try:
        process.wait(
            None if stop_at is None else max(0, stop_at - time.monotonic())
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return "stopped", "stopped at the time limit"
    # Imported here: it imports synthesizer, which the watcher (using this
    # module's iteration) shouldn't wait for
    from syncretize.cloudy.create_synthesizer_grid import check_if_failed

    reason = check_if_failed(
        os.path.join(directory, photoionisation_index), extensions
    )
    return ("failed", reason) if reason else ("done", None)


def _describe(record):
    """A log line for a worker's record of one model."""
    text = f"{record['owner']} {record['model']} {record['result']}"
    if record["attempt"] is not None:
        text += f" (attempt {record['attempt']})"
    if record["reason"]:
        text += f": {record['reason']}"
    if record["requeued"]:
        text += ", requeued"
    return f"{text} in {duration(record['seconds'])}"
