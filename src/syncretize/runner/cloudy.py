"""The Cloudy workers: claim models from the queue and run them.

A model ``<i>_<j>`` is the Cloudy input ``<cloudy_dir>/<i>/<j>.in``. A worker
runs a few models at once (Cloudy is single-threaded), each in its own loop
that claims models, runs them and records whether they worked, until the
queue is empty or the worker's time is nearly up.
"""

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from syncretize.runner.queue import Queue


def work(queue_dir, cloudy_dir, executable, job, slots, claim, deadline=None):
    """Run models from the queue until it is empty or the deadline passes.

    No model is claimed after the deadline; the models already claimed run
    on, so the deadline should leave time for the slowest ``claim`` models.

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
        deadline (float)
            The ``time.monotonic()`` after which no model is claimed, or
            None to run until the queue is empty.
    """
    queue = Queue(queue_dir)

    def loop(slot):
        owner = f"{job}.{slot}"
        while deadline is None or time.monotonic() < deadline:
            models = queue.claim(owner, claim)
            if not models:
                return
            for model in models:
                start = time.monotonic()
                if run_model(cloudy_dir, model, executable):
                    queue.done(model, owner)
                    result = "done"
                else:
                    result = f"failed (attempt {queue.fail(model, owner)})"
                # One write per line, so the loops' lines don't interleave
                print(
                    f"{owner} {model} {result} in "
                    f"{time.monotonic() - start:.0f} s\n",
                    end="",
                    flush=True,
                )

    with ThreadPoolExecutor(slots) as pool:
        # result() raises any loop's error here
        for future in [pool.submit(loop, slot) for slot in range(slots)]:
            future.result()


def run_model(cloudy_dir, model, executable):
    """Run one Cloudy model.

    Returns:
        bool
            Whether it worked (see ``succeeded``).
    """
    incident_index, photoionisation_index = model.split("_")
    directory = os.path.join(cloudy_dir, incident_index)
    subprocess.run(
        [
            os.path.join(executable, "source", "cloudy.exe"),
            "-r",
            photoionisation_index,
        ],
        cwd=directory,
        env={**os.environ, "CLOUDY_DATA_PATH": f"{executable}/data/:./"},
    )
    return succeeded(os.path.join(directory, photoionisation_index))


def succeeded(path):
    """Whether a Cloudy run finished and wrote its lines.

    Cloudy ends its output with "Cloudy exited OK" when it finished without
    errors, and a run that wrote its lines leaves a .emergent_elin of at
    least 100 bytes (the check ``create_synthesizer_grid`` makes).

    Args:
        path (str)
            The model's input file without ".in".
    """
    try:
        with open(path + ".out", "rb") as file:
            file.seek(0, os.SEEK_END)
            file.seek(max(0, file.tell() - 1000))
            if b"Cloudy exited OK" not in file.read():
                return False
        return os.path.getsize(path + ".emergent_elin") >= 100
    except FileNotFoundError:
        return False
