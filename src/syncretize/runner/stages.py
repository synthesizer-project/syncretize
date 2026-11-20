"""What the run and its Cloudy sections share: state files and Slurm jobs.

The run (``syncretize.runner.runner.Runner``) and each of its Cloudy
sections (``syncretize.runner.section.Section``) keep a state file,
``state.yaml``, in their own directory, and submit the Slurm jobs running
their stages. ``Stages`` holds what they do alike; the helpers below are
shared by both.
"""

import os
import time
from datetime import datetime, timezone

import h5py
import yaml

from syncretize.runner import slurm
from syncretize.runner.config import seconds, slurm_time

#: The kind of job running each stage, naming the jobs
#: R<round>-<run>[-C<section>]-<kind>
JOB_KINDS = {
    "start": "Incident",
    "work": "Cloudy",
    "finish": "Runner",
    "assemble": "Assemble",
}

#: The kind of the workers running perturbed models (see Section._perturb)
PERTURB_KIND = "PerturbCloudy"

#: Time limit of the finish jobs, which only tidy the queue and submit jobs
FINISH_WALLTIME = "01:00:00"


class Stages:
    """A state file and the Slurm jobs running stages, kept in a directory.

    Attributes:
        config (SimpleNamespace)
            The run config, from ``load_run``.
        directory (str)
            Where the state file (and everything else of this part of the
            run) is kept.
        index (int)
            The Cloudy section's number (None for the run as a whole), which
            the jobs pass on with --section.
    """

    def __init__(self, config, directory, index=None):
        """Keep the run config and where the state file goes.

        Args:
            config (SimpleNamespace)
                The run config, from ``load_run``.
            directory (str)
                The directory holding the state file.
            index (int)
                The Cloudy section's number, or None for the run.
        """
        self.config = config
        self.directory = directory
        self.index = index

    def job_started(self, stage):
        """Record in the state file that this Slurm job has started.

        Args:
            stage (str)
                The stage the job runs.
        """
        self._update_job(stage, started=_now())

    def job_ended(self, stage, outcome):
        """Record in the state file how this Slurm job ended.

        Args:
            stage (str)
                The stage the job ran.
            outcome (str)
                How it ended, e.g. "completed" or "failed: <error>".
        """
        self._update_job(stage, ended=_now(), outcome=outcome)

    def _update_job(self, stage, **values):
        """Add values to this Slurm job's entry in the state file.

        Args:
            stage (str)
                The stage the job runs (for a job submitted by hand, which
                has no entry yet).
            **values
                What to record, e.g. started=<time>.
        """
        job = os.environ.get("SLURM_JOB_ID")
        if job is None:
            return
        jobs = self._load_state().get("jobs", [])
        for entry in jobs:
            if entry["job"] == job:
                entry.update(values)
                break
        else:
            jobs.append({"stage": stage, "job": job, **values})
        self._save_state(jobs=jobs)

    def _submit_next(self, stage, resources, after=None):
        """Submit the job running stage, as the next job (``next_job``).

        Args:
            stage (str)
                The stage to run.
            resources (dict)
                The job's walltime, cores and (optionally) memory.
            after (str)
                A job ID to start after, however it ends.

        Returns:
            str
                The job's ID.
        """
        job = self._submit(
            stage,
            resources["walltime"],
            cores=resources["cores"],
            memory=resources.get("memory"),
            after=after,
        )
        self._save_state(next=stage, next_job=job)
        return job

    def _submit(self, stage, walltime, args=(), **job):
        """Submit a job running a stage, and record it in the state file.

        Args:
            stage (str)
                The stage to run.
            walltime (str)
                The job's time limit.
            args (sequence of str)
                More arguments for the stage.
            **job
                The rest of ``slurm.submit``'s arguments (cores, memory,
                array, after).

        Returns:
            str
                The job's ID.
        """
        name = self._job_name(stage)
        if self.index is not None:
            args = ["--section", str(self.index), *args]
        job_id = slurm.submit(
            self.config.run,
            self.config.path,
            stage,
            name,
            slurm_time(seconds(walltime)),
            args=args,
            **job,
        )
        jobs = self._load_state().get("jobs", [])
        jobs.append(
            {
                "stage": stage,
                "job": job_id,
                "name": name,
                "script": f"jobs/{name}.sh",
                "submitted": _now(),
            }
        )
        self._save_state(jobs=jobs)
        return job_id

    def _job_name(self, stage):
        """The name of the job running a stage.

        Args:
            stage (str)
                The stage.

        Returns:
            str
                e.g. "R0-bc03-Incident" or "R2-bc03-C1-Cloudy".
        """
        raise NotImplementedError

    def _details(self, stage):
        """The file a stage's scripts print to (see ``log.quiet``).

        Args:
            stage (str)
                The stage.

        Returns:
            str
                <work_dir>/logs/<job name>-<job ID>.details.
        """
        job = os.environ.get("SLURM_JOB_ID", "local")
        return os.path.join(
            self.config.run["work_dir"],
            "logs",
            f"{self._job_name(stage)}-{job}.details",
        )

    def _load_state(self):
        """The state file, <directory>/state.yaml, as a dict."""
        return _load(self._path("state.yaml"), yaml.safe_load)

    def _save_state(self, **values):
        """Add values to the state file, <directory>/state.yaml.

        Args:
            **values
                The keys to set.
        """
        state = self._load_state() | values | {"updated": _now()}
        _save(
            self._path("state.yaml"),
            lambda file: yaml.safe_dump(state, file, sort_keys=False),
        )

    def _path(self, *names):
        """A path in this part of the run's directory."""
        return os.path.join(self.directory, *names)


def _shape(grid):
    """A grid's axes and their sizes, e.g. "221 ages x 7 metallicities"."""
    with h5py.File(grid, "r") as hdf:
        return " x ".join(
            f"{len(hdf['axes'][axis])} {axis}" for axis in hdf.attrs["axes"]
        )


def _clock(when):
    """A ``time.monotonic()`` as the time of day (UTC)."""
    wall = time.time() + when - time.monotonic()
    return time.strftime("%H:%M:%S", time.gmtime(wall))


def _dump(values, indent=2):
    """Values as indented YAML (numpy arrays as lists).

    Args:
        values (dict)
            The values.
        indent (int)
            Spaces before every line.

    Returns:
        str
            The YAML's lines.
    """

    def plain(value):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    text = yaml.dump(
        plain(values), Dumper=_Dumper, sort_keys=False, width=1000
    )
    return "\n".join(" " * indent + line for line in text.splitlines())


class _Dumper(yaml.SafeDumper):
    """YAML with mappings one key a line and lists on one line."""


_Dumper.add_representer(
    list,
    lambda dumper, value: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", value, flow_style=True
    ),
)


def _count(n, thing, things=None):
    """E.g. "1 round" or "3 rounds"."""
    return f"{n} {thing if n == 1 else things or thing + 's'}"


def _now():
    """The time now (UTC), to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _elapsed(since):
    """The time since an ISO time (from ``_now``), in seconds."""
    began = datetime.fromisoformat(since)
    return (datetime.now(timezone.utc) - began).total_seconds()


def _load(path, read):
    """Read a state file (an empty dict when there's none).

    Args:
        path (str)
            The file.
        read (callable)
            Reads the open file, e.g. ``json.load``.

    Returns:
        dict
            What it holds.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as file:
        return read(file) or {}


def _save(path, write):
    """Write a state file whole or not at all (via a renamed temporary).

    Args:
        path (str)
            The file.
        write (callable)
            Writes to the open file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as file:
        write(file)
    os.replace(path + ".tmp", path)
