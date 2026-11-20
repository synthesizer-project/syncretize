"""Run a grid from one config: incident grid, then Cloudy, then assembly.

    python -m syncretize.runner run run.yaml [--local]
    python -m syncretize.runner resume run.yaml [--local]

See ``parameter_files/runs/run_template.yaml`` for the run config.

A run keeps its state in ``<work_dir>``:

    state.yaml     what has been done and what comes next (``next``), every
                   job and round, and the models that failed
    models.json    every Cloudy model's state and attempts
    queue/         the models waiting, running, done and failed
    state/         each worker's records, gathered into models.json by the
                   finish stage

Only one stage at a time writes state.yaml and models.json (the start,
finish and assembly jobs follow each other); the workers, which run
together, each write their own file in state/.
"""

import getpass
import json
import os
import platform
import socket
import statistics
import time
from datetime import datetime, timezone

import h5py
import numpy as np
import yaml
from synthesizer._version import __version__ as synthesizer_version

from syncretize._version import __version__ as syncretize_version
from syncretize.cloudy.create_cloudy_input_grid import create_cloudy_inputs
from syncretize.cloudy.create_synthesizer_grid import (
    create_synthesizer_grid,
    required_extensions,
)
from syncretize.cloudy.utils import get_cloudy_params, resolve_param_file
from syncretize.grid_io import write_run_config
from syncretize.incident.builder import build_grid
from syncretize.runner import perturb, slurm
from syncretize.runner.cloudy import work
from syncretize.runner.config import (
    STOP_BEFORE,
    incident_module,
    load_run,
    seconds,
    slurm_time,
)
from syncretize.runner.log import (
    BANNER,
    duration,
    log,
    logging_to,
    quiet,
    set_stage,
    setup,
    table,
    took,
)
from syncretize.runner.queue import STATES, Queue

#: The kind of job running each stage, naming the jobs R<round>-<run>-<kind>
JOB_KINDS = {
    "start": "Incident",
    "work": "Cloudy",
    "finish": "Runner",
    "assemble": "Assemble",
}

#: The most given-up models run.log lists after a round or the assembly
LIST_FAILURES = 20

#: Time limit of the finish jobs, which only tidy the queue and submit jobs
FINISH_WALLTIME = "01:00:00"


class Runner:
    """Drive the stages of one run.

    Attributes:
        config (SimpleNamespace)
            The run config, from ``load_run``.
    """

    def __init__(self, run_file):
        """Load and check the run config."""
        self.config = load_run(run_file)

    @property
    def queue(self):
        """The run's model queue, in <work_dir>/queue."""
        return Queue(self._path("queue"))

    def run(self, local=False):
        """Run every stage, here or as a chain of Slurm jobs.

        With Slurm, a start job makes the incident grid and the Cloudy
        inputs and submits the first round of workers. Each round ends with
        a finish job, which submits the next round or the assembly job.

        Args:
            local (bool)
                Run in this process rather than submitting Slurm jobs.

        Returns:
            str
                The path of the final grid (with local) or the start job's
                ID.
        """
        self._setup_log("run")
        if self._load_state().get("next") is not None:
            raise ValueError(
                f"{self.config.run['work_dir']} already holds a run; use "
                "resume to carry it on"
            )
        self._save_state(
            name=self.config.run["name"],
            run_file=self.config.path,
            started=_now(),
        )
        log.info(BANNER)
        log.info(self.describe())
        if local:
            self._save_state(next="start")
            return self._run_here()
        job = self._submit_next("start")
        log.info(f"Submitted the start job {job}")
        return job

    def resume(self, local=False):
        """Carry on a run from wherever it stopped.

        Runs (or submits) the stage ``next`` in the state file names.
        Nothing is done while the job running that stage is still queued or
        running.

        Args:
            local (bool)
                Run the remaining stages in this process.

        Returns:
            str
                The final grid (with local) or the submitted job's ID, or
                None when there is nothing to do.
        """
        self._setup_log("resume")
        state = self._load_state()
        stage = state.get("next")
        if stage is None:
            return self.run(local=local)
        if stage == "done":
            log.info(f"The run is already done: {state.get('grid')}")
            return None
        if local:
            log.info(f"Resuming the run here, from its {stage} stage")
            return self._run_here()
        job = state.get("next_job")
        if job is not None and slurm.job_running(job):
            log.info(
                f"Nothing to resume: the {stage} job {job} is still "
                "queued or running"
            )
            return None
        # A round whose workers are still going is finished after them
        after = None
        if stage == "finish" and state.get("rounds"):
            array = state["rounds"][-1].get("array_job")
            if array is not None and slurm.job_running(array):
                after = array
        new = self._submit_next(stage, after=after)
        log.info(
            f"Resuming the run from its {stage} stage"
            + (f" (job {job} had stopped)" if job is not None else "")
            + f": submitted the {stage} job {new}"
            + (f" to run after the workers ({after})" if after else "")
        )
        return new

    def incident(self):
        """Make the incident grid, in this process.

        Returns:
            str
                The path of the incident grid.
        """
        run, incident = self.config.run, self.config.incident
        module = incident_module(incident["model"])
        os.makedirs(run["grid_dir"], exist_ok=True)
        start = time.monotonic()
        log.info(f"Making the incident grid with {incident['model']}")
        with quiet(self._details("start")):
            grid = build_grid(
                module.setup(incident["config"], run["input_dir"]),
                module.generate_point,
                run["grid_dir"],
                param_file=incident["config"],
                cores=run["incident"]["cores"],
                verbose=False,
            )
        self._write_run_config(grid)
        log.info(
            f"Made the incident grid {grid} ({_shape(grid)}) in {took(start)}"
        )
        return grid

    def start(self, submit=True):
        """Make the incident grid and the Cloudy inputs, then submit a round.

        This is the start job. The incident grid isn't made when
        ``Incident.grid`` is given (or an earlier start made it), and
        nothing more is done without Cloudy.

        Args:
            submit (bool)
                Submit the first round of workers.

        Returns:
            str
                The path of the incident grid.
        """
        incident_grid = self._load_state().get("incident_grid")
        if incident_grid is None or not os.path.exists(incident_grid):
            incident_grid = self.config.incident["grid"] or self.incident()
            self._save_state(incident_grid=incident_grid)
        if self.config.incident["grid"] is not None:
            log.info(
                f"Using the incident grid {incident_grid} "
                f"({_shape(incident_grid)})"
            )
        if self.config.cloudy is None:
            self._save_state(grid=incident_grid, next="done")
            log.info(f"Done: the incident grid is {incident_grid}")
            return incident_grid
        self.cloudy_setup(incident_grid)
        self._save_state(next="finish")
        if submit:
            self.submit_round()
        return incident_grid

    def cloudy_setup(self, incident_grid):
        """Write the Cloudy inputs and put every Cloudy model in the queue.

        The inputs go in ``<work_dir>/cloudy/<grid name>/<i>/<j>.in`` and the
        queue holds one ``<i>_<j>`` per model. Repeating this keeps the
        models' states, so a run can be set up again after a crash.

        Args:
            incident_grid (str)
                The path of the incident grid.

        Returns:
            str
                The directory holding the Cloudy inputs.
        """
        cloudy = self.config.cloudy
        start = time.monotonic()
        with quiet(self._details("start")):
            name, n_incident, n_photoionisation = create_cloudy_inputs(
                os.path.splitext(os.path.basename(incident_grid))[0],
                os.path.dirname(incident_grid),
                self._path("cloudy"),
                cloudy["config"],
                cloudy["extra_config"],
            )
        cloudy_dir = self._path("cloudy", name)
        # Incident points below the ionising luminosity threshold get no
        # Cloudy runs
        with open(os.path.join(cloudy_dir, "grid_points.json")) as file:
            below = set(json.load(file).get("below_threshold", []))
        models = [
            f"{i}_{j}"
            for i in range(n_incident)
            if i not in below
            for j in range(n_photoionisation)
        ]
        self.queue.add(models)
        records = self._load_models()
        for model in models:
            records.setdefault(model, {"state": "todo", "attempts": []})
        self._save_models(records)
        counts = self.queue.counts()
        self._save_state(
            below_threshold=int(len(below) * n_photoionisation),
            cloudy_dir=cloudy_dir,
            round=self._load_state().get("round", 1),
            models=counts,
        )
        log.info(
            f"Wrote {len(models)} Cloudy inputs ({n_incident} incident "
            f"points x {n_photoionisation} photoionisation models) to "
            f"{cloudy_dir} in {took(start)}"
        )
        if below:
            log.info(
                f"{_count(len(below), 'incident point')} "
                f"({len(below) * n_photoionisation} models) are below the "
                "ionising luminosity threshold, so aren't run"
            )
        if counts["todo"] < len(models):
            log.info(
                f"Kept the earlier state of {len(models) - counts['todo']}"
                " models ("
                + ", ".join(
                    f"{n} {state}"
                    for state, n in counts.items()
                    if n and state != "todo"
                )
                + ")"
            )
        return cloudy_dir

    def work(self, job, walltime=None):
        """Run Cloudy models from the queue, as one worker.

        Args:
            job (str)
                The worker's name, e.g. its Slurm job ID.
            walltime (str)
                The worker's time limit. It claims no model within
                ``Run.cloudy.margin`` of it, and stops the models still
                running shortly before it. None runs until the queue is
                empty.
        """
        cloudy, settings = self.config.cloudy, self.config.run["cloudy"]
        deadline = stop_at = None
        start = time.monotonic()
        log.info(
            f"Worker {job} on {socket.gethostname()}: "
            f"{_count(settings['cores_per_worker'], 'Cloudy run')} at once, "
            f"{_count(settings['claim'], 'model')} per claim, up to "
            f"{_count(settings['max_retries'], 'retry', 'retries')}"
        )
        if walltime is not None:
            end = start + seconds(walltime)
            deadline = end - seconds(settings["margin"])
            stop_at = end - seconds(STOP_BEFORE)
            log.info(
                f"Time limit {walltime}: claiming models until "
                f"{_clock(deadline)} and stopping Cloudy runs at "
                f"{_clock(stop_at)} (UTC)"
            )
        records = work(
            self.queue.root,
            self._load_state()["cloudy_dir"],
            cloudy["executable"],
            job,
            slots=settings["cores_per_worker"],
            claim=settings["claim"],
            extensions=self._required_extensions(),
            max_retries=settings["max_retries"],
            record_file=self._path("state", f"{job}.jsonl"),
            deadline=deadline,
            stop_at=stop_at,
            can_grow=self._longer_walltime() is not None,
        )
        results = [r["result"] for r in records]
        busy = sum(r["seconds"] for r in records)
        log.info(
            f"Worker {job} ran {len(records)} models in {took(start)} ("
            + ", ".join(
                f"{results.count(x)} {x}"
                for x in ("done", "failed", "stopped")
                if results.count(x)
            )
            + f"); Cloudy ran for {duration(busy)} in all"
        )

    def finish(self, is_running):
        """Tidy the queue after a round of workers and say what comes next.

        Gathers the workers' records into models.json and the round's
        totals into the state file. Models claimed by workers that died
        without stopping them (lost) are put back without counting an
        attempt, like those the workers stopped at the time limit. When any
        were stopped or lost, the next round's time limit grows by
        ``walltime_growth``, up to ``max_walltime``; when it can't grow,
        lost models count as failed.

        Args:
            is_running (callable)
                Takes a claim's owner ("<job>.<slot>") and returns whether
                its worker is still running.

        Returns:
            str
                "work" when models are left for another round, whose time
                limit is ``walltime`` in the state file, or "assemble".
        """
        state = self._load_state()
        longer = self._longer_walltime()
        lost = self.queue.requeue_claims(is_running, fail=longer is None)
        claimed = self.queue.counts()["claimed"]
        if claimed:
            raise RuntimeError(
                f"{claimed} models are still claimed by running workers; "
                "finish runs once every worker has stopped"
            )
        # Lost models that failed (and any a larger max_retries now allows)
        self.queue.requeue_failed(self.config.run["cloudy"]["max_retries"])
        records = self._gather_records(lost)

        round_ = state.get("round", 1)
        counts = self.queue.counts()
        summary = {
            "round": round_,
            "walltime": slurm_time(self._walltime()),
            "finished": _now(),
        }
        for result in ("done", "failed", "stopped", "lost"):
            summary[result] = sum(r["result"] == result for r in records)
        summary["retried"] = sum(
            r["result"] == "failed" and r["requeued"] for r in records
        )
        rounds = state.get("rounds", [])
        previous = [r for r in rounds if r["round"] < round_ and "queue" in r]
        before = previous[-1]["queue"]["failed"] if previous else 0
        summary["given_up"] = counts["failed"] - before
        summary["queue"] = counts
        if rounds and rounds[-1]["round"] == round_:
            rounds[-1].update(summary)
        else:
            rounds.append(summary)
        self._log_round(rounds, records)
        # With every model run, those that failed every attempt are run again
        # either side of their parameters, to be interpolated
        if not counts["todo"] and self._perturb():
            counts = self.queue.counts()
        if not counts["todo"]:
            self._save_state(rounds=rounds, models=counts, next="assemble")
            return "assemble"
        walltime = self._walltime()
        if summary["stopped"] + summary["lost"] and longer is not None:
            walltime = longer
            log.info(
                f"{_count(summary['stopped'] + summary['lost'], 'model')} "
                "ran out of time, so the next round's time limit grows to "
                f"{slurm_time(walltime)}"
            )
        self._save_state(
            rounds=rounds,
            models=counts,
            round=round_ + 1,
            walltime=slurm_time(walltime),
            next="finish",
        )
        return "work"

    def _perturb(self):
        """Queue runs either side of the models that failed every attempt.

        See syncretize.runner.perturb. A side that failed is tried again
        further out, until Cloudy.perturb.max_steps.

        Returns:
            list of str
                The runs queued.
        """
        if self.config.cloudy["perturb"] is None:
            return []

        from syncretize.cloudy.create_cloudy_input_grid import (
            write_perturbed_input,
        )

        states = {}
        for state in STATES:
            for name in os.listdir(os.path.join(self.queue.root, state)):
                states[name.partition("@")[0]] = state
        failed = [
            model for model, state in states.items() if state == "failed"
        ]
        runs = perturb.advance(
            self._load_state()["cloudy_dir"],
            failed,
            states,
            self.config.cloudy["perturb"],
            write_perturbed_input,
        )
        if runs:
            self.queue.add(runs)
            models = self._load_models()
            for run in runs:
                models.setdefault(run, {"state": "todo", "attempts": []})
            self._save_models(models)
            points = {run.partition("p")[0] for run in runs}
            log.info(
                f"Perturbing {_count(len(points), 'failed model')}: queued "
                f"{_count(len(runs), 'run')} either side of their Cloudy "
                "parameters, to interpolate them from"
            )
        return runs

    def assemble(self):
        """Make the final grid from the Cloudy outputs, in this process.

        Models that failed every try are marked in its failures dataset.

        Returns:
            str
                The path of the final grid.
        """
        state, cloudy = self._load_state(), self.config.cloudy
        incident_grid = state["incident_grid"]
        os.makedirs(self.config.run["grid_dir"], exist_ok=True)
        start = time.monotonic()
        log.info("Assembling the final grid from the Cloudy outputs")
        with quiet(self._details("assemble")):
            grid = self._assemble(state, cloudy, incident_grid)
        self._write_run_config(grid)

        # The failed models interpolated from perturbed runs haven't failed
        from syncretize.cloudy.create_synthesizer_grid import _interpolations

        interpolated = sorted(
            _interpolations(
                os.path.dirname(state["cloudy_dir"]),
                os.path.basename(state["cloudy_dir"]),
                self._required_extensions(),
            )
        )
        failures = {
            model: why
            for model, why in (state.get("failures") or {}).items()
            if model not in interpolated
        }
        self._save_state(
            grid=grid,
            next="done",
            failures=failures,
            interpolated=interpolated,
        )
        log.info(
            f"Wrote {grid} ({_shape(grid)}, "
            f"{os.path.getsize(grid) / 1e6:.0f} MB) in {took(start)}"
        )
        if interpolated:
            log.info(
                f"Interpolated {_count(len(interpolated), 'failed model')} "
                "from runs perturbed either side of them: "
                + ", ".join(interpolated[:LIST_FAILURES])
                + (" …" if len(interpolated) > LIST_FAILURES else "")
            )
        self._log_failures(failures)
        self._log_done()
        return grid

    def _assemble(self, state, cloudy, incident_grid):
        """Run create_synthesizer_grid for this run."""
        return create_synthesizer_grid(
            grid_dir=self.config.run["grid_dir"],
            cloudy_dir=os.path.dirname(state["cloudy_dir"]),
            incident_grid_name=os.path.basename(incident_grid),
            cloudy_param_file=cloudy["config"],
            extra_cloudy_param_file=cloudy["extra_config"],
            include_spectra=cloudy["include_spectra"],
            include_intrinsic_lines=cloudy["include_intrinsic_lines"],
            incident_grid_dir=os.path.dirname(incident_grid),
        )

    def submit_round(self):
        """Submit a round of workers and the finish job that follows them.

        The round has ``Run.cloudy.workers`` workers (fewer when fewer models
        are left), each with the round's time limit.

        Returns:
            str
                The finish job's ID.
        """
        settings = self.config.run["cloudy"]
        round_ = self._load_state().get("round", 1)
        walltime = slurm_time(self._walltime())
        todo = self.queue.counts()["todo"]
        # An empty queue still gets one, so its finish job runs
        workers = max(
            1,
            min(settings["workers"], -(-todo // settings["cores_per_worker"])),
        )
        array = self._submit(
            "work",
            walltime,
            cores=settings["cores_per_worker"],
            array=workers,
            args=["--walltime", walltime],
        )
        rounds = self._load_state().get("rounds", [])
        rounds.append(
            {
                "round": round_,
                "walltime": walltime,
                "workers": workers,
                "array_job": array,
            }
        )
        self._save_state(rounds=rounds)
        finish = self._submit_next("finish", after=array)
        log.info(
            f"Round {round_}: submitted {_count(workers, 'worker')} ("
            f"{settings['cores_per_worker']} cores each, time limit "
            f"{walltime}) as job {array}, then finish job {finish}; "
            f"{_count(todo, 'model')} to run"
        )
        return finish

    def continue_run(self):
        """Finish a round and submit what comes next.

        This is the finish job: another round when models are left, else
        the assembly job.

        Returns:
            str
                The ID of the next finish job or of the assembly job.
        """
        if self.finish(slurm.is_running()) == "work":
            return self.submit_round()
        job = self._submit_next("assemble")
        log.info(f"Every model has run: submitted the assembly job {job}")
        return job

    def job_started(self, stage):
        """Record in the state file that this Slurm job has started."""
        self._update_job(stage, started=_now())

    def job_ended(self, stage, outcome):
        """Record in the state file how this Slurm job ended."""
        self._update_job(stage, ended=_now(), outcome=outcome)

    def _update_job(self, stage, **values):
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

    def _run_here(self):
        """Run the remaining stages in this process."""
        while True:
            stage = self._load_state()["next"]
            set_stage(stage if stage != "done" else "run")
            if stage == "start":
                self.start(submit=False)
                if self._load_state()["next"] == "finish":
                    set_stage("work")
                    self.work("local")
            elif stage == "finish":
                # Every local worker has stopped once work returns
                if self.finish(lambda owner: False) == "work":
                    set_stage("work")
                    self.work("local")
            elif stage == "assemble":
                self.assemble()
            else:
                return self._load_state()["grid"]

    def _submit_next(self, stage, after=None):
        """Submit the job running stage, as the run's next job."""
        if stage == "start":
            resources = self.config.run["incident"]
        elif stage == "assemble":
            resources = self.config.run["assemble"]
        else:
            resources = {"walltime": FINISH_WALLTIME, "cores": 1}
        job = self._submit(
            stage,
            resources["walltime"],
            cores=resources["cores"],
            memory=resources.get("memory"),
            after=after,
        )
        self._save_state(next=stage, next_job=job)
        return job

    def _submit(self, stage, walltime, **job):
        """Submit a job running a stage of this run (see slurm.submit)."""
        name = self._job_name(stage)
        job_id = slurm.submit(
            self.config.run,
            self.config.path,
            stage,
            name,
            slurm_time(seconds(walltime)),
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

    def _gather_records(self, lost):
        """Add the workers' records (and lost claims) to models.json.

        The workers' record files are removed once gathered, and the state
        file's ``failures`` lists the models given up on with why they last
        failed.

        Args:
            lost (dict)
                Each dead worker's models, from ``Queue.requeue_claims``.

        Returns:
            list of dict
                The records gathered.
        """
        records_dir = self._path("state")
        files = []
        if os.path.isdir(records_dir):
            files = sorted(
                os.path.join(records_dir, name)
                for name in os.listdir(records_dir)
                if name.endswith(".jsonl")
            )
        records = []
        for path in files:
            with open(path) as file:
                for line in file:
                    # A worker killed mid-write leaves a partial last line
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        for owner, models in lost.items():
            records += [
                {
                    "model": model,
                    "owner": owner,
                    "result": "lost",
                    "reason": "its job ended without stopping it",
                    "found": _now(),
                }
                for model in models
            ]

        models = self._load_models()
        for record in records:
            entry = models.setdefault(
                record["model"], {"state": "todo", "attempts": []}
            )
            # A finish stage that died after saving models.json gathers the
            # same records again
            if record not in entry["attempts"]:
                entry["attempts"].append(record)
        for state in STATES:
            for name in os.listdir(os.path.join(self.queue.root, state)):
                model = name.partition("@")[0]
                models.setdefault(model, {"attempts": []})["state"] = state
        self._save_models(models)
        for path in files:
            os.remove(path)

        # The grid points that failed (perturbed runs that failed are in
        # models.json and perturb.json)
        failures = {
            model: entry["attempts"][-1]["reason"]
            if entry["attempts"]
            else None
            for model, entry in models.items()
            if entry["state"] == "failed" and not perturb.is_perturbed(model)
        }
        self._save_state(failures=failures)
        return records

    def _walltime(self):
        """This round's worker time limit, in seconds."""
        return seconds(
            self._load_state().get(
                "walltime", self.config.run["cloudy"]["walltime"]
            )
        )

    def _longer_walltime(self):
        """The next round's longer time limit (s), or None if it can't grow."""
        settings = self.config.run["cloudy"]
        walltime = self._walltime()
        longer = min(
            round(walltime * settings["walltime_growth"]),
            seconds(settings["max_walltime"]),
        )
        return longer if longer > walltime else None

    def _setup_log(self, stage):
        """Log to run.log too, unless the command line already set it up."""
        if not logging_to(self.config.run["work_dir"]):
            setup(self.config.run["work_dir"], stage)

    def _cloudy_params(self):
        """The Cloudy parameters: (fixed, varied), extra_config's winning."""
        fixed, varied = {}, {}
        for key in ("config", "extra_config"):
            if self.config.cloudy[key] is not None:
                fixed_, varied_ = get_cloudy_params(
                    resolve_param_file(self.config.cloudy[key])
                )
                fixed |= fixed_
                varied |= varied_
        return fixed, varied

    def _required_extensions(self):
        """The Cloudy output files a model must write to succeed."""
        cloudy = self.config.cloudy
        return required_extensions(
            cloudy["include_spectra"],
            cloudy["include_intrinsic_lines"],
            self._cloudy_params()[0].get("geometry"),
        )

    def describe(self):
        """What this run is: versions, where it runs, and its whole config.

        Returns:
            str
                The lines run.log starts with.
        """
        config = self.config
        lines = [
            f"syncretize {syncretize_version}, synthesizer "
            f"{synthesizer_version}, Python {platform.python_version()}",
            f"On {socket.gethostname()} as {getpass.getuser()} at {_now()}",
            "",
            f"Run config {config.path} (with every default):",
            _dump(
                {
                    "Run": config.run,
                    "Incident": config.incident,
                    "Cloudy": config.cloudy,
                }
            ),
        ]
        if config.incident["config"] is not None:
            with open(config.incident["config"]) as file:
                params = yaml.safe_load(file)
            lines += [
                f"Incident model parameters ({config.incident['config']}):",
                _dump(params),
            ]
        if config.cloudy is not None:
            fixed, varied = self._cloudy_params()
            n_models = int(np.prod([len(v) for v in varied.values()]))
            files = [config.cloudy["config"], config.cloudy["extra_config"]]
            lines += [
                "Cloudy parameters ("
                + " + ".join(str(f) for f in files if f is not None)
                + "):",
                "  fixed:",
                _dump(fixed, 4),
                f"  varied ({n_models} photoionisation models per incident "
                "point):",
                _dump(varied, 4) if varied else "    (none)",
                "  A model succeeds when Cloudy exited OK and wrote "
                + ", ".join(f".{e}" for e in self._required_extensions())
                + " (at least 100 bytes each)",
            ]
        return "\n".join(lines)

    def _log_round(self, rounds, records):
        """Log a round's table, run times and given-up models to run.log."""
        headers = [
            "round",
            "time limit",
            "workers",
            "done",
            "errors",
            "retried",
            "stopped",
            "lost",
            "failed",
            "left",
        ]
        rows = [
            [
                r["round"],
                r.get("walltime", "-"),
                r.get("workers", "here"),
                # Rounds summarised by an earlier version may lack some
                r.get("done", 0),
                r.get("failed", 0),
                r.get("retried", 0),
                r.get("stopped", 0),
                r.get("lost", 0),
                r.get("given_up", 0),
                r["queue"]["todo"],
            ]
            for r in rounds
            if "queue" in r
        ]
        log.info(
            f"Round {rounds[-1]['round']} has finished. The rounds so far:\n"
            + table(headers, rows)
        )
        runs = [r for r in records if r["result"] in ("done", "failed")]
        if runs:
            times = sorted(r["seconds"] for r in runs)
            longest = max(runs, key=lambda r: r["seconds"])
            log.info(
                f"Cloudy run times this round: median "
                f"{duration(statistics.median(times))}, 95% under "
                f"{duration(times[int(0.95 * (len(times) - 1))])}, longest "
                f"{duration(longest['seconds'])} ({longest['model']})"
            )
        failures = self._load_state().get("failures", {})
        models = {r["model"] for r in records}
        self._log_failures(
            {m: why for m, why in failures.items() if m in models},
            "this round",
        )
        counts = rounds[-1]["queue"]
        log.info(
            f"{counts['done']} of {sum(counts.values())} models done, "
            f"{counts['failed']} failed, {counts['todo']} left to run; "
            f"{self._cpu_hours():.1f} CPU-hours of Cloudy so far"
        )

    def _log_failures(self, failures, when=""):
        """List (some of) the models that failed every attempt, with why."""
        if not failures:
            return
        lines = [
            f"  {model}: {why}"
            for model, why in list(failures.items())[:LIST_FAILURES]
        ]
        if len(failures) > LIST_FAILURES:
            lines.append(
                f"  and {len(failures) - LIST_FAILURES} more (see state.yaml)"
            )
        log.info(
            f"{len(failures)} models failed every attempt {when}".rstrip()
            + " (with why the last failed):\n"
            + "\n".join(lines)
        )

    def _log_done(self):
        """Log the run's totals once the final grid is made."""
        state = self._load_state()
        counts = state.get("models", {})
        rounds = len([r for r in state.get("rounds", []) if "queue" in r])
        since = ""
        if "started" in state:
            began = datetime.fromisoformat(state["started"])
            elapsed = (datetime.now(timezone.utc) - began).total_seconds()
            since = f" over {duration(elapsed)}"
        # What the grid's points hold: Cloudy's results, interpolations
        # from perturbed runs, failures, or no Cloudy (below the threshold)
        with h5py.File(state["grid"], "r") as hdf:
            total = int(hdf["failures"].size)
            filled = {
                key: int(hdf[key][()].sum())
                for key in ("failures", "interpolated", "below_threshold")
                if key in hdf
            }
        ran = total - sum(filled.values())
        runs = sum(counts.values())
        log.info(
            f"Done: {ran} of {total} grid points from Cloudy, "
            f"{filled.get('interpolated', 0)} interpolated from perturbed "
            f"runs, {filled.get('failures', 0)} failed, "
            f"{filled.get('below_threshold', 0)} below the ionising "
            "threshold; "
            f"{_count(runs, 'Cloudy run')} in "
            f"{_count(rounds, 'round')}{since}, "
            f"{self._cpu_hours():.1f} CPU-hours. The grid is {state['grid']}"
        )

    def _cpu_hours(self):
        """The Cloudy run time recorded in models.json, in hours."""
        return (
            sum(
                attempt.get("seconds", 0)
                for entry in self._load_models().values()
                for attempt in entry["attempts"]
            )
            / 3600
        )

    def _job_name(self, stage):
        """A stage's job name, e.g. "R2-bc03-Cloudy" (start is round 0)."""
        round_ = 0 if stage == "start" else self._load_state().get("round", 1)
        return f"R{round_}-{self.config.run['name']}-{JOB_KINDS[stage]}"

    def _details(self, stage):
        """The file a stage's scripts print to (see log.quiet)."""
        job = os.environ.get("SLURM_JOB_ID", "local")
        return self._path("logs", f"{self._job_name(stage)}-{job}.details")

    def _write_run_config(self, grid):
        """Store the run config in the grid's RunConfig group."""
        config = self.config
        write_run_config(
            grid,
            config.path,
            {
                "Run": config.run,
                "Incident": config.incident,
                "Cloudy": config.cloudy,
            },
        )

    def _load_state(self):
        """The run's state file, <work_dir>/state.yaml, as a dict."""
        return _load(self._path("state.yaml"), yaml.safe_load)

    def _save_state(self, **values):
        """Add values to the run's state file, <work_dir>/state.yaml."""
        state = self._load_state() | values | {"updated": _now()}
        _save(
            self._path("state.yaml"),
            lambda file: yaml.safe_dump(state, file, sort_keys=False),
        )

    def _load_models(self):
        """Every Cloudy model's state and attempts, from models.json."""
        return _load(self._path("models.json"), json.load)

    def _save_models(self, models):
        """Write models.json."""
        _save(
            self._path("models.json"),
            lambda file: json.dump(models, file, indent=1),
        )

    def _path(self, *names):
        """A path in the run's work directory."""
        return os.path.join(self.config.run["work_dir"], *names)


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
    """Values as indented YAML (numpy arrays as lists)."""

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


def _load(path, read):
    """Read a state file (an empty dict when there's none)."""
    if not os.path.exists(path):
        return {}
    with open(path) as file:
        return read(file) or {}


def _save(path, write):
    """Write a state file whole or not at all (via a renamed temporary)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as file:
        write(file)
    os.replace(path + ".tmp", path)
