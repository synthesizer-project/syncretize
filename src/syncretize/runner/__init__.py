"""Run a grid from one config: incident grid, then Cloudy, then assembly.

    python -m syncretize.runner run run.yaml [--local]
    python -m syncretize.runner resume run.yaml [--local]

See ``parameter_files/runs/run_template.yaml`` for the run config.

A run keeps its state in ``<work_dir>``:

    state.yaml     what has been done and what comes next (``next``), every
                   job and round, and the models given up on
    models.json    every Cloudy model's state and attempts
    queue/         the models waiting, running, done and failed
    state/         each worker's records, gathered into models.json by the
                   finish stage

Only one stage at a time writes state.yaml and models.json (the start,
finish and assembly jobs follow each other); the workers, which run
together, each write their own file in state/.
"""

import json
import os
import time
from datetime import datetime, timezone

import yaml

from syncretize.cloudy.create_cloudy_input_grid import create_cloudy_inputs
from syncretize.cloudy.create_synthesizer_grid import (
    create_synthesizer_grid,
    required_extensions,
)
from syncretize.cloudy.utils import get_cloudy_params, resolve_param_file
from syncretize.grid_io import write_run_config
from syncretize.incident.builder import build_grid
from syncretize.runner import slurm
from syncretize.runner.cloudy import work
from syncretize.runner.config import (
    STOP_BEFORE,
    incident_module,
    load_run,
    seconds,
    slurm_time,
)
from syncretize.runner.queue import STATES, Queue

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
        if self._load_state().get("next") is not None:
            raise ValueError(
                f"{self.config.run['work_dir']} already holds a run; use "
                "resume to carry it on"
            )
        self._save_state(
            name=self.config.run["name"], run_file=self.config.path
        )
        if local:
            self._save_state(next="start")
            return self._run_here()
        return self._submit_next("start")

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
        state = self._load_state()
        stage = state.get("next")
        if stage is None:
            return self.run(local=local)
        if stage == "done":
            print(f"The run is done: {state.get('grid')}", flush=True)
            return None
        if local:
            return self._run_here()
        job = state.get("next_job")
        if job is not None and slurm.job_running(job):
            print(f"The {stage} job {job} is still queued or running")
            return None
        # A round whose workers are still going is finished after them
        after = None
        if stage == "finish" and state.get("rounds"):
            array = state["rounds"][-1].get("array_job")
            if array is not None and slurm.job_running(array):
                after = array
        return self._submit_next(stage, after=after)

    def incident(self):
        """Make the incident grid, in this process.

        Returns:
            str
                The path of the incident grid.
        """
        run, incident = self.config.run, self.config.incident
        module = incident_module(incident["model"])
        os.makedirs(run["grid_dir"], exist_ok=True)
        grid = build_grid(
            module.setup(incident["config"], run["input_dir"]),
            module.generate_point,
            run["grid_dir"],
            param_file=incident["config"],
            cores=run["incident"]["cores"],
        )
        self._write_run_config(grid)
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
        if self.config.cloudy is None:
            self._save_state(grid=incident_grid, next="done")
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
        name, n_incident, n_photoionisation = create_cloudy_inputs(
            os.path.splitext(os.path.basename(incident_grid))[0],
            os.path.dirname(incident_grid),
            self._path("cloudy"),
            cloudy["config"],
            cloudy["extra_config"],
        )
        models = [
            f"{i}_{j}"
            for i in range(n_incident)
            for j in range(n_photoionisation)
        ]
        self.queue.add(models)
        records = self._load_models()
        for model in models:
            records.setdefault(model, {"state": "todo", "attempts": []})
        self._save_models(records)
        cloudy_dir = self._path("cloudy", name)
        self._save_state(
            cloudy_dir=cloudy_dir,
            round=self._load_state().get("round", 1),
            models=self.queue.counts(),
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
        if walltime is not None:
            end = time.monotonic() + seconds(walltime)
            deadline = end - seconds(settings["margin"])
            stop_at = end - seconds(STOP_BEFORE)
        work(
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
        summary["queue"] = counts
        rounds = state.get("rounds", [])
        if rounds and rounds[-1]["round"] == round_:
            rounds[-1].update(summary)
        else:
            rounds.append(summary)
        print(
            f"Round {round_}: {summary['done']} models done, "
            f"{summary['failed']} failed, {summary['stopped']} stopped at "
            f"the time limit, {summary['lost']} lost with their job; "
            f"{counts['done']} done in all, {counts['failed']} given up, "
            f"{counts['todo']} left to run",
            flush=True,
        )
        if not counts["todo"]:
            self._save_state(rounds=rounds, models=counts, next="assemble")
            return "assemble"
        walltime = self._walltime()
        if summary["stopped"] + summary["lost"] and longer is not None:
            walltime = longer
        self._save_state(
            rounds=rounds,
            models=counts,
            round=round_ + 1,
            walltime=slurm_time(walltime),
            next="finish",
        )
        return "work"

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
        grid = create_synthesizer_grid(
            grid_dir=self.config.run["grid_dir"],
            cloudy_dir=os.path.dirname(state["cloudy_dir"]),
            incident_grid_name=os.path.basename(incident_grid),
            cloudy_param_file=cloudy["config"],
            extra_cloudy_param_file=cloudy["extra_config"],
            include_spectra=cloudy["include_spectra"],
            include_intrinsic_lines=cloudy["include_intrinsic_lines"],
            incident_grid_dir=os.path.dirname(incident_grid),
        )
        self._write_run_config(grid)
        self._save_state(grid=grid, next="done")
        return grid

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
            f"work-{round_}",
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
        return self._submit_next("finish", after=array)

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
        return self._submit_next("assemble")

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
            if stage == "start":
                self.start(submit=False)
                if self._load_state()["next"] == "finish":
                    self.work("local")
            elif stage == "finish":
                # Every local worker has stopped once work returns
                if self.finish(lambda owner: False) == "work":
                    self.work("local")
            elif stage == "assemble":
                self.assemble()
            else:
                return self._load_state()["grid"]

    def _submit_next(self, stage, after=None):
        """Submit the job running stage, as the run's next job."""
        if stage == "start":
            resources, name = self.config.run["incident"], "start"
        elif stage == "assemble":
            resources, name = self.config.run["assemble"], "assemble"
        else:
            resources = {"walltime": FINISH_WALLTIME, "cores": 1}
            name = f"finish-{self._load_state().get('round', 1)}"
        job = self._submit(
            stage,
            name,
            resources["walltime"],
            cores=resources["cores"],
            memory=resources.get("memory"),
            after=after,
        )
        self._save_state(next=stage, next_job=job)
        return job

    def _submit(self, stage, script_name, walltime, **job):
        """Submit a job running a stage of this run (see slurm.submit)."""
        job_id = slurm.submit(
            self.config.run,
            self.config.path,
            stage,
            script_name,
            slurm_time(seconds(walltime)),
            **job,
        )
        jobs = self._load_state().get("jobs", [])
        jobs.append(
            {
                "stage": stage,
                "job": job_id,
                "script": f"jobs/{script_name}.sh",
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

        failures = {
            model: entry["attempts"][-1]["reason"]
            if entry["attempts"]
            else None
            for model, entry in models.items()
            if entry["state"] == "failed"
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

    def _required_extensions(self):
        """The Cloudy output files a model must write to succeed."""
        cloudy = self.config.cloudy
        params = {}
        for key in ("config", "extra_config"):
            if cloudy[key] is not None:
                params |= get_cloudy_params(resolve_param_file(cloudy[key]))[0]
        return required_extensions(
            cloudy["include_spectra"],
            cloudy["include_intrinsic_lines"],
            params.get("geometry"),
        )

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
