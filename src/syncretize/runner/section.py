"""One Cloudy section of a run: its Cloudy inputs, rounds and final grid.

A run config may have several Cloudy sections (``Cloudy`` or ``Cloudy1``,
``Cloudy2``, ...), each running Cloudy on the run's incident grid with its
own set-up. Once the start job has made the incident grid, every section
goes its own way at once: it writes its Cloudy inputs, runs rounds of
workers, each ended by a finish job that submits the next round, perturbs
the models that failed every attempt (if asked to), and assembles its final
grid. A section keeps everything in ``<work_dir>/<name>``:

    state.yaml     what it has done and runs next (``next``), its jobs and
                   rounds, and the models that failed
    models.json    every Cloudy model's state and attempts
    queue/         the models waiting, running, done and failed
    state/         each worker's records, gathered into models.json by the
                   finish stage
    cloudy/        the Cloudy inputs and outputs

Only one stage of a section at a time writes its state.yaml and models.json
(its start, finish and assembly jobs follow each other); its workers, which
run together, each write their own file in state/.
"""

import json
import os
import socket
import statistics
import time

import h5py
import numpy as np

from syncretize.cloudy.create_cloudy_input_grid import create_cloudy_inputs
from syncretize.cloudy.create_synthesizer_grid import (
    create_synthesizer_grid,
    required_extensions,
)
from syncretize.cloudy.utils import get_cloudy_params, resolve_param_file
from syncretize.grid_io import write_cloudy_build, write_run_config
from syncretize.runner import perturb, slurm
from syncretize.runner.cloudy import work
from syncretize.runner.config import (
    STOP_BEFORE,
    seconds,
    section_name,
    slurm_time,
)
from syncretize.runner.log import duration, log, quiet, table, took
from syncretize.runner.queue import STATES, Queue
from syncretize.runner.stages import (
    FINISH_WALLTIME,
    JOB_KINDS,
    PERTURB_KIND,
    Stages,
    _clock,
    _count,
    _dump,
    _elapsed,
    _load,
    _now,
    _save,
    _shape,
)

#: The most given-up models run.log lists after a round or the assembly
LIST_FAILURES = 20


class Section(Stages):
    """Drive one Cloudy section of a run.

    Attributes:
        runner (Runner)
            The run this section belongs to.
        cloudy (dict)
            Its Cloudy section of the run config.
        name (str)
            Its name (see ``section_name``), naming its directory.
    """

    def __init__(self, runner, index):
        """Find a section of the run.

        Args:
            runner (Runner)
                The run.
            index (int)
                The section's number (1 for Cloudy or Cloudy1).
        """
        self.runner = runner
        self.cloudy = runner.config.sections[index]
        self.name = section_name(self.cloudy)
        super().__init__(
            runner.config,
            os.path.join(runner.config.run["work_dir"], self.name),
            index,
        )

    @property
    def label(self):
        """The section's short label, e.g. "C2" (in job names and logs)."""
        return f"C{self.index}"

    @property
    def queue(self):
        """The section's model queue, in <directory>/queue."""
        return Queue(self._path("queue"))

    def setup(self, incident_grid):
        """Write the Cloudy inputs and put every Cloudy model in the queue.

        The inputs go in ``<directory>/cloudy/<grid name>/<i>/<j>.in`` and
        the queue holds one ``<i>_<j>`` per model. Repeating this keeps the
        models' states, so a section can be set up again after a crash.

        Args:
            incident_grid (str)
                The path of the incident grid.

        Returns:
            str
                The directory holding the Cloudy inputs.
        """
        cloudy = self.cloudy
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
            name=self.name,
            index=self.index,
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
        settings = self.config.run["cloudy"]
        deadline = stop_at = None
        start = time.monotonic()
        log.info(
            f"Worker {job} of {self.name} on {socket.gethostname()}: "
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
            self.cloudy["executable"],
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
        if state.get("perturbing"):
            summary["perturb"] = True
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

        See ``syncretize.runner.perturb``. A side that failed is tried again
        further out, until ``Cloudy.perturb.max_steps``.

        Returns:
            list of str
                The runs queued.
        """
        if self.cloudy["perturb"] is None:
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
            self.cloudy["perturb"],
            write_perturbed_input,
        )
        if runs:
            self.queue.add(runs)
            models = self._load_models()
            for run in runs:
                models.setdefault(run, {"state": "todo", "attempts": []})
            self._save_models(models)
            # Its rounds (until the assembly) are the PerturbCloudy phase
            self._save_state(perturbing=True)
            # A run's level ends its name (e.g. 1_3p0-2): level 1 is a model
            # just perturbed, a higher one a failed side tried further out
            new = [run for run in runs if run.endswith(("-1", "+1"))]
            points = {run.partition("p")[0] for run in new}
            if new:
                log.info(
                    f"Perturbing {_count(len(points), 'failed model')}: "
                    f"queued {_count(len(new), 'run')} either side of their "
                    "Cloudy parameters, to interpolate them from"
                )
            if len(runs) > len(new):
                log.info(
                    f"Retrying {_count(len(runs) - len(new), 'perturbed run')}"
                    " that failed, twice as far out"
                )
        return runs

    def assemble(self):
        """Make the section's final grid from its Cloudy outputs, here.

        Models that failed every try are marked in its failures dataset.
        Once every section of the run is done, so is the run.

        Returns:
            str
                The path of the final grid.
        """
        state, cloudy = self._load_state(), self.cloudy
        incident_grid = self.runner._load_state()["incident_grid"]
        os.makedirs(self.config.run["grid_dir"], exist_ok=True)
        start = time.monotonic()
        log.info("Assembling the final grid from the Cloudy outputs")
        with quiet(self._details("assemble")):
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
        write_run_config(
            grid,
            self.config.path,
            {
                "Run": self.config.run,
                "Incident": self.config.incident,
                "Cloudy": cloudy,
            },
        )
        write_cloudy_build(grid, build=cloudy["executable"])

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
        self.runner.section_done()
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
        state = self._load_state()
        round_ = state.get("round", 1)
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
        finish = self._submit_next("finish", self._resources("finish"), array)
        phase = " (perturbing)" if state.get("perturbing") else ""
        log.info(
            f"Round {round_}{phase}: submitted {_count(workers, 'worker')} ("
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
        job = self._submit_next("assemble", self._resources("assemble"))
        log.info(f"Every model has run: submitted the assembly job {job}")
        return job

    def resume(self):
        """Submit the job carrying the section on from where it stopped.

        Nothing is done while the job running its next stage is still
        queued or running.

        Returns:
            str
                The submitted job's ID, or None when there is nothing to do.
        """
        state = self._load_state()
        stage = state.get("next")
        if stage == "done":
            return None
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
        new = self._submit_next(stage, self._resources(stage), after)
        log.info(
            f"Resuming from the {stage} stage"
            + (f" (job {job} had stopped)" if job is not None else "")
            + f": submitted the {stage} job {new}"
            + (f" to run after the workers ({after})" if after else "")
        )
        return new

    def run_here(self):
        """Run the section's remaining stages in this process.

        Returns:
            str
                The path of its final grid.
        """
        while True:
            stage = self._load_state()["next"]
            if stage == "finish":
                # Every local worker has stopped once work returns
                if self.finish(lambda owner: False) == "work":
                    perturbing = self._load_state().get("perturbing")
                    self.runner.set_stage(
                        "perturb" if perturbing else "work", self
                    )
                    self.work("local")
                    self.runner.set_stage("finish", self)
            elif stage == "assemble":
                self.runner.set_stage("assemble", self)
                self.assemble()
            else:
                return self._load_state()["grid"]

    def describe(self):
        """The section's Cloudy parameters, for run.log.

        Returns:
            str
                The lines describing it.
        """
        fixed, varied = self._cloudy_params()
        n_models = int(np.prod([len(v) for v in varied.values()]))
        files = [self.cloudy["config"], self.cloudy["extra_config"]]
        return "\n".join(
            [
                f"Cloudy section {self.label}, {self.name} ("
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
        )

    def _resources(self, stage):
        """The Slurm resources of a finish or assembly job.

        Args:
            stage (str)
                "finish" or "assemble".

        Returns:
            dict
                The job's walltime, cores and memory.
        """
        if stage == "assemble":
            return self.config.run["assemble"]
        return {"walltime": FINISH_WALLTIME, "cores": 1}

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

    def _cloudy_params(self):
        """The Cloudy parameters: (fixed, varied), extra_config's winning."""
        fixed, varied = {}, {}
        for key in ("config", "extra_config"):
            if self.cloudy[key] is not None:
                fixed_, varied_ = get_cloudy_params(
                    resolve_param_file(self.cloudy[key])
                )
                fixed |= fixed_
                varied |= varied_
        return fixed, varied

    def _required_extensions(self):
        """The Cloudy output files a model must write to succeed."""
        return required_extensions(
            self.cloudy["include_spectra"],
            self.cloudy["include_intrinsic_lines"],
            self._cloudy_params()[0].get("geometry"),
        )

    def _log_round(self, rounds, records):
        """Log a round's table, run times and given-up models to run.log.

        Args:
            rounds (list of dict)
                Every round so far, the one just finished last.
            records (list of dict)
                The round's worker records.
        """
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
                f"{r['round']} perturb" if r.get("perturb") else r["round"],
                r.get("walltime", "-"),
                r.get("workers", "here"),
                r["done"],
                r["failed"],
                r["retried"],
                r["stopped"],
                r["lost"],
                r["given_up"],
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
        """List (some of) the models that failed every attempt, with why.

        Args:
            failures (dict)
                Each model's reason for its last failure.
            when (str)
                Ends the line, e.g. "this round".
        """
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
        models = _count(len(failures), "model")
        log.info(
            f"{models} failed every attempt {when}".rstrip()
            + " (with why the last failed):\n"
            + "\n".join(lines)
        )

    def _log_done(self):
        """Log the section's totals once its final grid is made."""
        state = self._load_state()
        counts = state.get("models", {})
        rounds = len([r for r in state.get("rounds", []) if "queue" in r])
        started = self.runner._load_state().get("started")
        since = f" over {duration(_elapsed(started))}" if started else ""
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
        """A stage's job name, e.g. "R2-bc03-C1-Cloudy" (setup is round 0).

        Args:
            stage (str)
                The stage.

        Returns:
            str
                The job's name.
        """
        if stage == "start":
            # The section's setup runs in the run's start job
            return self.runner._job_name("start")
        state = self._load_state()
        kind = JOB_KINDS[stage]
        if stage == "work" and state.get("perturbing"):
            kind = PERTURB_KIND
        return (
            f"R{state.get('round', 1)}-{self.config.run['name']}-"
            f"{self.label}-{kind}"
        )

    def _load_models(self):
        """Every Cloudy model's state and attempts, from models.json."""
        return _load(self._path("models.json"), json.load)

    def _save_models(self, models):
        """Write models.json.

        Args:
            models (dict)
                Every model's state and attempts.
        """
        _save(
            self._path("models.json"),
            lambda file: json.dump(models, file, indent=1),
        )
