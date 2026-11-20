"""Run a grid from one config: incident grid, then Cloudy, then assembly.

    python -m syncretize.runner run run.yaml --local

See ``parameter_files/runs/run_template.yaml`` for the run config.
"""

import os
import time

import yaml

from syncretize.cloudy.create_cloudy_input_grid import create_cloudy_inputs
from syncretize.cloudy.create_synthesizer_grid import create_synthesizer_grid
from syncretize.grid_io import write_run_config
from syncretize.incident.builder import build_grid
from syncretize.runner.cloudy import work
from syncretize.runner.config import (
    incident_module,
    load_run,
    seconds,
    slurm_time,
)
from syncretize.runner.queue import Queue


class Runner:
    """Drive the stages of one run.

    Attributes:
        config (SimpleNamespace)
            The run config, from ``load_run``.
    """

    def __init__(self, run_file):
        """Load and check the run config."""
        self.config = load_run(run_file)

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
        cloudy, work_dir = self.config.cloudy, self.config.run["work_dir"]
        name, n_incident, n_photoionisation = create_cloudy_inputs(
            os.path.splitext(os.path.basename(incident_grid))[0],
            os.path.dirname(incident_grid),
            os.path.join(work_dir, "cloudy"),
            cloudy["config"],
            cloudy["extra_config"],
        )
        Queue(os.path.join(work_dir, "queue")).add(
            f"{i}_{j}"
            for i in range(n_incident)
            for j in range(n_photoionisation)
        )
        cloudy_dir = os.path.join(work_dir, "cloudy", name)
        self._save_state(incident_grid=incident_grid, cloudy_dir=cloudy_dir)
        return cloudy_dir

    def work(self, job, walltime=None):
        """Run Cloudy models from the queue, as one worker.

        Args:
            job (str)
                The worker's name, e.g. its Slurm job ID.
            walltime (str)
                The worker's time limit; it starts no model within
                ``Run.cloudy.margin`` of it. None runs until the queue is
                empty.
        """
        cloudy, settings = self.config.cloudy, self.config.run["cloudy"]
        deadline = None
        if walltime is not None:
            deadline = (
                time.monotonic()
                + seconds(walltime)
                - seconds(settings["margin"])
            )
        work(
            os.path.join(self.config.run["work_dir"], "queue"),
            self._load_state()["cloudy_dir"],
            cloudy["executable"],
            job,
            slots=settings["cores_per_worker"],
            claim=settings["claim"],
            deadline=deadline,
        )

    def finish(self, is_running):
        """Tidy the queue after a round of workers and say what comes next.

        Failed models are put back to run again, up to
        ``Run.cloudy.max_retries`` times. Models whose worker stopped (at its
        time limit, say) are put back without counting an attempt, and the
        next round's time limit grows by ``walltime_growth``, up to
        ``max_walltime``; when it can't grow, they count as failed.

        Args:
            is_running (callable)
                Takes a claim's owner ("<job>.<slot>") and returns whether
                its worker is still running.

        Returns:
            str
                "work" when models are left for another round, whose time
                limit is ``walltime`` in the state file, or "assemble".
        """
        settings = self.config.run["cloudy"]
        state = self._load_state()
        queue = Queue(os.path.join(self.config.run["work_dir"], "queue"))
        walltime = seconds(state.get("walltime", settings["walltime"]))
        longer = min(
            round(walltime * settings["walltime_growth"]),
            seconds(settings["max_walltime"]),
        )

        stopped = queue.requeue_claims(is_running, fail=longer <= walltime)
        stopped = sorted(model for models in stopped.values() for model in models)
        requeued, final = queue.requeue_failed(settings["max_retries"])
        counts = queue.counts()
        if counts["claimed"]:
            raise RuntimeError(
                f"{counts['claimed']} models are still claimed by running "
                "workers; finish runs once every worker has stopped"
            )
        if stopped and longer > walltime:
            walltime = longer
        round_ = state.get("round", 1)
        print(
            f"Round {round_}: {counts['done']} models done, "
            f"{len(stopped)} stopped by the time limit, {len(requeued)} "
            f"failures to retry, {len(final)} given up, {counts['todo']} "
            "left to run",
            flush=True,
        )
        if not counts["todo"]:
            return "assemble"
        self._save_state(round=round_ + 1, walltime=slurm_time(walltime))
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
        self._save_state(grid=grid)
        return grid

    def _write_run_config(self, grid):
        """Store the run config in the grid's RunConfig group."""
        config = self.config
        write_run_config(
            grid,
            config.path,
            {"Run": config.run, "Incident": config.incident, "Cloudy": config.cloudy},
        )

    def _load_state(self):
        """The run's state file, <work_dir>/state.yaml, as a dict."""
        path = os.path.join(self.config.run["work_dir"], "state.yaml")
        if not os.path.exists(path):
            return {}
        with open(path) as file:
            return yaml.safe_load(file) or {}

    def _save_state(self, **values):
        """Add values to the run's state file, <work_dir>/state.yaml."""
        state = self._load_state() | values
        path = os.path.join(self.config.run["work_dir"], "state.yaml")
        with open(path, "w") as file:
            yaml.safe_dump(state, file)

    def run(self, local=False):
        """Run every stage.

        Args:
            local (bool)
                Run in this process rather than submitting Slurm jobs.

        Returns:
            str
                The path of the final grid (with --local).
        """
        if not local:
            raise NotImplementedError(
                "Slurm submission isn't written yet; use --local"
            )
        incident_grid = self.config.incident["grid"] or self.incident()
        if self.config.cloudy is None:
            return incident_grid
        self.cloudy_setup(incident_grid)
        self.work("local")
        # Locally every worker has stopped once work returns
        while self.finish(lambda owner: False) == "work":
            self.work("local")
        return self.assemble()
