"""Run a grid from one config: incident grid, then Cloudy, then assembly.

    python -m syncretize.runner run run.yaml --local

See ``parameter_files/runs/run_template.yaml`` for the run config.
"""

import os

import yaml

from syncretize.cloudy.create_cloudy_input_grid import create_cloudy_inputs
from syncretize.incident.builder import build_grid
from syncretize.runner.config import incident_module, load_run
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
        return build_grid(
            module.setup(incident["config"], run["input_dir"]),
            module.generate_point,
            run["grid_dir"],
            param_file=incident["config"],
            cores=run["incident"]["cores"],
        )

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

    def _save_state(self, **values):
        """Add values to the run's state file, <work_dir>/state.yaml."""
        path = os.path.join(self.config.run["work_dir"], "state.yaml")
        state = {}
        if os.path.exists(path):
            with open(path) as file:
                state = yaml.safe_load(file) or {}
        with open(path, "w") as file:
            yaml.safe_dump(state | values, file)

    def run(self, local=False):
        """Run every stage.

        Args:
            local (bool)
                Run in this process rather than submitting Slurm jobs.
        """
        if not local:
            raise NotImplementedError(
                "Slurm submission isn't written yet; use --local"
            )
        incident_grid = self.config.incident["grid"] or self.incident()
        if self.config.cloudy is None:
            return
        self.cloudy_setup(incident_grid)
        raise NotImplementedError("The Cloudy workers aren't written yet")
