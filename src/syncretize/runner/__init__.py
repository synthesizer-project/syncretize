"""Run a grid from one config: incident grid, then Cloudy, then assembly.

    python -m syncretize.runner run run.yaml --local

See ``parameter_files/runs/run_template.yaml`` for the run config.
"""

import os

from syncretize.incident.builder import build_grid
from syncretize.runner.config import incident_module, load_run


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
        if self.config.incident["grid"] is None:
            self.incident()
        if self.config.cloudy is not None:
            raise NotImplementedError("The Cloudy stages aren't written yet")
