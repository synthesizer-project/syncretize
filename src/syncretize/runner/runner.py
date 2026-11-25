"""Run grids from one config: an incident grid, then its Cloudy grids.

    python -m syncretize.runner run run.yaml [--local]
    python -m syncretize.runner resume run.yaml [--local]

See ``parameter_files/runs/run_template.yaml`` for the run config.

A run makes one incident grid and runs Cloudy on it once for each of its
Cloudy sections (``syncretize.runner.section``), each making its own final
grid. With Slurm, the start job makes the incident grid and every section's
Cloudy inputs, then submits each section's first round; the sections then go
their own ways at once. The run keeps its state in ``<work_dir>``:

    state.yaml     the incident grid, what the run does next (``next``:
                   start, cloudy while its sections run, or done), the start
                   job, and the final grids
    run.log        the run's story, from every job but the workers (lines
                   name the section they are about, e.g. [C2 finish 123])
    jobs/, logs/   every job's script and log
    <section>/     each Cloudy section's own state (see section.py)
"""

import getpass
import os
import platform
import socket
import time

import yaml

from syncretize.grid_io import write_run_config
from syncretize.incident.builder import build_grid
from syncretize.parser import YAMLLoader
from syncretize.provenance import version
from syncretize.runner import log as logs
from syncretize.runner import slurm
from syncretize.runner.config import incident_module, load_run
from syncretize.runner.log import BANNER, log, logging_to, quiet, setup, took
from syncretize.runner.section import Section
from syncretize.runner.stages import (
    JOB_KINDS,
    Stages,
    _dump,
    _now,
    _shape,
)


class Runner(Stages):
    """Drive the stages of one run.

    Attributes:
        config (SimpleNamespace)
            The run config, from ``load_run``.
        sections (list of Section)
            The run's Cloudy sections, in order (none for a run making only
            the incident grid).
    """

    def __init__(self, run_file):
        """Load and check the run config.

        Args:
            run_file (str)
                The run config (run.yaml).
        """
        config = load_run(run_file)
        super().__init__(config, config.run["work_dir"])
        self.sections = [Section(self, index) for index in config.sections]

    def section(self, index):
        """One of the run's Cloudy sections, by its number.

        Args:
            index (int)
                The section's number (1 for Cloudy or Cloudy1).

        Returns:
            Section
                The section.
        """
        for section in self.sections:
            if section.index == index:
                return section
        raise ValueError(
            f"{self.config.path} has no Cloudy section {index} (it has "
            f"{[section.index for section in self.sections]})"
        )

    def run(self, local=False):
        """Run every stage, here or as a chain of Slurm jobs.

        With Slurm, a start job makes the incident grid and every section's
        Cloudy inputs and submits each section's first round of workers.
        Each round ends with a finish job, which submits the section's next
        round or its assembly job.

        Args:
            local (bool)
                Run in this process rather than submitting Slurm jobs.

        Returns:
            list of str or str
                The final grids (with local: every section's, or the
                incident grid without Cloudy), or the start job's ID.
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
            sections={s.label: s.name for s in self.sections},
        )
        log.info(BANNER)
        log.info(self.describe())
        if local:
            self._save_state(next="start")
            return self._run_here()
        job = self._submit_next("start", self.config.run["incident"])
        log.info(f"Submitted the start job {job}")
        return job

    def resume(self, local=False):
        """Carry on a run from wherever it stopped.

        Runs (or submits) the start stage if the incident grid or the Cloudy
        set-up isn't done, else each unfinished section's next stage.
        Nothing is done for a job still queued or running.

        Args:
            local (bool)
                Run the remaining stages in this process.

        Returns:
            list of str or str
                The final grids (with local), the submitted jobs' IDs (None
                for a job already queued or running), or None when the run
                is done.
        """
        self._setup_log("resume")
        state = self._load_state()
        stage = state.get("next")
        if stage is None:
            return self.run(local=local)
        if stage == "done":
            log.info(f"The run is already done: {state.get('grids')}")
            return None
        if local:
            log.info(f"Resuming the run here, from its {stage} stage")
            return self._run_here()
        if stage == "start":
            job = state.get("next_job")
            if job is not None and slurm.job_running(job):
                log.info(
                    f"Nothing to resume: the start job {job} is still "
                    "queued or running"
                )
                return None
            new = self._submit_next("start", self.config.run["incident"])
            log.info(
                "Resuming the run from its start stage"
                + (f" (job {job} had stopped)" if job is not None else "")
                + f": submitted the start job {new}"
            )
            return new
        jobs = []
        for section in self.sections:
            self.set_stage("resume", section)
            jobs.append(section.resume())
        self.set_stage("resume")
        # Every section done but not the run: the last assembly died before
        # it could mark it
        self.section_done()
        return jobs

    def recheck(self):
        """Sort every section's Cloudy outputs again (see Section.recheck).

        A section that changed carries on from its next stage once the run
        is resumed. Its jobs must have stopped.

        Returns:
            list of str
                What each section does next (None when nothing changed).
        """
        self._setup_log("recheck")
        nexts = []
        for section in self.sections:
            self.set_stage("recheck", section)
            nexts.append(section.recheck())
        self.set_stage("recheck")
        if any(nexts) and self._load_state().get("next") == "done":
            self._save_state(next="cloudy")
        if any(nexts):
            log.info(
                "Resume the run to carry on: python -m syncretize.runner "
                f"resume {self.config.path}"
            )
        return nexts

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
        write_run_config(
            grid,
            self.config.path,
            {"Run": run, "Incident": incident, "Cloudy": None},
        )
        log.info(
            f"Made the incident grid {grid} ({_shape(grid)}) in {took(start)}"
        )
        return grid

    def start(self, submit=True):
        """Make the incident grid and set up every Cloudy section.

        This is the start job. The incident grid isn't made when
        ``Incident.grid`` is given (or an earlier start made it), and
        nothing more is done without Cloudy. Each section then writes its
        Cloudy inputs and (with submit) its first round of workers.

        Args:
            submit (bool)
                Submit each section's first round of workers.

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
        if not self.sections:
            self._save_state(grids=[incident_grid], next="done")
            log.info(f"Done: the incident grid is {incident_grid}")
            return incident_grid
        for section in self.sections:
            self.set_stage("start", section)
            # A section a crashed start job had already set up is left as
            # it is, but its first round is submitted if it hadn't been
            if section._load_state().get("next") is None:
                section.setup(incident_grid)
                section._save_state(next="finish")
            if (
                submit
                and section._load_state().get("next") == "finish"
                and not section._round_submitted()
            ):
                section.submit_round()
        self.set_stage("start")
        self._save_state(next="cloudy")
        return incident_grid

    def section_done(self):
        """Mark the run done once every section has made its grid.

        Called by each section's assembly; sections finishing at once may
        both see every section done, which only marks the run done twice.
        """
        states = [s._load_state() for s in self.sections]
        if all(state.get("next") == "done" for state in states):
            grids = [state["grid"] for state in states]
            self._save_state(grids=grids, next="done")
            # About the whole run, not the section that finished last
            self.set_stage("run")
            log.info(
                "Every Cloudy section is done; the grids are:\n"
                + "\n".join(f"  {grid}" for grid in grids)
            )

    def set_stage(self, stage, section=None):
        """Say which stage (and section) run.log lines come from.

        Args:
            stage (str)
                The stage.
            section (Section)
                The section the lines are about, if any.
        """
        logs.set_stage(
            stage,
            os.environ.get("SLURM_JOB_ID"),
            section.label if section is not None else None,
        )

    def _run_here(self):
        """Run the remaining stages in this process, a section at a time.

        Returns:
            list of str
                The final grids.
        """
        if self._load_state()["next"] == "start":
            self.set_stage("start")
            self.start(submit=False)
        for section in self.sections:
            if section._load_state()["next"] != "done":
                # The first round, which the start job submits with Slurm
                if not section._load_state().get("rounds"):
                    self.set_stage("work", section)
                    section.work("local")
                self.set_stage("finish", section)
                section.run_here()
        self.set_stage("run")
        return self._load_state()["grids"]

    def describe(self):
        """What this run is: versions, where it runs, and its whole config.

        Returns:
            str
                The lines run.log starts with.
        """
        config = self.config
        lines = [
            f"syncretize {version('syncretize')}, synthesizer "
            f"{version('synthesizer')}, Python {platform.python_version()}",
            f"On {socket.gethostname()} as {getpass.getuser()} at {_now()}",
            "",
            f"Run config {config.path} (with every default):",
            _dump(
                {
                    "Run": config.run,
                    "Incident": config.incident,
                    **{
                        f"Cloudy{index}": cloudy
                        for index, cloudy in config.sections.items()
                    },
                }
            ),
        ]
        if config.incident["config"] is not None:
            with open(config.incident["config"]) as file:
                # As the config was read, so e.g. 1e6 is a number
                params = yaml.load(file, Loader=YAMLLoader)
            lines += [
                f"Incident model parameters ({config.incident['config']}):",
                _dump(params),
            ]
        lines += [section.describe() for section in self.sections]
        return "\n".join(lines)

    def _setup_log(self, stage):
        """Log to run.log too, unless the command line already set it up.

        Args:
            stage (str)
                The stage logging, e.g. "start".
        """
        if not logging_to(self.config.run["work_dir"]):
            setup(self.config.run["work_dir"], stage)

    def _job_name(self, stage):
        """The start job's name, e.g. "R0-bc03-Incident".

        Args:
            stage (str)
                The stage (only the start stage is the run's own).

        Returns:
            str
                The job's name.
        """
        return f"R0-{self.config.run['name']}-{JOB_KINDS[stage]}"
