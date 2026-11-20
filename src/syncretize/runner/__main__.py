"""Command line for the Runner: python -m syncretize.runner <stage> run.yaml"""

import argparse
import os

from syncretize.runner import Runner
from syncretize.runner.log import log, setup


def main():
    """Run one stage of a run."""
    parser = argparse.ArgumentParser(
        prog="python -m syncretize.runner",
        description="Run a grid from a run config (see "
        "parameter_files/runs/run_template.yaml).",
    )
    parser.add_argument(
        "stage",
        choices=[
            "run",
            "resume",
            "incident",
            "start",
            "work",
            "finish",
            "assemble",
        ],
        help="run: every stage (submits the start job, or runs here with "
        "--local); resume: carry on a stopped run from its next stage; "
        "incident: only the incident grid, here. The Slurm jobs run start "
        "(incident grid, Cloudy inputs, first round), work (one Cloudy "
        "worker), finish (tidy the queue, submit the next job) and assemble "
        "(the final grid)",
    )
    parser.add_argument("run_file", help="The run config")
    parser.add_argument(
        "--local",
        action="store_true",
        help="run, resume: run the stages in this process instead of "
        "submitting jobs",
    )
    parser.add_argument(
        "--walltime",
        help="work: the worker's time limit (it stops claiming models "
        "Run.cloudy.margin before it)",
    )
    args = parser.parse_args()

    runner = Runner(args.run_file)
    job = os.environ.get("SLURM_JOB_ID")
    setup(
        runner.config.run["work_dir"],
        args.stage,
        job,
        # The workers, which run together, keep out of run.log
        run_log=args.stage not in ("work", "incident"),
    )
    if args.stage in ("run", "resume"):
        getattr(runner, args.stage)(local=args.local)
    elif args.stage == "work":
        # An array task is "<array job>_<task>" to sacct
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        job = (
            f"{os.environ['SLURM_ARRAY_JOB_ID']}_{task}"
            if task is not None
            else "local"
        )
        runner.work(job, walltime=args.walltime)
    elif args.stage == "incident":
        runner.incident()
    else:
        # The start, finish and assemble jobs record when they ran and how
        # they ended in the state file
        method = {
            "start": runner.start,
            "finish": runner.continue_run,
            "assemble": runner.assemble,
        }[args.stage]
        runner.job_started(args.stage)
        try:
            method()
        except BaseException as error:
            runner.job_ended(args.stage, f"failed: {error!r}")
            log.error(
                f"The {args.stage} job failed: {error!r}. Once it is fixed, "
                "carry on with: python -m syncretize.runner resume "
                f"{runner.config.path}"
            )
            log.debug("", exc_info=True)
            raise
        runner.job_ended(args.stage, "completed")


# Guarded, since the incident builder's spawned processes import this module
if __name__ == "__main__":
    main()
