"""Command line for the Runner: python -m syncretize.runner <stage> run.yaml"""

import argparse
import os

from syncretize.runner import Runner


def main():
    """Run one stage of a run."""
    parser = argparse.ArgumentParser(
        prog="python -m syncretize.runner",
        description="Run a grid from a run config (see "
        "parameter_files/runs/run_template.yaml).",
    )
    parser.add_argument(
        "stage",
        choices=["run", "incident", "start", "work", "finish", "assemble"],
        help="run: every stage (submits the start job, or runs here with "
        "--local); incident: only the incident grid, here. The Slurm jobs "
        "run start (incident grid, Cloudy inputs, first round), work (one "
        "Cloudy worker), finish (tidy the queue, submit the next job) and "
        "assemble (the final grid)",
    )
    parser.add_argument("run_file", help="The run config")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run every stage in this process instead of submitting jobs",
    )
    parser.add_argument(
        "--walltime",
        help="work: the worker's time limit (it stops claiming models "
        "Run.cloudy.margin before it)",
    )
    args = parser.parse_args()

    runner = Runner(args.run_file)
    if args.stage == "run":
        result = runner.run(local=args.local)
        print(result if args.local else f"Submitted the start job {result}")
    elif args.stage == "work":
        # An array task is "<array job>_<task>" to sacct
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        job = (
            f"{os.environ['SLURM_ARRAY_JOB_ID']}_{task}"
            if task is not None
            else "local"
        )
        runner.work(job, walltime=args.walltime)
    elif args.stage == "finish":
        runner.continue_run()
    else:
        getattr(runner, args.stage)()


# Guarded, since the incident builder's spawned processes import this module
if __name__ == "__main__":
    main()
