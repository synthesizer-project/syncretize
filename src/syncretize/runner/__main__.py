"""Command line for the Runner: python -m syncretize.runner <stage> run.yaml"""

import argparse

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
        choices=["run", "incident"],
        help="run: every stage; incident: only the incident grid, here",
    )
    parser.add_argument("run_file", help="The run config")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run every stage in this process instead of submitting jobs",
    )
    args = parser.parse_args()

    runner = Runner(args.run_file)
    if args.stage == "run":
        runner.run(local=args.local)
    else:
        runner.incident()


# Guarded, since the incident builder's spawned processes import this module
if __name__ == "__main__":
    main()
