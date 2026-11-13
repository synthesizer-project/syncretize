"""
Run cloudy.

Depending on the choice of --incident-index and --photoionisation-index it is
possible to run either a single model (setting both), all models for a given
incident grid point (setting only --incident-index, the recommended approach),
or all models (setting neither).
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import yaml

from syncretize.cloudy.utils import cloudy_run_status

if __name__ == "__main__":
    # Initial parser
    parser = argparse.ArgumentParser(
        description="Run the cloudy models",
    )

    # Add additional parameters which are specific to this script

    # Path to directory where output folders are stored
    parser.add_argument(
        "--cloudy-output-dir",
        type=str,
        required=False,
        default=None,
        help="The path to the output directory where cloudy runs are stored.",
    )

    # The full grid name
    parser.add_argument(
        "--grid-name",
        type=str,
        required=False,
        default=None,
        help="The full grid name.",
    )

    # Path to cloudy directory (not the executable; this is assumed to
    # {cloudy}/{cloudy_version}/source/cloudy.exe)
    parser.add_argument(
        "--cloudy-executable-path",
        type=str,
        required=True,
        help="Path to cloudy directory.",
    )

    # The incident-index. If not set loop over all incident indices. NOTE:
    # this will be slow and should only be used for small grids. In practice
    # this should be argument set by a HPC array job.
    parser.add_argument(
        "--incident-index",
        type=int,
        required=False,
        default=None,
        help="Incident grid point index.",
    )

    # The photoionisation-index. If not set loop over all
    # photoionisation-indicies indices. By default this should be None so
    # that each call to run_cloudy.py loops over all photoionisation models.
    parser.add_argument(
        "--photoionisation-index",
        type=int,
        required=False,
        default=None,
        help="Photoionisation grid point index.",
    )

    # The filename of a list of incident and photoionisation indices to run.
    parser.add_argument(
        "--list-file",
        type=str,
        required=False,
        default=None,
        help="File with one model per row: incident_index "
        "photoionisation_index (e.g. a .failures file).",
    )

    # The index in the list to run.
    parser.add_argument(
        "--list-index",
        type=int,
        required=False,
        default=None,
        help="The specific entry in the list to run.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Shorthand for cloudy_output_dir
    output_directory = f"{args.cloudy_output_dir}/{args.grid_name}"

    # Open the file containing the details of the photoionisation model
    parameter_file = f"{output_directory}/grid_parameters.yaml"
    with open(parameter_file, "r") as file:
        parameters = yaml.safe_load(file)

    # set CLOUDY_DATA_PATH environment variable
    os.environ["CLOUDY_DATA_PATH"] = (
        f"{args.cloudy_executable_path}/"
        f"{parameters['cloudy_version']}/data/:./"
    )

    # Create arrays of the the incident and photoionisation indices for
    # the models we want to run

    # If a list is provided open the file containing the list
    if args.list_file is not None:
        # One row per failed model: incident_index photoionisation_index
        incident_indices, photoionisation_indices = np.loadtxt(
            args.list_file, dtype=int, ndmin=2
        ).T

        # If an index is also provided just choose this model
        if args.list_index is not None:
            incident_indices = [incident_indices[args.list_index]]
            photoionisation_indices = [
                photoionisation_indices[args.list_index]
            ]

    # Otherwise set the list of models to run based on either the number of
    # grid points or a specific incident and photoionisation index
    else:
        # Set incident_indices
        # If an index is provided just use that
        if args.incident_index is not None:
            incident_indices_ = [args.incident_index]
        # Otherwise use the full range
        else:
            incident_indices_ = np.array(
                range(parameters["incident_n_models"])
            )

        # Set photoionisation_indices
        # If an index is provided just use that
        if args.photoionisation_index is not None:
            photoionisation_indices_ = [args.photoionisation_index]
        # Otherwise use the full range
        else:
            photoionisation_indices_ = np.array(
                range(parameters["photoionisation_n_models"])
            )

        # Convert these into a mesh and flatten so that we have a list of every
        # incident and photoionisation index.
        incident_indices, photoionisation_indices = np.meshgrid(
            incident_indices_, photoionisation_indices_
        )

        incident_indices = incident_indices.flatten()
        photoionisation_indices = photoionisation_indices.flatten()

    print(f"incident model indices: {incident_indices}")
    print(f"photoionisation model indices: {photoionisation_indices}")

    # Define the cloudy executable path
    cloudy_executable = (
        f"{args.cloudy_executable_path}/{parameters['cloudy_version']}"
        "/source/cloudy.exe"
    )

    # Loop over the list of indices and run each one, in its incident
    # grid point's directory
    failures = []
    for incident_index, photoionisation_index in zip(
        incident_indices, photoionisation_indices
    ):
        model_dir = f"{output_directory}/{incident_index}"
        command = [cloudy_executable, "-r", str(photoionisation_index)]
        print(" ".join(command), f"(in {model_dir})")
        result = subprocess.run(command, cwd=model_dir)

        # Cloudy can exit 0 without finishing, so also check its output
        reason = cloudy_run_status(f"{model_dir}/{photoionisation_index}")
        if result.returncode != 0:
            reason = f"exit status {result.returncode}" + (
                f", {reason}" if reason else ""
            )
        if reason is not None:
            failures.append((incident_index, photoionisation_index, reason))

    # Report failures and exit non-zero, so a scheduler marks the job
    # as failed
    for incident_index, photoionisation_index, reason in failures:
        print(f"FAILED {incident_index}/{photoionisation_index}: {reason}")
    if failures:
        sys.exit(1)
