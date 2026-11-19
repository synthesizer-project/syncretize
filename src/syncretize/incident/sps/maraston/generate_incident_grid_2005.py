"""
Download M05 and convert to HDF5 synthesizer grid.
"""

import os
import tarfile
from pathlib import Path

import numpy as np
import wget
from synthesizer.conversions import llam_to_lnu
from unyt import Hz, angstrom, dimensionless, erg, s, yr

from syncretize.incident.builder import (
    GridSpec,
    Output,
    build_grid,
    table_point,
)
from syncretize.incident.sps.utils import get_model_filename
from syncretize.parser import Parser, load_params


def download_data(input_dir, url):
    """
    TODO: These could be replaced by our own mirror
    """

    # Download the original data to the working directory
    filename = wget.download(url)

    Path(input_dir).mkdir(parents=True, exist_ok=True)

    # Untar the main directory
    tar = tarfile.open(filename)
    tar.extractall(path=input_dir)
    tar.close()
    os.remove(filename)


def read_grid(model, imf, hr_morphology, input_dir):
    """Read the Maraston 2005 tables into a GridSpec (a single point)."""

    # get synthesizer model name
    synthesizer_model_name = get_model_filename(model)

    print(synthesizer_model_name)

    # NOTE THE LOWEST METALLICITY MODEL DOES NOT HAVE YOUNG AGES so don't use
    metallicities = np.array(
        [0.001, 0.01, 0.02, 0.04]
    )  # array of avialable metallicities

    # Codes for converting metallicty
    metallicity_codes = {
        0.0001: "10m4",
        0.001: "0001",
        0.01: "001",
        0.02: "002",
        0.04: "004",
        0.07: "007",
    }

    # Open first file to get age
    fn = (
        f"{input_dir}/sed.{imf}z{metallicity_codes[metallicities[0]]}."
        f"{hr_morphology}"
    )
    ages_, _, lam_, flam_ = np.loadtxt(fn).T

    ages_Gyr = np.sort(np.array(list(set(ages_))))  # Gyr
    ages = ages_Gyr * 1e9  # yr

    lam = lam_[ages_ == ages_[0]] * angstrom

    spec = np.zeros((len(ages), len(metallicities), len(lam)))

    for imetal, metallicity in enumerate(metallicities):
        for ia, age_Gyr in enumerate(ages_Gyr):
            fn = (
                f"{input_dir}/sed.{imf}z{metallicity_codes[metallicity]}."
                f"{hr_morphology}"
            )
            print(imetal, ia, fn)
            ages_, _, lam_, flam_ = np.loadtxt(fn).T

            llam = flam_[ages_ == age_Gyr] * erg / s / angstrom
            lnu = llam_to_lnu(lam, llam)
            spec[ia, imetal] = lnu

    return GridSpec(
        grid_name=synthesizer_model_name,
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=lam,
        outputs={"incident": Output(units=erg / s / Hz)},
        alt_axes=("ages", "metallicities"),
        # A dictionary with Boolean values for each axis, where True
        # indicates that the attribute should be interpolated in
        # logarithmic space.
        log_on_read={"ages": True, "metallicities": False},
        weight="initial_masses",
        data={"outputs": {"incident": spec}},
    )


def setup(config_file, input_dir):
    """Read the config and the Maraston 2005 tables.

    See syncretize.incident.builder.

    Args:
        config_file (str)
            The parameter file.
        input_dir (str)
            The directory holding the source data, in maraston05.

    Returns:
        GridSpec
            The whole grid, as a single point.
    """
    # open the config file (Maraston 2005 has nothing to choose)
    load_params(config_file)

    # Define the model metadata (the only variant: Salpeter, red horizontal
    # branch)
    model = {
        "sps_name": "maraston05",
        "sps_version": False,
        "alpha_enhancement": False,
        "imf_type": "bpl",
        "imf_masses": [0.1, 100],
        "imf_slopes": [2.35],
        "sps_variant": "rhb",
    }
    return read_grid(model, "ss", "rhb", f"{input_dir}/maraston05")


#: The whole table is read in setup, so the grid is a single point
generate_point = table_point


# Lets include a way to call this script not via an entry point
if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="Maraston download and grid creation")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # Unpack the arguments
    args = parser.parse_args()

    # the directory to store downloaded input files
    input_dir = f"{args.input_dir}/maraston05"

    # create directory to store downloaded output if it doesn't exist
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    # Define the download URL
    # NOTE: the original download URL (icg.port.ac.uk/~maraston/SSPn/SED/
    # Sed_Mar05_SSP_Salpeter.tar.gz) no longer exists (checked 2026-07-29).
    # The input data must be obtained by other means and placed in
    # input_dir before running with --download unset.
    original_data_url = None

    # Download the data if necessary
    if args.download:
        if original_data_url is None:
            raise ValueError(
                "No working download URL is available for "
                "maraston05 (imf=ss). Obtain the input data by "
                f"other means, place it in {input_dir}, and rerun "
                "without --download."
            )
        download_data(input_dir, original_data_url)

    # Get and write the grid
    build_grid(
        setup(args.config_file, args.input_dir),
        generate_point,
        args.grid_dir,
        param_file=args.config_file,
        cores=args.cores,
    )
