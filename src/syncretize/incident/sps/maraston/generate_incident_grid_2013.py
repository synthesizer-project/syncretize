"""
Download the Maraston2013 SPS model and convert to HDF5 synthesizer grid.
"""

import os

import numpy as np
from synthesizer.conversions import llam_to_lnu
from unyt import Angstrom, Hz, dimensionless, erg, s, yr

from syncretize.incident.builder import (
    GridSpec,
    Output,
    build_grid,
    table_point,
)
from syncretize.incident.sps.utils import get_model_filename
from syncretize.parser import Parser, get_choice, load_params

#: Codes for each IMF in the Maraston 2013 file names
imf_code = {"salpeter": "ss", "kroupa": "kr"}


def read_grid(model, imf, input_dir):
    """Read the Maraston 2013 tables into a GridSpec (a single point).
    Args:
        synthesizer_model_name (string):
            name of the model to be saved.
        imf (string):
            Initial mass function, Salpeter or
            Kroupa
        input_dir (string):
            directory where the raw Maraston+13 files are read from
    Returns:
        GridSpec
    """

    synthesizer_model_name = get_model_filename(model)
    print(synthesizer_model_name)

    metallicities = np.array(
        [0.001, 0.01, 0.02, 0.04]
    )  # array of available metallicities

    metallicity_codes = {
        0.001: "0001",
        0.01: "001",
        0.02: "002",
        0.04: "004",
    }  # codes for converting metallicty

    # open first raw data file to get age
    fn = (
        f"{input_dir}/sed_M13.{imf_code[imf]}"
        f"z{metallicity_codes[metallicities[0]]}"
    )

    ages_, _, lam_, llam_ = np.loadtxt(fn).T  # llam is in (ergs /s /AA /Msun)

    ages_Gyr = np.sort(np.array(list(set(ages_))))  # Gyr
    ages = ages_Gyr * 1e9

    lam = lam_[ages_ == ages_[0]] * Angstrom

    spec = np.zeros((len(ages), len(metallicities), len(lam)))

    # at each point in spec convert the units
    for iZ, Z in enumerate(metallicities):
        for ia, age_Gyr in enumerate(ages_Gyr):
            fn = f"{input_dir}/sed_M13.{imf_code[imf]}z{metallicity_codes[Z]}"
            print(iZ, ia, fn)
            ages_, _, lam_, llam_ = np.loadtxt(fn).T

            llam = llam_[ages_ == age_Gyr] * erg / s / Angstrom
            lnu = llam_to_lnu(lam, llam)
            spec[ia, iZ] = lnu

    return GridSpec(
        grid_name=synthesizer_model_name,
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=lam,
        outputs={"incident": Output(units=erg / s / Hz)},
        alt_axes=("log10ages", "metallicities"),
        # A dictionary with Boolean values for each axis, where True
        # indicates that the attribute should be interpolated in
        # logarithmic space.
        log_on_read={"ages": True, "metallicities": False},
        weight="initial_masses",
        data={"outputs": {"incident": spec}},
    )


def setup(config_file, input_dir):
    """Read the config and the Maraston 2013 tables.

    See syncretize.incident.builder.

    Args:
        config_file (str)
            The parameter file.
        input_dir (str)
            The directory holding the source data, in maraston13.

    Returns:
        GridSpec
            The whole grid, as a single point.
    """
    # open the config file and extract parameters
    config = load_params(config_file, model_keys=["imf_type"])

    # The IMF chosen in the config file
    imf = get_choice(config.model, "imf_type", ["salpeter", "kroupa"])

    # Define the model metadata
    model = {
        "sps_name": "maraston13",
        "sps_version": False,
        "sps_variant": False,
        "imf_type": imf,
        "imf_masses": [0.1, 100],
        "imf_slopes": False,
        "alpha_enhancement": False,
    }
    return read_grid(model, imf, f"{input_dir}/maraston13")


#: The whole table is read in setup, so the grid is a single point
generate_point = table_point


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="Maraston+13 download and grid creation")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    args = parser.parse_args()

    # NOTE: the original download site (icg.port.ac.uk/~maraston) no
    # longer exists (checked 2026-07-29). The input data must be obtained
    # by other means and placed in input_dir before running.

    # create directory to store downloaded output if it doesn't exist
    input_dir = f"{args.input_dir}/maraston13"
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    build_grid(
        setup(args.config_file, args.input_dir),
        generate_point,
        args.grid_dir,
        param_file=args.config_file,
        cores=args.cores,
    )
