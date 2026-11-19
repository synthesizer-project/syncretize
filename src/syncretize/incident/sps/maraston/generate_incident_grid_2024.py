"""
Download the Maraston2024 SPS model
and convert to HDF5 synthesizer grid.
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


def read_grid(model, rotation, model_type, imf, input_dir):
    """Read the Maraston 2024 tables into a GridSpec (a single point).
    Args:
        model (dict):
            dictionary containing model parameters.
        rotation (string):
            value of stellar rotation for the model,
            "0.00" for no rotation or "0.40" for rotation.
        model_type (string):
            either "" for standard model or "Tenc" for non-corrected
            effective temperatures.
        imf (string):
            either "kr" for a Kroupa IMF or "ss" for Salpeter.
        input_dir (string):
            directory where the raw Maraston+24 files are read from.
    Returns:
        GridSpec
    """

    synthesizer_model_name = get_model_filename(model)
    print(synthesizer_model_name)

    # Array of available metallicities
    metallicities = np.array([0.0003, 0.002, 0.006, 0.014, 0.02])

    # Codes for converting metallicty
    metallicity_code = {
        0.0003: "-1.7",
        0.002: "-1.35",
        0.006: "-0.33",
        0.014: "+0.00",  # solar metallicity
        0.02: "+0.35",
    }

    if model_type == "Tenc":
        model_type = "_Tenc"
    if model_type == "Te":
        model_type = ""

    # Open first raw data file to get age
    fn = (
        f"{input_dir}/sed_ssp_M24_vini0.{rotation}{model_type}_{imf}"
        f"{metallicity_code[metallicities[0]]}"
    )

    ages_, _, lam_, llam_ = np.loadtxt(fn).T  # llam is in (ergs /s /AA /Msun)

    ages_Gyr = np.sort(np.array(list(set(ages_))))  # Gyr

    # Convert units to years
    ages = ages_Gyr * 1e9

    # Get wavelengths for first age
    lam = lam_[ages_ == ages_[0]] * Angstrom

    spec = np.zeros((len(ages), len(metallicities), len(lam)))

    # At each point in spec convert the units
    for imetal, metallicity in enumerate(metallicities):
        for ia, age_Gyr in enumerate(ages_Gyr):
            fn = (
                f"{input_dir}/sed_ssp_M24_vini0.{rotation}{model_type}_{imf}"
                f"{metallicity_code[metallicity]}"
            )
            ages_, _, lam_, llam_ = np.loadtxt(fn).T

            llam = llam_[ages_ == age_Gyr] * erg / s / Angstrom

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
        alt_axes=("log10ages", "metallicities"),
        # A dictionary with Boolean values for each axis, where True
        # indicates that the attribute should be interpolated in
        # logarithmic space.
        log_on_read={"ages": True, "metallicities": False},
        weight="initial_masses",
        data={"outputs": {"incident": spec}},
    )


def setup(config_file, input_dir):
    """Read the config and the Maraston 2024 tables.

    See syncretize.incident.builder.

    Args:
        config_file (str)
            The parameter file.
        input_dir (str)
            The directory holding the source data, in maraston24.

    Returns:
        GridSpec
            The whole grid, as a single point.
    """
    # open the config file and extract parameters
    config = load_params(
        config_file, model_keys=["imf_type", "rotation", "model_type"]
    )

    # The model chosen in the config file
    rotation_codes = {0.0: "00", 0.4: "40"}
    rotation = rotation_codes[
        get_choice(config.model, "rotation", rotation_codes)
    ]
    model_type = get_choice(config.model, "model_type", ["Te", "Tenc"])
    imf_codes = {"kroupa": "kr", "salpeter": "ss"}
    imf_type = get_choice(config.model, "imf_type", imf_codes)

    # Define the model metadata
    model = {
        "sps_name": "maraston24",
        "sps_version": False,
        "sps_variant": f"{model_type}{rotation}",
        "imf_type": imf_type,
        "imf_masses": [0.1, 100],
        "imf_slopes": False,
        "alpha_enhancement": False,
    }
    return read_grid(
        model,
        rotation,
        model_type,
        imf_codes[imf_type],
        f"{input_dir}/maraston24",
    )


#: The whole table is read in setup, so the grid is a single point
generate_point = table_point


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="Maraston+24 download and grid creation")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    args = parser.parse_args()

    # NOTE: there is no public download URL for this model
    # (checked 2026-07-29). The input data must be obtained by other
    # means (e.g. from the authors) and placed in input_dir before
    # running.

    # Create directory to store downloaded output if it doesn't exist
    input_dir = f"{args.input_dir}/maraston24"
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    build_grid(
        setup(args.config_file, args.input_dir),
        generate_point,
        args.grid_dir,
        param_file=args.config_file,
        cores=args.cores,
    )
