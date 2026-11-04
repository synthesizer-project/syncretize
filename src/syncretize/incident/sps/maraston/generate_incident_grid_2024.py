"""
Convert the Maraston (2024) SPS model to an HDF5 synthesizer grid.

There is no public download URL for this model. The source data must be
obtained from the authors and placed in <input-dir>/maraston24.

Example:
    python generate_incident_grid_2024.py maraston24.yaml \
        --grid-dir path/to/grids --input-dir path/to/input
"""

import os

import numpy as np
from synthesizer.conversions import llam_to_lnu
from unyt import Angstrom, Hz, dimensionless, erg, s, yr

from syncretize.grid_io import GridFile
from syncretize.incident.sps.utils import get_model_filename
from syncretize.parser import get_choice, incident_parser, load_param_file


def make_grid(
    model, rotation, model_type, imf, input_dir, grid_dir, param_file=None
):
    """Main function to convert Maraston 2024 and
    produce grids used by synthesizer
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
        grid_dir (string):
            directory where the grids are created.
    Returns:
        fname (string):
            output filename
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

    # Create the GridFile ready to take outputs
    out_grid = GridFile(
        f"{grid_dir}/{synthesizer_model_name}.hdf5", param_file=param_file
    )

    log_on_read = {"ages": True, "metallicities": False}

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

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=lam,
        spectra={"incident": spec * erg / s / Hz},  # check this unit
        alt_axes=("log10ages", "metallicities"),
        log_on_read=log_on_read,
        weight="initial_masses",
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    parser = incident_parser("Maraston (2024) grid creation", input_dir=True)
    args = parser.parse_args()
    params = load_param_file(args.param_file)

    imf_codes = {"kroupa": "kr", "salpeter": "ss"}
    rotation_codes = {0.0: "00", 0.4: "40"}
    imf_type = get_choice(params, "imf_type", imf_codes)
    rotation = rotation_codes[get_choice(params, "rotation", rotation_codes)]
    model_type = get_choice(params, "model_type", ["Te", "Tenc"])

    sps_name = "maraston24"

    input_dir = f"{args.input_dir}/{sps_name}"
    os.makedirs(input_dir, exist_ok=True)

    model = {
        "sps_name": sps_name,
        "sps_version": False,
        "sps_variant": f"{model_type}{rotation}",
        "imf_type": imf_type,
        "imf_masses": [0.1, 100],
        "imf_slopes": False,
        "alpha": False,
    }

    make_grid(
        model,
        rotation,
        model_type,
        imf_codes[imf_type],
        input_dir,
        args.grid_dir,
        param_file=args.param_file,
    )
