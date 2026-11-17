"""
Convert the Maraston (2005) SPS model to an HDF5 synthesizer grid.

The original download URL no longer exists, so the source data must be
obtained by other means and placed in <input-dir>/maraston05.

Example:
    python generate_incident_grid_2005.py maraston05.yaml \
        --grid-dir path/to/grids --input-dir path/to/input
"""

import os
import tarfile
from pathlib import Path

import numpy as np
import wget
from synthesizer.conversions import llam_to_lnu
from unyt import Hz, angstrom, dimensionless, erg, s, yr

from syncretize.grid_io import GridFile
from syncretize.incident.sps.utils import get_model_filename
from syncretize.parser import get_choice, incident_parser, load_param_file


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


def make_grid(model, imf, hr_morphology, input_dir, grid_dir, param_file=None):
    """Main function to convert BC03 grids and
    produce grids used by synthesizer"""

    # get synthesizer model name
    synthesizer_model_name = get_model_filename(model)

    print(synthesizer_model_name)

    # Define output
    out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

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

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename, param_file=param_file)

    # A dictionary with Boolean values for each axis, where True
    # indicates that the attribute should be interpolated in
    # logarithmic space.
    log_on_read = {"ages": True, "metallicities": False}

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=lam,
        spectra={"incident": spec * erg / s / Hz},
        alt_axes=("ages", "metallicities"),
        log_on_read=log_on_read,
        weight="initial_masses",
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    parser = incident_parser(
        "Maraston (2005) grid creation", input_dir=True, download=True
    )
    args = parser.parse_args()
    params = load_param_file(args.param_file)

    # Only one variant is currently supported, but check the parameter file
    # asks for it
    get_choice(params, "imf_type", ["salpeter"])
    hr_morphology = get_choice(params, "hr_morphology", ["rhb"])
    imf = "ss"

    sps_name = "maraston05"

    # append sps_name to input_dir to define where to store downloaded input
    # files
    input_dir = f"{args.input_dir}/{sps_name}"
    os.makedirs(input_dir, exist_ok=True)

    # NOTE: the original download URL (icg.port.ac.uk/~maraston/SSPn/SED/
    # Sed_Mar05_SSP_Salpeter.tar.gz) no longer exists (checked 2026-07-29).
    original_data_url = None
    if args.download:
        if original_data_url is None:
            raise ValueError(
                "No working download URL is available for maraston05. "
                f"Obtain the input data by other means, place it in "
                f"{input_dir}, and rerun without --download."
            )
        download_data(input_dir, original_data_url)

    model = {
        "sps_name": sps_name,
        "sps_version": False,
        "sps_variant": hr_morphology,
        "imf_type": "bpl",
        "imf_masses": [0.1, 100],
        "imf_slopes": [2.35],
        "alpha_enhancement": False,
    }

    make_grid(
        model,
        imf,
        hr_morphology,
        input_dir,
        args.grid_dir,
        param_file=args.param_file,
    )
