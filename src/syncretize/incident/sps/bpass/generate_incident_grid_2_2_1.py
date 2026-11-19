"""
Convert BPASS v2.2.1 outputs to synthesizer grid.

There are two model variants available (with or without binary stars) and
various IMFs.

Individual models can be specified with --model, or all models can be
processed by setting --model=all.

Note: there is no automatic download available so models need to be downloaded
manually from:
https://warwick.ac.uk/fac/sci/physics/research/astro/research/catalogues/bpass/v2p2

Example:
    python generate_incident_grid_2_2_1.py \
    --input-dir path/to/input/dir \
    --grid-dir path/to/grid/dir \
    --models=all
"""

import os
import tarfile

import numpy as np
from unyt import Hz, angstrom, dimensionless, erg, s, yr

from syncretize.incident.builder import (
    GridSpec,
    Output,
    build_grid,
    table_point,
)
from syncretize.incident.sps.utils import get_model_filename
from syncretize.parser import Parser, get_choice, load_params


def resolve_name(original_model_name, bin):
    """Resolve the original BPASS model name into what we need"""

    bpass_imf = original_model_name.split("imf")[-1]
    hmc = float(bpass_imf[-3:])  # high-mass cutoff

    if bpass_imf[:5] == "_chab":
        imf_type = "chabrier03"
        imf_masses = [0.1, hmc]
        imf_slopes = ""
    else:
        imf_type = "bpl"
        imf_masses = [0.1, 1.0, hmc]
        imf_slopes = [1.3, np.round(float(bpass_imf[:3]) / 100 + 1, 2)]

    model = {
        "original_model_name": original_model_name,
        "sps_name": "bpass",
        "sps_version": "2.2.1",
        "sps_variant": bin,
        "imf_type": imf_type,  # named IMF or bpl (broken power law)
        "imf_masses": imf_masses,
        "imf_slopes": imf_slopes,
        "alpha_enhancement": False,
    }

    return model, bpass_imf


def parse_starmass_file(filename):
    """
    Parse a BPASS starmass file.
    """
    data = np.loadtxt(filename).T
    log10ages = data[0]
    stellar_fraction = data[1] / 1e6
    remnant_fraction = data[2] / 1e6
    # the final element is broken so replace with the previous one
    stellar_fraction[-1] = stellar_fraction[-2]

    return log10ages, stellar_fraction, remnant_fraction


def parse_spectra_file(filename):
    """
    Parse a BPASS spectra file.
    """
    data = np.loadtxt(filename).T
    wavelength = data[0]
    spectra = data[1:]

    return wavelength, spectra


def untar_data(model, filename, input_dir):
    model_dir = f"{input_dir}/{model}"
    with tarfile.open(filename) as tar:
        tar.extractall(path=model_dir)

    os.remove(filename)


def read_grid(original_model_name, bin, input_dir):
    """Read one BPASS model's tables into a GridSpec (a single point)."""
    # returns a dictionary containing the sps model parameters
    model, bpass_imf = resolve_name(original_model_name, bin)

    # generate the synthesizer_model_name
    synthesizer_model_name = get_model_filename(model)

    # input directory of this specific bpass model (hence the trailing "_")
    input_dir_ = f"{input_dir}/{model['original_model_name']}"

    # dictionary mapping filename metallicity to float
    map_key_to_met = {
        "zem5": 0.00001,
        "zem4": 0.0001,
        "z001": 0.001,
        "z002": 0.002,
        "z003": 0.003,
        "z004": 0.004,
        "z006": 0.006,
        "z008": 0.008,
        "z010": 0.01,
        "z014": 0.014,
        "z020": 0.020,
        "z030": 0.030,
        "z040": 0.040,
    }

    map_met_to_key = {k: v for v, k in map_key_to_met.items()}
    metallicities = np.sort(np.array(list(map_met_to_key.keys())))

    # get ages and remaining fraction of first metallicity
    fn_ = f"starmass-{bin}-imf{bpass_imf}.zem5.dat"
    log10ages, stellar_fraction_, remnant_fraction_ = parse_starmass_file(
        f"{input_dir_}/{fn_}"
    )

    # open spectra file
    fn_ = f"spectra-{bin}-imf{bpass_imf}.zem5.dat"
    wavelengths, spectra_ = parse_spectra_file(f"{input_dir_}/{fn_}")
    nu = 3e8 / (wavelengths * 1e-10)

    # set up output arrays
    nmetal = len(metallicities)
    na = len(log10ages)
    stellar_fraction = np.zeros((na, nmetal))
    remnant_fraction = np.zeros((na, nmetal))
    spectra = np.zeros((na, nmetal, len(wavelengths)))

    # loop over metallicity
    for imetal, metal in enumerate(metallicities):
        metallicity_key = map_met_to_key[metallicities[imetal]]

        # get ages and remaining fraction
        fn_ = f"starmass-{bin}-imf{bpass_imf}.{metallicity_key}.dat"
        log10ages, stellar_fraction_, remnant_fraction_ = parse_starmass_file(
            f"{input_dir_}/{fn_}"
        )

        # open spectra file
        fn_ = f"spectra-{bin}-imf{bpass_imf}.{metallicity_key}.dat"
        wavelengths, spectra_ = parse_spectra_file(f"{input_dir_}/{fn_}")

        stellar_fraction[:, imetal] = stellar_fraction_
        remnant_fraction[:, imetal] = remnant_fraction_

        for ia, log10age in enumerate(log10ages):
            spectra[ia, imetal, :] = spectra_[ia]  # Lsol AA^-1 10^6 Msol^-1

    # Convert spectra to synthesizer base units
    spectra /= 1e6  # Lsol AA^-1 Msol^-1
    spectra *= 3.826e33  # erg s^-1 AA^-1 Msol^-1
    spectra *= wavelengths / nu  # erg s^-1 Hz^-1 Msol^-1

    # convert log10ages to ages
    ages = 10**log10ages

    return GridSpec(
        grid_name=synthesizer_model_name,
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=wavelengths * angstrom,
        outputs={
            "incident": Output(units=erg / s / Hz),
            "star_fraction": Output(
                spectra=False,
                description="Two-dimensional remaining stellar fraction "
                "grid, [age, Z]",
            ),
            "remnant_fraction": Output(
                spectra=False,
                description="Two-dimensional remaining remnant fraction "
                "grid, [age, Z]",
            ),
        },
        alt_axes=("log10ages", "metallicities"),
        # A dictionary with Boolean values for each axis, where True
        # indicates that the attribute should be interpolated in
        # logarithmic space.
        log_on_read={"ages": True, "metallicities": True},
        weight="initial_masses",
        data={
            "outputs": {
                "incident": spectra,
                "star_fraction": stellar_fraction,
                "remnant_fraction": remnant_fraction,
            }
        },
    )


def setup(config_file, input_dir):
    """Read the config and the BPASS tables (see syncretize.incident.builder).

    Args:
        config_file (str)
            The parameter file.
        input_dir (str)
            The directory holding the source data, in bpass/<bpass_model>.

    Returns:
        GridSpec
            The whole grid, as a single point.
    """
    # open the config file and extract parameters
    config = load_params(config_file, model_keys=["bpass_model", "variant"])

    # get the model chosen in the config file
    model = get_choice(
        config.model,
        "bpass_model",
        [
            "bpass_v2.2.1_imf_chab100",
            "bpass_v2.2.1_imf_chab300",
            "bpass_v2.2.1_imf100_300",
            "bpass_v2.2.1_imf135_300",
            "bpass_v2.2.1_imf170_300",
            "bpass_v2.2.1_imf100_100",
            "bpass_v2.2.1_imf135_100",
            "bpass_v2.2.1_imf170_100",
        ],
    )
    bin = get_choice(config.model, "variant", ["bin", "sin"])
    print(model, bin)
    return read_grid(model, bin, f"{input_dir}/bpass")


#: The whole table is read in setup, so the grid is a single point
generate_point = table_point


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="BPASS_2.2.1 download and grid creation")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # Unpack the arguments
    args = parser.parse_args()

    build_grid(
        setup(args.config_file, args.input_dir),
        generate_point,
        args.grid_dir,
        param_file=args.config_file,
        cores=args.cores,
    )
