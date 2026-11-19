"""
Download Populations 3 star spectra from Yggdrasil models and convert to HDF5
synthesizer grid.

There are 3 versions implemented:
    1. Pop III.1 - A zero-metallicity population with an extremely top-heavy
                   IMF (50-500 Msolar, Salpeter slope), using a SSP from
                   Schaerer et al. (2002, A&A, 382, 28)
    2. Pop III.2 - A zero-metallicity population with a moderately top-heavy
                   IMF (log-normal with characteristic mass M_c=10 Msolar,
                   dispersion sigma=1 Msolar and wings extending from 1-500
                   Msolar) from Raiter et al. (2010, A&A 523, 64)
    3. Pop III, Kroupa IMF - A zero-metallicity population with a normal
                             IMF (universal Kroupa 2001 IMF in the interval
                             0.1-100 Msolar), based on a rescaled SSP from
                             Schaerer et al. (2002, A&A, 382, 28)

We also just pick the instantaneous burst model with the 3 different gas
covering factor of 0 (no nebular contribution), 0.5, 1 (maximal nebular
contribution)

Warning: the nebular procesed grids here differ from the rest of the
nebular processing implementation in synthesizer, where we self consistently
run pure stellar spectra through CLOUDY. For full self consistency the
nebular grids here should not be used, but we provide anyway for reference.

Reference: Zackrisson, E., Rydberg, C.-E., Schaerer, D.,
Ostlin, G. & Tuli, M. 2011, ApJ, 740, 13
https://ui.adsabs.harvard.edu/abs/2011ApJ...740...13Z/abstract
"""

import os
import re

import numpy as np
import requests
from spectres import spectres
from tqdm import tqdm
from unyt import Hz, angstrom, c, dimensionless, erg, s, yr

from syncretize.incident.builder import (
    GridSpec,
    Output,
    build_grid,
    table_point,
)
from syncretize.incident.sps.utils import get_model_filename
from syncretize.parser import Parser, get_choice, load_params


def download_data(input_dir, ver, fcov):
    """
    Function to access Yggdrasil spectra from website
    """
    # Define base path
    filename = f"PopIII{ver}_fcov_{fcov}_SFR_inst_Spectra"
    save_path = input_dir + "/"

    if not os.path.exists(save_path):
        # Create the directory
        print(
            f"Directory for downloading does not exist\ndirectory={save_path}"
        )
        os.makedirs(save_path)
        print("Directory created successfully!")

    # Define the url
    url = f"https://www.astro.uu.se/~ez/yggdrasil/YggdrasilSpectra/{filename}"

    # Call the server and get the response
    response = requests.get(url, stream=True, verify=False)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    # Stream the file to disk with a nice progress bar.
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(save_path + filename, "wb") as f:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)

    return save_path


def convertPOPIII(fileloc):
    """
    Convert POPIII outputs for Yggdrasil
    Wavelength in Angstrom
    Flux is in erg/s/AA
    """

    # Initialise ---------------------------------------------------------
    ageBins = None
    lambdaBins = None
    metalBins = np.array([0])
    seds = np.array([[[None]]])

    print("Reading POPIII files and converting...")
    # Open SED table containing different ages
    print("Converting file ", fileloc)
    data = open(fileloc, "r")
    text = data.read()

    # Get age values
    ages = re.findall(r"Age\s(.*?)\n", text)
    ageBins = np.array(
        [
            float(
                re.findall(
                    r" [+\-]?[^\w]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)",
                    ages[ii],
                )[0]
            )
            for ii in range(len(ages))
        ]
    )

    # Get the number of wavelength points: lam_num
    lam_num = np.array(re.findall(r"points:(.*?)\n", text), dtype=int)
    diff = np.diff(lam_num)
    if np.sum(diff) != 0:
        print("Age bins are not identical everywhere!!!")
        print("CANCELLING CONVERSION!!!")
        return

    seds = np.zeros((len(ageBins), len(metalBins), lam_num[0]))

    """
        Format of the file is 10 header lines at begining followed by
        lam_num lines of wavelength and flux, then one empty line and
        7 string lines giving the ages
    """
    data = open(fileloc, "r")
    tmp = data.readlines()
    mass = float(re.findall(r"\d+\.\d+(?:[eE][+-]?\d+)?", tmp[0])[0])
    begin = 9
    end = begin + lam_num[0]
    for ii in range(len(ageBins)):
        this_data = tmp[begin:end]
        if ii == 0:
            lambdaBins = np.array(
                [
                    float(re.findall(r"[-+]?([0-9]*\.[0-9]+|[0-9]+)", jj)[0])
                    for jj in this_data
                ]
            )

        seds[ii, 0] = np.array(
            [
                float(re.findall(r"[-+]?([0-9]*\.[0-9]+|[0-9]+)", jj)[1])
                * (
                    10
                    ** float(
                        re.findall(r"[-+]?([0-9]*\.[0-9]+|[0-9]+)", jj)[2]
                    )
                )
                for jj in this_data
            ]
        )

        begin = end + 8
        end = begin + lam_num[0]

    return (
        np.array(seds / mass, dtype=np.float64),
        np.array(metalBins, dtype=np.float64),
        np.array(ageBins, dtype=np.float64),
        np.array(lambdaBins, dtype=np.float64),
    )


def read_grid(input_dir, ver, fcov, model, grid_lam):
    """Read one POPIII model into a GridSpec (a single point)."""

    # Generate model filename
    synthesizer_model_name = get_model_filename(model)
    print(synthesizer_model_name)

    # Define input path
    filename = f"PopIII{ver}_fcov_{fcov}_SFR_inst_Spectra"
    input_path = f"{input_dir}/{filename}"

    # Get spectra and attributes
    out = convertPOPIII(input_path)

    metallicities = out[1]

    ages = out[2] * 1e6  # since ages are quoted in Myr

    lam = out[3]

    # Converting L_lam to L_nu using
    # L_lam dlam = L_nu dnu
    # L_nu = L_lam (lam)^2 / c
    # c in units of AA/s for conversion

    light_speed = c.to(angstrom / s).value  # in AA/s
    spec = out[0]

    spec *= (lam**2) / light_speed  # now in erg s^-1 Hz^-1 Msol^-1

    if fcov != "0":
        # Add a suffix to non common nebular model
        add = "" if fcov == "1" else f"_fcov_{fcov}"
        name = f"nebular{add}"
        wavelength = lam * angstrom
    else:
        # incident spectra
        interp_spec = np.zeros((len(ages), len(metallicities), len(grid_lam)))
        for ii, _spec in enumerate(spec):
            interp_spec[ii] = spectres(grid_lam, lam, _spec)
        interp_spec[np.isnan(interp_spec)] = 0.0
        name, spec, wavelength = "incident", interp_spec, grid_lam

    return GridSpec(
        grid_name=synthesizer_model_name,
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=wavelength,
        outputs={name: Output(units=erg / s / Hz)},
        alt_axes=("log10ages", "metallicities"),
        # A dictionary with Boolean values for each axis, where True
        # indicates that the attribute should be interpolated in
        # logarithmic space.
        log_on_read={"ages": True, "metallicities": False},
        weight="initial_masses",
        # Include the specific ionising photon luminosity of the incident
        # spectra
        ionising_luminosity=fcov == "0",
        data={"outputs": {name: spec}},
    )


def choices(config_file):
    """The population version and covering fraction in the config."""
    config = load_params(
        config_file, model_keys=["population", "covering_fraction"]
    )
    populations = {
        "PopIII.1": ".1",
        "PopIII.2": ".2",
        "PopIII_kroupa": "_kroupa_IMF",
    }
    ver = populations[get_choice(config.model, "population", populations)]
    fcov_codes = {0: "0", 0.5: "0.5", 1: "1"}
    fcov = fcov_codes[
        get_choice(config.model, "covering_fraction", fcov_codes)
    ]
    return ver, fcov


def setup(config_file, input_dir):
    """Read the config and the Yggdrasil tables.

    See syncretize.incident.builder.

    Args:
        config_file (str)
            The parameter file.
        input_dir (str)
            The directory holding the source data, in yggdrasil.

    Returns:
        GridSpec
            The whole grid, as a single point.
    """
    ver, fcov = choices(config_file)
    input_dir = f"{input_dir}/yggdrasil"

    # Different forms of the IMFs
    imf_type = {".1": "salpeter", ".2": "lognormal", "_kroupa_IMF": "kroupa"}
    imf_masses = {
        ".1": [50, 500],
        ".2": [10, 1, 500],
        "_kroupa_IMF": [0.1, 100],
    }

    model = {
        "sps_name": "yggdrasil",
        "sps_variant": "PopIII",
        "sps_version": "1.3.3",
        "imf_masses": imf_masses[ver],
        "imf_type": imf_type[ver],
        "alpha_enhancement": False,
    }
    # Update name to include the covering fraction
    # if run has nebular data
    if fcov != "0":
        model.update({"sps_variant": f"POPIII-fcov_{fcov}"})

    # Standardising the wavelength to the nebular grid: the pure stellar
    # (incident) is resampled onto the fcov = 1 wavelengths, which can be
    # done with minimal errors, as it is featureless
    lam = convertPOPIII(f"{input_dir}/PopIII{ver}_fcov_1_SFR_inst_Spectra")[3]

    return read_grid(input_dir, ver, fcov, model, lam * angstrom)


#: The whole table is read in setup, so the grid is a single point
generate_point = table_point


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="Yggdrasil download and grid creation")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # Unpack the arguments
    args = parser.parse_args()

    # Download the data if necessary. The fcov = 1 data are always needed,
    # since their wavelengths are used for every grid.
    if args.download:
        ver, chosen_fcov = choices(args.config_file)
        for fcov in {"1", chosen_fcov}:
            download_data(f"{args.input_dir}/yggdrasil", ver, fcov)

    build_grid(
        setup(args.config_file, args.input_dir),
        generate_point,
        args.grid_dir,
        param_file=args.config_file,
        cores=args.cores,
    )
