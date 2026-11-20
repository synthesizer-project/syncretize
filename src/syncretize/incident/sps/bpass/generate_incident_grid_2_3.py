"""
Process BPASS v2.3 and convert to HDF5 synthesizer grid.

For BPASS v2.3 there is currently only a single variant (bin) and IMF. However
v2.3 has 5 different values of the alpha enhancement. The processing at the
moment create grid files for both individual alpha enhancements and then a 3D
grid where alpha_enhancement is an axis.

--download fetches the tables of the alpha enhancements the config needs from
the BPASS v2.3 release
(https://warwick.ac.uk/fac/sci/physics/research/astro/research/catalogues/bpass/)
into <input-dir>/bpass/bpass_v2.3_imf135_300. The release stores them in 5
separate archives, one for each alpha enhancement, whose tables all go in that
one directory.

Example:
    python generate_incident_grid_2_3.py \
    --input-dir path/to/input/dir \
    --grid-dir path/to/grid/dir \
"""

import os

import numpy as np
from unyt import Hz, angstrom, dimensionless, erg, s, yr

from syncretize.incident.builder import (
    GridSpec,
    Output,
    build_grid,
    table_point,
)
from syncretize.incident.sps.utils import (
    download_sharepoint_file,
    extract_gzipped_tables,
    get_model_filename,
)
from syncretize.parser import Parser, get_choice, load_params


def resolve_name(original_model_name, bin, alpha=False):
    """Resolve the original BPASS model name into what we need. This is
    specific to 2.3, e.g. 'bpass_v2.3_imf135_300'"""

    # bpass_imf = original_model_name.split("_")[-1]
    bpass_imf = original_model_name.split("imf")[-1]
    print("bpass imf:", bpass_imf)
    hmc = float(bpass_imf[-3:])  # high-mass cutoff

    if bpass_imf[:4] == "chab":
        imf_type = "chabrier03"
        imf_masses = [0.1, hmc]
        imf_slopes = False

    else:
        imf_type = "bpl"
        imf_masses = [0.1, 1.0, hmc]
        imf_slopes = [1.3, np.round(float(bpass_imf[:3]) / 100 + 1, 2)]

    model = {
        "original_model_name": original_model_name,
        "sps_name": "bpass",
        "sps_version": "2.3",
        "sps_variant": bin,
        "imf_type": imf_type,  # named IMF or bpl (broken power law)
        "imf_masses": imf_masses,
        "imf_slopes": imf_slopes,
        "alpha_enhancement": alpha,
    }

    print(model)

    return model, bpass_imf


def parse_ionizing_file(filename):
    """
    Parse a BPASS ionizing file.
    """
    data = np.loadtxt(filename).T
    log10ages = data[0]
    ages = 10**log10ages

    return ages


def parse_spectra_file(filename):
    """
    Parse a BPASS spectra file.
    """
    data = np.loadtxt(filename).T
    wavelength = data[0]
    spectra = data[1:]

    return wavelength, spectra


def read_single_alpha_grid(original_model_name, input_dir, ae="+00", bs="bin"):
    """Read a grid for a single alpha enhancement (a single point)."""

    # convert bpass alpha code (e.g. '+02' into a numerical alpha e.g. 0.2)
    alpha = float(ae) / 10.0

    # returns a dictionary containing the sps model parameters
    model, bpass_imf = resolve_name(original_model_name, bs, alpha=alpha)

    print(bpass_imf)

    # generate the synthesizer_model_name
    synthesizer_model_name = get_model_filename(model)

    print(synthesizer_model_name)

    # input directory of this specific bpass model (hence the trailing "_")
    input_dir_ = f"{input_dir}/{model['original_model_name']}"

    # create metallicity grid and dictionary
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

    # get ages
    fn_ = f"ionizing-{bs}-imf{bpass_imf}.a{ae}.zem5.dat"
    ages = parse_ionizing_file(f"{input_dir_}/{fn_}")

    # open spectra file
    fn_ = f"spectra-{bs}-imf{bpass_imf}.a{ae}.zem5.dat"
    wavelengths, spectra_ = parse_spectra_file(f"{input_dir_}/{fn_}")
    nu = 3e8 / (wavelengths * 1e-10)

    # number of metallicities and ages
    nmetal = len(metallicities)
    na = len(ages)

    # set up outputs

    # the ionising photon production rate
    log10_specific_ionising_lum = {}
    for ion in ["HI", "HeII"]:
        log10_specific_ionising_lum[ion] = np.zeros((na, nmetal))

    spectra = np.zeros((na, nmetal, len(wavelengths)))

    for imetal, metal in enumerate(metallicities):
        metallicity_key = map_met_to_key[metallicities[imetal]]

        # get ages and remaining fraction
        fn_ = f"ionizing-{bs}-imf{bpass_imf}.a{ae}.{metallicity_key}.dat"
        ages = parse_ionizing_file(f"{input_dir_}/{fn_}")

        # open spectra file
        fn_ = f"spectra-{bs}-imf{bpass_imf}.a{ae}.{metallicity_key}.dat"
        wavelengths, spectra_ = parse_spectra_file(f"{input_dir_}/{fn_}")

        for ia, _ in enumerate(ages):
            spec_ = spectra_[ia]  # Lsol AA^-1 10^6 Msol^-1

            # convert from Llam to Lnu
            spec_ /= 1e6  # Lsol AA^-1 Msol^-1
            spec_ *= 3.826e33  # erg s^-1 AA^-1 Msol^-1
            spec_ *= wavelengths / nu  # erg s^-1 Hz^-1 Msol^-1
            spectra[ia, imetal, :] = spec_

    return GridSpec(
        grid_name=synthesizer_model_name,
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=wavelengths * angstrom,
        outputs={"incident": Output(units=erg / s / Hz)},
        alt_axes=("log10ages", "metallicities"),
        # A dictionary with Boolean values for each axis, where True
        # indicates that the attribute should be interpolated in
        # logarithmic space.
        log_on_read={"ages": True, "metallicities": True},
        weight="initial_masses",
        data={"outputs": {"incident": spectra}},
    )


def read_full_grid(original_model_name, input_dir, bs="bin"):
    """Read a full grid for different alpha-ehancements (a single point)."""

    # returns a dictionary containing the sps model parameters
    model, bpass_imf = resolve_name(original_model_name, bs)

    # generate the synthesizer_model_name
    synthesizer_model_name = get_model_filename(model)

    # input directory of this specific bpass model (hence the trailing "_")
    input_dir_ = f"{input_dir}/{model['original_model_name']}"

    # create metallicity grid and dictionary
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
    log10metallicities = np.log10(metallicities)

    # create alpha-enhancement grid

    # list of available alpha enhancements
    alpha_enhancements = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])

    # look up dictionary for filename
    ae_to_aek = {-0.2: "-02", 0.0: "+00", 0.2: "+02", 0.4: "+04", 0.6: "+06"}

    # first metallicity
    metalk = map_met_to_key[metallicities[0]]

    # get ages and remaining fraction for first alpha-enhancement and
    # metallicity
    fn_ = f"""ionizing-{bs}-imf{bpass_imf}.a+00.{metalk}.dat"""
    ages = parse_ionizing_file(f"{input_dir_}/{fn_}")

    # open spectra file for first alpha-enhancement and
    # metallicity
    fn_ = f"""spectra-{bs}-imf{bpass_imf}.a+00.{metalk}.dat"""
    wavelengths, spectra_ = parse_spectra_file(f"{input_dir_}/{fn_}")
    nu = 3e8 / (wavelengths * 1e-10)

    na = len(ages)
    nmetal = len(log10metallicities)
    nae = len(alpha_enhancements)

    # set up outputs

    # the ionising photon production rate
    log10_specific_ionising_lum = {}
    log10_specific_ionising_lum["HI"] = np.zeros((na, nmetal, nae))
    log10_specific_ionising_lum["HeII"] = np.zeros((na, nmetal, nae))

    spectra = np.zeros((na, nmetal, nae, len(wavelengths)))

    for imetal, metal in enumerate(metallicities):
        for iae, alpha_enhancement in enumerate(alpha_enhancements):
            aek = ae_to_aek[alpha_enhancement]
            metalk = map_met_to_key[metal]

            # --- get remaining and remnant fraction
            fn_ = f"""ionizing-{bs}-imf{bpass_imf}.a{aek}.{metalk}.dat"""

            # get ages and remaining fraction
            ages = parse_ionizing_file(f"{input_dir_}/{fn_}")

            # open spectra file
            fn_ = f"""spectra-{bs}-imf{bpass_imf}.a{aek}.{metalk}.dat"""
            wavelengths, spectra_ = parse_spectra_file(f"{input_dir_}/{fn_}")

            for ia, age in enumerate(ages):
                spec_ = spectra_[ia]  # Lsol AA^-1 10^6 Msol^-1

                # --- convert from Llam to Lnu
                spec_ /= 1e6  # Lsol AA^-1 Msol^-1
                spec_ *= 3.826e33  # erg s^-1 AA^-1 Msol^-1
                spec_ *= wavelengths / nu  # erg s^-1 Hz^-1 Msol^-1

                spectra[ia, imetal, iae, :] = spec_  # Lsol AA^-1 10^6 Msol^-1

    return GridSpec(
        grid_name=synthesizer_model_name,
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
            "alpha_enhancement": alpha_enhancements * dimensionless,
        },
        wavelength=wavelengths * angstrom,
        outputs={"incident": Output(units=erg / s / Hz)},
        descriptions={"alpha_enhancement": r"alpha ehanncement [\alpha/Fe]"},
        alt_axes=("log10ages", "metallicities", "alpha_enhancements"),
        log_on_read={
            "ages": True,
            "metallicities": False,
            "alpha_enhancement": False,
        },
        weight="initial_masses",
        data={"outputs": {"incident": spectra}},
    )


#: The BPASS v2.3 release's (public) SharePoint folder
RELEASE_URL = (
    "https://livewarwickac.sharepoint.com/:f:/s/"
    "Physics-BinaryPopulationandSpectralSynthesisBPASS/"
    "EvWhfKfIfFJCs-yyo7YztrUBFHtE4QzQk3WDLSf9ytLuEw"
)

#: The BPASS file codes for each alpha enhancement
ALPHA_CODES = {-0.2: "-02", 0.0: "+00", 0.2: "+02", 0.4: "+04", 0.6: "+06"}

#: The only v2.3 model (binary stars, broken power law IMF up to 300 Msun)
MODEL = "bpass_v2.3_imf135_300"


def choices(config_file):
    """The alpha enhancement codes a config needs, and if it is an axis."""
    config = load_params(
        config_file,
        constant_keys=["alpha_enhancement"],
        axis_names=["alpha_enhancement"],
    )
    if "alpha_enhancement" in config.axes:
        if list(config.axes["alpha_enhancement"]) != list(ALPHA_CODES):
            raise ValueError(
                f"The alpha_enhancement axis must be {list(ALPHA_CODES)}"
            )
        return list(ALPHA_CODES.values()), True
    alpha = get_choice(config.constants, "alpha_enhancement", ALPHA_CODES)
    return [ALPHA_CODES[alpha]], False


def download_data(codes, input_dir):
    """Download the BPASS v2.3 spectra and ionising photon tables.

    Args:
        codes (list of str)
            The alpha enhancement codes, e.g. ["+00"].
        input_dir (str)
            The directory to put them in, as <input_dir>/<MODEL>/.
    """
    os.makedirs(input_dir, exist_ok=True)
    for code in codes:
        archive = f"{input_dir}/bpass_v2.3.a{code}.tar.gz"
        download_sharepoint_file(
            RELEASE_URL, f"bpass_v2.3.a{code}.tar.gz", archive
        )
        extract_gzipped_tables(
            archive, f"{input_dir}/{MODEL}", ("spectra-", "ionizing-")
        )
        os.remove(archive)


def setup(config_file, input_dir):
    """Read the config and the BPASS tables (see syncretize.incident.builder).

    Args:
        config_file (str)
            The parameter file.
        input_dir (str)
            The directory holding the source data, in bpass/<model>.

    Returns:
        GridSpec
            The whole grid, as a single point.
    """
    # (the only model available; no single star models)
    bs = "bin"
    input_dir = f"{input_dir}/bpass"

    # make a full 3D grid if alpha enhancement is an axis
    codes, full = choices(config_file)
    if full:
        return read_full_grid(MODEL, input_dir, bs=bs)

    # otherwise a grid with a single alpha enahancement value
    print(codes[0])
    return read_single_alpha_grid(MODEL, input_dir, ae=codes[0], bs=bs)


#: The whole table is read in setup, so the grid is a single point
generate_point = table_point


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="BPASS_2.3 download and grid creation")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # Unpack the arguments
    args = parser.parse_args()

    if args.download:
        download_data(choices(args.config_file)[0], f"{args.input_dir}/bpass")

    build_grid(
        setup(args.config_file, args.input_dir),
        generate_point,
        args.grid_dir,
        param_file=args.config_file,
        cores=args.cores,
    )
