"""
Create FSPS synthesizer grids.

Example:
    python generate_incident_grid.py \
    --input-dir path/to/input/dir \
    --grid-dir path/to/grid/dir \
    --include-chabrier \
    --include-imf-variants \
"""

import fsps
import numpy as np
from unyt import Hz, angstrom, dimensionless, erg, s, yr

from syncretize.grid_io import GridFile
from syncretize.incident.sps.fsps_utils import (
    check_fsps_model,
    check_fsps_version,
    fsps_libraries,
)
from syncretize.incident.sps.utils import get_model_filename
from syncretize.parser import Parser, get_choice, load_params


def generate_grid(model):
    """Main function to create fsps grids used by synthesizer"""

    # set up StellarPopulation grid
    imf_type = model["imf_type"]
    # Floats, so the grid name reads e.g. 1.0,120.0 however they were given
    imf_masses = [float(m) for m in model["imf_masses"]]
    model["imf_masses"] = imf_masses

    if imf_type == "chabrier03":
        sp = fsps.StellarPopulation(
            imf_type=1,
            imf_upper_limit=imf_masses[-1],
            imf_lower_limit=imf_masses[0],
        )
    elif imf_type == "bpl":
        imf_slopes = model["imf_slopes"]
        # NOTE THIS ASSUMES boundaries of 0.5, 1.0

        if (imf_masses[1] != 0.5) or (imf_masses[2] != 1.0):
            # raise exception
            print(
                """WARNING: this IMF definition requires that the boundaries
                are [m_low, 0.5, 1.0, m_high]"""
            )

        sp = fsps.StellarPopulation(
            imf_type=2,
            imf_upper_limit=imf_masses[-1],
            imf_lower_limit=imf_masses[0],
            imf1=imf_slopes[0],
            imf2=imf_slopes[1],
            imf3=imf_slopes[2],
        )
    else:
        # raise exception
        pass

    # Record the compiled libraries; the isochrones and spectra name the
    # variant
    libraries = fsps_libraries(sp, config.model.get("libraries"))
    model["fsps_libraries"] = ",".join(libraries)
    model["sps_variant"] = "".join(libraries[:2])
    print(model["sps_variant"])

    # generate the synthesizer_model_name
    synthesizer_model_name = get_model_filename(model)

    # this is the full path to the ultimate HDF5 grid file
    out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

    lam = sp.wavelengths  # units: Angstroms
    log10ages = sp.log_age  # units: log10(years)
    ages = 10**log10ages
    metallicities = sp.zlegend  # units: log10(metal)

    na = len(log10ages)
    nmetal = len(metallicities)

    spec = np.zeros((na, nmetal, len(lam)))
    stellar_fraction = np.zeros((na, nmetal))

    for imetal in range(nmetal):
        spec_ = sp.get_spectrum(zmet=imetal + 1)[1]  # 2D array Lsol / AA
        _mfrac = sp.stellar_mass / sp.formed_mass
        for ia in range(na):
            lnu = spec_[ia]  # Lsol / Hz
            lnu *= 3.826e33  # erg s^-1 Hz^-1 Msol^-1
            spec[ia, imetal] = lnu
            stellar_fraction[ia, imetal] = _mfrac[ia]

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename, param_file=args.config_file)

    print("metallicities:", metallicities)

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
        wavelength=lam * angstrom,
        spectra={"incident": spec * erg / s / Hz},
        alt_axes=("log10ages", "metallicities"),
        log_on_read=log_on_read,
        weight="initial_masses",
    )

    # Write datasets specific to fsps
    out_grid.write_dataset(
        "star_fraction",
        stellar_fraction * dimensionless,
        "Two-dimensional remaining stellar fraction grid, [age, Z]",
        log_on_read=False,
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="FSPS download and grid creation")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # Unpack the arguments
    args = parser.parse_args()
    grid_dir = args.grid_dir

    # open the config file and extract parameters
    config = load_params(
        args.config_file,
        model_keys=["sps_version", "imf_type", "libraries"],
        constant_keys=["imf_masses", "imf_slopes"],
    )

    # The FSPS version and libraries are fixed when FSPS is built, so check
    # the installation matches the config before making anything
    check_fsps_model(config.model)
    check_fsps_version(str(config.model["sps_version"]), fsps.__version__)

    # No download for FSPS
    if args.download:
        print("FSPS is a python package, no download required")

    # Define the parameters of the defaul model. Variants simply update these
    # parameters as required.
    default_model = {
        "sps_name": "fsps",
        # this is set later depending on the isochrones/spectra used
        "sps_variant": False,
        "imf_type": "bpl",  # named IMF or bpl (broken power law)
        "imf_masses": [0.08, 0.5, 1, 120],
        "imf_slopes": [1.3, 2.3, 2.3],
        "alpha_enhancement": False,
        "pyfsps_version": str(fsps.__version__),
        "sps_version": str(config.model["sps_version"]),
    }

    # The IMF chosen in the config file. bpl needs imf_masses
    # [m_low, 0.5, 1.0, m_high] and three imf_slopes; chabrier03 needs
    # imf_masses [m_low, m_high].
    imf_type = get_choice(config.model, "imf_type", ["bpl", "chabrier03"])
    imf_masses = config.constants["imf_masses"]
    imf_slopes = config.constants.get("imf_slopes", [])
    if imf_type == "bpl" and len(imf_slopes) != 3:
        raise ValueError("The bpl IMF needs three imf_slopes")
    if imf_type == "chabrier03" and not (
        len(imf_masses) == 2 and 0 < imf_masses[0] < imf_masses[1]
    ):
        raise ValueError(
            "The chabrier03 IMF needs imf_masses [m_low, m_high] "
            "with 0 < m_low < m_high"
        )
    models = [
        {
            "imf_type": imf_type,
            "imf_masses": imf_masses,
            "imf_slopes": imf_slopes if imf_type == "bpl" else [],
        }
    ]

    # Loop over models
    for model_ in models:
        # Combine default model parameters with variants
        model = default_model | model_
        print(model)

        # Make grid
        generate_grid(model)
