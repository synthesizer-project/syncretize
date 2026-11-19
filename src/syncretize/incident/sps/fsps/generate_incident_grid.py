"""
Create FSPS synthesizer grids.

Example:
    python generate_incident_grid.py \
    --grid-dir path/to/grid/dir \
    --config-file \
    parameter_files/incident/sps/fsps/generate_incident_grid_template.yaml \
    [--cores 4]
"""

import fsps
from unyt import Hz, angstrom, dimensionless, erg, s, yr

from syncretize.incident.builder import GridSpec, Output, build_grid
from syncretize.incident.sps.fsps_utils import (
    check_fsps_model,
    check_fsps_version,
    fsps_libraries,
)
from syncretize.incident.sps.utils import get_model_filename
from syncretize.parser import Parser, get_choice, load_params

#: StellarPopulations already built in this process, by their arguments
_STELLAR_POPULATIONS = {}


def stellar_population(sp_args):
    """Return a StellarPopulation for these arguments, building it once."""
    key = tuple(sorted(sp_args.items()))
    if key not in _STELLAR_POPULATIONS:
        _STELLAR_POPULATIONS[key] = fsps.StellarPopulation(**sp_args)
    return _STELLAR_POPULATIONS[key]


def setup(config_file, input_dir=None):
    """Read the config and describe the grid (see syncretize.incident.builder).

    Args:
        config_file (str)
            The parameter file.
        input_dir (str)
            Unused; FSPS has no source data to read.

    Returns:
        GridSpec
            The grid, with one point per metallicity.
    """
    # open the config file and extract parameters
    config = load_params(
        config_file,
        model_keys=["sps_version", "imf_type", "libraries"],
        constant_keys=["imf_masses", "imf_slopes"],
    )

    # The FSPS version and libraries are fixed when FSPS is built, so check
    # the installation matches the config before making anything
    check_fsps_model(config.model)
    check_fsps_version(str(config.model["sps_version"]), fsps.__version__)

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

    model = {
        "sps_name": "fsps",
        # this is set later depending on the isochrones/spectra used
        "sps_variant": False,
        "imf_type": imf_type,
        "imf_masses": imf_masses,
        "imf_slopes": imf_slopes if imf_type == "bpl" else [],
        "alpha_enhancement": False,
        "pyfsps_version": str(fsps.__version__),
        "sps_version": str(config.model["sps_version"]),
    }
    print(model)

    # Floats, so the grid name reads e.g. 1.0,120.0 however they were given
    imf_masses = [float(m) for m in model["imf_masses"]]
    model["imf_masses"] = imf_masses

    if imf_type == "chabrier03":
        sp_args = {
            "imf_type": 1,
            "imf_upper_limit": imf_masses[-1],
            "imf_lower_limit": imf_masses[0],
        }
    else:
        imf_slopes = model["imf_slopes"]
        # NOTE THIS ASSUMES boundaries of 0.5, 1.0
        if (imf_masses[1] != 0.5) or (imf_masses[2] != 1.0):
            print(
                """WARNING: this IMF definition requires that the boundaries
                are [m_low, 0.5, 1.0, m_high]"""
            )
        sp_args = {
            "imf_type": 2,
            "imf_upper_limit": imf_masses[-1],
            "imf_lower_limit": imf_masses[0],
            "imf1": imf_slopes[0],
            "imf2": imf_slopes[1],
            "imf3": imf_slopes[2],
        }
    sp = stellar_population(sp_args)

    # Record the compiled libraries; the isochrones and spectra name the
    # variant
    libraries = fsps_libraries(sp, config.model.get("libraries"))
    model["fsps_libraries"] = ",".join(libraries)
    model["sps_variant"] = "".join(libraries[:2])
    print(model["sps_variant"])
    print("metallicities:", sp.zlegend)

    return GridSpec(
        grid_name=get_model_filename(model),
        model=model,
        axes={
            "ages": 10**sp.log_age * yr,
            "metallicities": sp.zlegend * dimensionless,
        },
        wavelength=sp.wavelengths * angstrom,
        outputs={
            "incident": Output(units=erg / s / Hz),
            "star_fraction": Output(
                spectra=False,
                description="Two-dimensional remaining stellar fraction "
                "grid, [age, Z]",
            ),
        },
        # FSPS computes every age for one metallicity at once
        point_axes=("metallicities",),
        alt_axes=("log10ages", "metallicities"),
        log_on_read={"ages": True, "metallicities": False},
        weight="initial_masses",
        data={"sp_args": sp_args},
    )


def generate_point(spec, index):
    """Return every age's spectrum and stellar fraction at one metallicity."""
    sp = stellar_population(spec.data["sp_args"])
    spectra = sp.get_spectrum(zmet=index["metallicities"] + 1)[1]
    return {
        # Lsol / Hz to erg s^-1 Hz^-1 Msol^-1
        "incident": spectra * 3.826e33,
        "star_fraction": sp.stellar_mass / sp.formed_mass,
    }


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="FSPS download and grid creation")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # Unpack the arguments
    args = parser.parse_args()

    # No download for FSPS
    if args.download:
        print("FSPS is a python package, no download required")

    # Make grid
    build_grid(
        setup(args.config_file),
        generate_point,
        args.grid_dir,
        param_file=args.config_file,
        cores=args.cores,
    )
