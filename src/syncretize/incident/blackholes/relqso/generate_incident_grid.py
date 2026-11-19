"""
Create a synthesizer incident grid for the RELQSO model of Hagen and Done
(2023) (https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.3455H/abstract).

This uses the relagn module (https://github.com/scotthgn/RELAGN.git). To use
this we first need to download it by cloning the github
repo, i.e.:
git clone https://github.com/scotthgn/RELAGN.git
Since it doesn't have an __init__.py we need to add the path to our python
path.
This also requires that xspec (https://heasarc.gsfc.nasa.gov/xanadu/xspec/) is
installed.
"""

import sys

import numpy as np
from unyt import Angstrom, Hz, Msun, dimensionless, erg, s

from syncretize.incident.builder import GridSpec, Output, build_grid
from syncretize.parser import Parser, load_params

sys.path.append("RELAGN/src/python_version")

# Import relagn module
from relagn import relagn, relqso  # noqa: E402


def setup(config_file, input_dir=None):
    """Read the config and describe the grid (see syncretize.incident.builder).

    Args:
        config_file (str)
            The parameter file.
        input_dir (str)
            Unused; the model needs no source data.

    Returns:
        GridSpec
            The grid, with one point per combination of the axes.
    """

    axes_names = [
        "masses",
        "accretion_rates_eddington",
        "spins",
        "cosine_inclinations",
    ]

    axes_descriptions = {
        "masses": "blackhole mass",
        "accretion_rates_eddington": "BH accretion rate / Eddington accretion"
        " rate [LEdd=eta MdotEdd c^2]",
        "cosine_inclinations": "cosine of the inclination",
        "spins": "dimensionless spin",
    }

    axes_units = {
        "masses": Msun,
        "accretion_rates_eddington": dimensionless,
        "cosine_inclinations": dimensionless,
        "spins": dimensionless,
    }

    # Set log_on_read, i.e. which axes should be logged when extracted
    # cosine_inclination is set later
    log_on_read = {
        "masses": True,
        "accretion_rates_eddington": True,
        "spins": False,
    }

    # open the config file and extract parameters
    config = load_params(
        config_file,
        model_keys=["grid_name"],
        constant_keys=["cosine_inclinations"],
        axis_names=[
            "log10masses",
            "log10accretion_rates_eddington",
            "spins",
            "cosine_inclinations",
        ],
    )
    parameters = {**config.constants, **config.axes}

    model_name = config.model["grid_name"]
    mass = 10 ** np.array(parameters["log10masses"])
    accretion_rate_eddington = 10 ** np.array(
        parameters["log10accretion_rates_eddington"]
    )
    spin = np.array(parameters["spins"])

    # check whether isotropic or not
    if isinstance(parameters["cosine_inclinations"], float):
        cosine_inclination = parameters["cosine_inclinations"]
        axes_names.remove("cosine_inclinations")
        isotropic = True
    else:
        cosine_inclination = np.array(parameters["cosine_inclinations"])
        isotropic = False

    # Model defintion dictionary
    model = {
        "name": model_name,
        "type": "agn",
        "family": "relqso",
    }

    axes_values = {
        "masses": mass,
        "accretion_rates_eddington": accretion_rate_eddington,
        "spins": spin,
    }

    if not isotropic:
        axes_values["cosine_inclinations"] = np.array(cosine_inclination)

    # define axes dictionary which is saved to the HDF5 file
    axes = {}
    for axis_name in axes_names:
        axes[axis_name] = axes_values[axis_name] * axes_units[axis_name]

    print(axes_values)
    print(axes)

    # initialise default model, to get wavelength grid
    dagn = relagn()
    lam = dagn.wave_grid[::-1] * Angstrom

    if not isotropic:
        log_on_read["cosine_inclinations"] = False

    return GridSpec(
        grid_name=model_name,
        model=model,
        axes=axes,
        descriptions=axes_descriptions,
        wavelength=lam,
        outputs={"incident": Output(units=erg / s / Hz)},
        point_axes=tuple(axes_names),
        log_on_read=log_on_read,
        weight="None",
        data={
            "axes_values": axes_values,
            "cosine_inclination": cosine_inclination,
        },
    )


def generate_point(spec, index):
    """Return the spectrum at one grid point."""
    values = {
        name: spec.data["axes_values"][name][i] for name, i in index.items()
    }
    dagn = relqso(
        a=values["spins"],
        cos_inc=values.get(
            "cosine_inclinations", spec.data["cosine_inclination"]
        ),
        log_mdot=np.log10(values["accretion_rates_eddington"]),
        M=values["masses"],
    )

    # get relativistic sed
    lnu = dagn.get_totSED(rel=True)

    # record spectra, making sure to reverse it so it is
    # wavelength order.
    return {"incident": lnu[::-1]}


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="RELQSO AGN model creation.")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # get the arguments
    args = parser.parse_args()

    build_grid(
        setup(args.config_file),
        generate_point,
        args.grid_dir,
        param_file=args.config_file,
        cores=args.cores,
    )
