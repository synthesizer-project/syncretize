"""
Create a synthesizer incident grid for the RELAGN model of Hagen and Done
(2023) (https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.3455H/abstract).

This uses the relagn module (https://github.com/scotthgn/RELAGN.git). To use
this we first need to download it by cloning the github
repo, i.e.:
git clone https://github.com/scotthgn/RELAGN.git
Since it doesn't have an __init__.py we need to add the path to our python
path. Please follow instructions in the relagn README for this.
This also requires that xspec (https://heasarc.gsfc.nasa.gov/xanadu/xspec/) is
installed.
"""

import numpy as np
from numpy.typing import NDArray

# Import relagn module
from relagn import relagn
from scipy.optimize import brentq
from synthesizer.exceptions import InconsistentParameter
from unyt import Angstrom, Hz, Msun, dimensionless, erg, s

from syncretize.incident.builder import GridSpec, Output, build_grid
from syncretize.parser import Parser, load_params


def rename_dictionary_key(
    dictionary: dict,
    old_key: str,
    new_key: str,
    new_value=None,
) -> dict:
    """Rename a key in a dictionary.

    Attrs:
        dictionary (dict): Input dictionary
        old_key (str): Old key name
        new_key (str): New key name
        new_value: New value for the renamed key

    Returns:
        dictionary (dict): Updated dictionary
    """

    return {
        (new_key if k == old_key else k): (new_value if k == old_key else v)
        for k, v in dictionary.items()
    }


def calc_risco(spin: float) -> float:
    """Calculating innermost stable circular orbit for a spinning
    black hole.

    Attrs:
        spin (float): Dimensionless spin parameter (-0.998 to 0.998)

    Returns:
        risco (float): Innermost stable circular orbit in units of GM/c^2
    """
    Z1 = 1 + (1 - spin**2) ** (1 / 3) * (
        (1 + spin) ** (1 / 3) + (1 - spin) ** (1 / 3)
    )
    Z2 = np.sqrt(3 * spin**2 + Z1**2)

    risco = 3 + Z2 - np.sign(spin) * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))

    return risco


def calc_efficiency(spin: float) -> float:
    """Calculate the accretion efficiency for a given spin.

    Attrs:
        spin (float): Dimensionless spin parameter (0.0 to 0.998)

    Returns:
        eta (float): Accretion efficiency
    """
    risco = calc_risco(spin)

    eta = 1 - np.sqrt(1 - 2 / (3 * risco))

    return eta


def invert_spin(
    eta_target: float, spin_min: float = 0.0, spin_max: float = 0.998
) -> float:
    """Invert efficiency to get spin.

    Attrs:
        eta_target (float): Accretion efficiency (0 to 1)
        spin_min (float): Minimum spin to search, RELAGN only
        valid for prograde spins, so min is 0.0
        spin_max (float): Maximum spin to search

    Returns:
        spin (float): Dimensionless spin parameter (0.0 to 0.998)

    Raises:
        InconsistentParameter: If eta_target is outside achievable
        spin range.
    """

    eta_min = calc_efficiency(spin_min)
    eta_max = calc_efficiency(spin_max)
    if not (eta_min <= eta_target <= eta_max):
        raise InconsistentParameter(
            f"eta_target={eta_target} outside achievable range "
            f"[{eta_min:.4f}, {eta_max:.4f}] for spin in "
            f"[{spin_min}, {spin_max}]"
        )

    def func(spin: float) -> float:
        """Function to find root of."""
        return eta_target - calc_efficiency(spin)

    return brentq(func, spin_min, spin_max, xtol=1e-12, rtol=1e-12)


def compute_single_spectrum(
    mass: float,
    mdot: float,
    spin: float,
    cos_inc: float,
) -> NDArray:
    """Function to calculate SED for a single set of parameters.

    Attrs:
        mass (float): Black hole mass in solar masses.
        mdot (float): Accretion rate in Eddington units.
        spin (float): Dimensionless spin parameter.
        cos_inc (float): Cosine of the inclination angle.

    Returns:
        lnu (NDArray): The computed SED in wavelength order.
    """
    dagn = relagn(
        a=spin,
        cos_inc=cos_inc,
        log_mdot=np.log10(mdot),
        M=mass,
    )

    # Get relativistic sed
    lnu = dagn.get_totSED(rel=True)

    # Return reversed array (wavelength order)
    return lnu[::-1]


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
        "radiative_efficiencies": "radiative efficiency of the accreting "
        "black hole",
    }

    axes_units = {
        "masses": Msun,
        "accretion_rates_eddington": dimensionless,
        "cosine_inclinations": dimensionless,
        "spins": dimensionless,
        "radiative_efficiencies": dimensionless,
    }

    # Set log_on_read, i.e. which axes should be logged when extracted
    # cosine_inclination is set later
    log_on_read = {
        "masses": True,
        "accretion_rates_eddington": True,
        "spins": False,
        "radiative_efficiencies": False,
    }

    # Parameter ranges
    cosine_inclination_range = [0.09, 1.0]
    spin_range = [0.0, 0.998]

    # open the config file and extract parameters
    config = load_params(
        config_file,
        model_keys=["grid_name"],
        constant_keys=["cosine_inclinations"],
        axis_names=[
            "log10masses",
            "log10accretion_rates_eddington",
            "spins",
            "radiative_efficiencies",
            "cosine_inclinations",
        ],
    )
    parameters = {**config.constants, **config.axes}

    model_name = config.model["grid_name"]
    mass = 10 ** np.array(parameters["log10masses"])
    accretion_rate_eddington = 10 ** np.array(
        parameters["log10accretion_rates_eddington"]
    )

    if "spins" in parameters and "radiative_efficiencies" in parameters:
        raise InconsistentParameter(
            "Cannot specify both 'spins' and 'radiative_efficiencies' in "
            "parameters. Please specify only one of these"
        )

    if "spins" not in parameters:
        if "radiative_efficiencies" not in parameters:
            raise InconsistentParameter(
                "Must specify either spins or radiative_efficiencies in "
                "parameters"
            )
        else:
            rad_efficiencies = np.array(parameters["radiative_efficiencies"])
            spin = np.array([invert_spin(eta) for eta in rad_efficiencies])
            print(f"spin={spin} for rad_efficiencies={rad_efficiencies}")
    else:
        spin = np.array(parameters["spins"])

    # check whether isotropic or not
    cos_inc_param = parameters["cosine_inclinations"]
    if np.isscalar(cos_inc_param):
        cosine_inclination = float(cos_inc_param)
        axes_names.remove("cosine_inclinations")
        isotropic = True
    else:
        cosine_inclination = np.array(cos_inc_param)
        isotropic = False

    # Check if parameters within range
    if np.any(spin < spin_range[0]) or np.any(spin > spin_range[1]):
        raise ValueError(
            f"Spin values must be within range {spin_range}, "
            f"but got spins={spin}"
        )

    if np.any(cosine_inclination < cosine_inclination_range[0]) or np.any(
        cosine_inclination > cosine_inclination_range[1]
    ):
        raise ValueError(
            f"Cosine inclination values must be within range "
            f"{cosine_inclination_range}, but got "
            f"cosine_inclinations="
            f"{cosine_inclination}"
        )

    # Model defintion dictionary
    model = {
        "name": model_name,
        "type": "agn",
        "family": "relagn",
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
        axes[axis_name] = axes[axis_name].to(axes_units[axis_name])

    print(axes_values)
    print(axes)

    # initialise default model, to get wavelength grid
    dagn = relagn()
    lam = dagn.wave_grid[::-1] * Angstrom

    if not isotropic:
        log_on_read["cosine_inclinations"] = False

    if "spins" not in parameters:
        if np.atleast_1d(rad_efficiencies).size > 1:
            # Add radiative efficiencies to axes and descriptions.
            # We need to rename the spins axis to radiative_efficiencies,
            # and ensure the same orders of values are maintained since
            # the order was used to set the spectra shape
            axes = rename_dictionary_key(
                axes,
                old_key="spins",
                new_key="radiative_efficiencies",
                new_value=rad_efficiencies * dimensionless,
            )
        else:
            del axes["spins"]

    return GridSpec(
        grid_name=model_name,
        model=model,
        axes=axes,
        descriptions=axes_descriptions,
        wavelength=lam,
        outputs={"incident": Output(units=erg / s / Hz)},
        point_axes=tuple(axes),
        log_on_read=log_on_read,
        weight="None",
        data={
            "axes_values": axes_values,
            "cosine_inclination": cosine_inclination,
        },
    )


def generate_point(spec, index):
    """Return the spectrum at one grid point."""
    axes_values = spec.data["axes_values"]
    # The radiative_efficiencies axis indexes the spins; with a single
    # radiative efficiency there is no spin axis
    spin_index = index.get("spins", index.get("radiative_efficiencies", 0))
    if "cosine_inclinations" in index:
        cos_inc = axes_values["cosine_inclinations"][
            index["cosine_inclinations"]
        ]
    else:
        cos_inc = spec.data["cosine_inclination"]
    return {
        "incident": compute_single_spectrum(
            mass=axes_values["masses"][index["masses"]],
            mdot=axes_values["accretion_rates_eddington"][
                index["accretion_rates_eddington"]
            ],
            spin=axes_values["spins"][spin_index],
            cos_inc=cos_inc,
        )
    }


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="RELAGN AGN model creation.")

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
