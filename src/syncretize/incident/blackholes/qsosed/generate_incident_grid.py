"""
Create a synthesizer incident grid for the QSOSED model of Kubota and Done
(2018) (https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.1247K/abstract).

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
from synthesizer.utils.integrate import trapezoid
from unyt import Angstrom, Hz, Msun, c, dimensionless, erg, s

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
            The grid, with one point per mass and accretion rate.
    """

    axes_names = ["masses", "accretion_rates_eddington", "cosine_inclinations"]

    axes_descriptions = {
        "masses": "blackhole masses",
        "accretion_rates_eddington": "BH accretion rate / Eddington accretion"
        " rate [LEdd=eta MdotEdd c^2]",
        "cosine_inclinations": "cosine of the inclination",
    }

    axes_units = {
        "masses": Msun,
        "accretion_rates_eddington": dimensionless,
        "cosine_inclinations": dimensionless,
    }

    # open the config file and extract parameters
    config = load_params(
        config_file,
        model_keys=["grid_name"],
        constant_keys=["cosine_inclinations"],
        axis_names=[
            "log10masses",
            "log10accretion_rates_eddington",
            "cosine_inclinations",
        ],
    )
    parameters = {**config.constants, **config.axes}

    model_name = config.model["grid_name"]
    mass = 10 ** np.array(parameters["log10masses"])
    accretion_rate_eddington = 10 ** np.array(
        parameters["log10accretion_rates_eddington"]
    )

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
        "family": "qsosed",
    }

    # Define axes values
    axes_values = {
        "masses": mass,
        "accretion_rates_eddington": accretion_rate_eddington,
    }

    if not isotropic:
        axes_values["cosine_inclinations"] = cosine_inclination

    # Set log_on_read, i.e. which axes should be logged when extracted
    # cosine_inclination is set later
    log_on_read = {
        "masses": True,
        "accretion_rates_eddington": True,
    }

    # define axes dictionary which is saved to the HDF5 file
    axes = {}
    for axis_name in axes_names:
        axes[axis_name] = axes_values[axis_name] * axes_units[axis_name]

    axes["masses"].to("Msun")

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
        # Each mass and accretion rate is normalised using its own
        # isotropic spectrum, so the inclinations come as a block
        point_axes=("masses", "accretion_rates_eddington"),
        log_on_read=log_on_read,
        weight="bolometric_luminosities",
        data={
            "axes_values": axes_values,
            "isotropic": isotropic,
            "cosine_inclination": cosine_inclination,
        },
    )


def generate_point(spec, index):
    """Return the normalised spectra for one mass and accretion rate."""
    axes_values = spec.data["axes_values"]
    mass_ = axes_values["masses"][index["masses"]]
    accretion_rate_eddington_ = axes_values["accretion_rates_eddington"][
        index["accretion_rates_eddington"]
    ]
    isotropic = spec.data["isotropic"]
    cosine_inclination = spec.data["cosine_inclination"]
    nu = c / spec.wavelength
    nu_hz = nu.to("Hz").value

    def qsosed_lnu(cosine_inclination_):
        # spin is assumed to be zero here
        dagn = relqso(
            a=0.0,
            cos_inc=cosine_inclination_,
            log_mdot=np.log10(accretion_rate_eddington_),
            M=mass_,
        )

        # lnu = dagn.get_totSED(rel=True) # relativistic
        lnu = dagn.get_totSED(rel=False)  # non-relativistic

        return lnu[::-1]

    # Normalise the spectra using the isotropic spectra (i.e.
    # cosine_inclination=0.5)

    # If only a grid containing isoptropic spectra this simply means
    # normalising every spectra to unit
    if isotropic:
        spec_ = qsosed_lnu(cosine_inclination)
        bolometric_luminosity = -trapezoid(spec_, nu_hz)
        return {"incident": spec_ / bolometric_luminosity}

    # Otherwise identify the index correspinding and divide all by this
    spec_ = np.array([qsosed_lnu(cos_inc) for cos_inc in cosine_inclination])

    # Find the index corresponding to cosine_inclination=0.5
    isotropic_index = np.where(cosine_inclination == 0.5)[0]

    # Determine the bolometric luminosity
    bolometric_luminosity = -trapezoid(spec_[isotropic_index], nu_hz)

    return {"incident": spec_ / bolometric_luminosity}


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="QSOSED AGN model creation.")

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
