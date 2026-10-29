"""
Create a synthesizer incident grid for the cloudy ``agn`` input shape.

This writes the spectrum produced by cloudy's ``agn`` command
(cloudy documentation, Hazy 1) as an incident grid, so it can be
used like the other incident spectra, rather than using
create_cloudy_input_grid_from_cloudy.py.

The functional form follows the cloudy c23.01 source code (ParseAgn
in parse_agn.cpp and ffun1 in cont_ffun.cpp). Note: there are some
slight differences and additions in the cloudy source code compared
to the documentation. We choose to follow the source code as closely
as possible.

The grid axis ``blue_bump_temperature`` is the temperature of the big blue
bump, which sets where the bump turns over at high energies (the first
parameter of cloudy's ``agn`` command). Black holes using this grid will
either use the default temperature calculated following Inayoshi and
Maiolino (2025) (https://ui.adsabs.harvard.edu/abs/arXiv:2409.07805/abstract),
or take the value of ``blue_bump_temperature`` if it is set by the user.

Each spectrum is normalised to a bolometric luminosity of 1 erg/s,
integrated over cloudy's full energy range (10 MHz to 100 MeV).
"""

import itertools

import numpy as np
import yaml
from unyt import Hz, K, c, dimensionless, erg, s

from syncretize.grid_io import GridFile
from syncretize.parser import Parser

# 1 Ryd in Hz and in K (cloudy's TE1RYD = h c R_inf / k)
RYD_HZ = 3.289841960250e15
TE1RYD = 157887.512

# cloudy's energy range (mesh.h): 10 MHz to 100 MeV, in Ryd
X_MIN = 1.0e7 / RYD_HZ
X_MAX = 1.0e8 / 13.605693122994

# cloudy's sexp(x) returns exactly zero beyond this
SEXP_LIMIT = 84.0

# Spacing of the wavelength grid in dex. cloudy interpolates the incident
# spectrum in log-log space, so this is fine enough to follow the blue bump's
# exponential cutoff at 10^4 K to better than 0.1%.
DEX_STEP = 0.005


def sexp(x):
    """
    Return exp(-x), set to zero beyond cloudy's limit as its sexp does.

    Args:
        x (array-like, float):
            The argument of the exponential.

    Returns:
        array, float:
            exp(-x), or zero where x >= 84.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    keep = x < SEXP_LIMIT
    out[keep] = np.exp(-x[keep])
    return out


def agn_photon_flux(x, blue_bump_temperature, alpha_ox, alpha_uv, alpha_x):
    """
    Return the cloudy agn continuum in photons per unit energy.

    This is the quantity ffun1 returns, with an arbitrary normalisation.

    Args:
        x (array, float):
            Photon energy in Ryd.
        blue_bump_temperature (float):
            Big blue bump temperature in K.
        alpha_ox (float):
            Optical to X-ray index.
        alpha_uv (float):
            Low energy slope of the big blue bump.
        alpha_x (float):
            Slope of the X-ray power law.

    Returns:
        array, float:
            Photon flux per unit energy at x.
    """
    x = np.asarray(x, dtype=float)

    # Blue bump cutoff energy in Ryd
    x_bb = blue_bump_temperature / TE1RYD

    # Normalisation of the X-ray power law from alpha_ox (ParseAgn)
    ratio = 403.3 ** (alpha_ox - 1.0)
    blue_bump = 0.3645 ** (alpha_uv - 1.0) * sexp(0.3645 / x_bb)
    xrays = 147.0 ** (alpha_x - 1.0)
    if blue_bump <= 0.0:
        raise ValueError(
            "The blue bump is zero at 0.3645 Ryd for "
            f"T = {blue_bump_temperature} K"
        )
    a = ratio / (xrays / blue_bump)

    # Blue bump with its infrared cutoff
    flux = x ** (alpha_uv - 1.0) * sexp(x / x_bb) * sexp(0.01 / x)

    # X-ray power law, only above 0.1 Ryd, falling as nu^-2 above 100 keV
    mid = (x > 0.1) & (x < 7350.0)
    high = x >= 7350.0
    flux[mid] += x[mid] ** (alpha_x - 1.0) * a * sexp(1.0 / x[mid])
    flux[high] += 7350.0 ** (alpha_x - 1.0) * a / (x[high] / 7350.0) ** 3

    return flux


def wavelength_grid(dex_step=DEX_STEP):
    """
    Return a log-uniform wavelength grid covering cloudy's energy range.

    Covering the full range stops cloudy padding the incident spectrum with
    zeros (Hazy 1, section 6.1).

    Args:
        dex_step (float):
            The spacing of the grid in dex.

    Returns:
        unyt_array:
            Wavelengths in Angstrom, ascending.
    """
    x = 10 ** np.arange(np.log10(X_MIN), np.log10(X_MAX), dex_step)
    nu = x[::-1] * RYD_HZ * Hz
    return (c / nu).to("Angstrom")


def agn_lnu(lam, blue_bump_temperature, alpha_ox, alpha_uv, alpha_x):
    """
    Return L_nu of the cloudy agn continuum for L_bol = 1 erg/s.

    Args:
        lam (unyt_array):
            Wavelengths, ascending.
        blue_bump_temperature (float):
            Big blue bump temperature in K.
        alpha_ox (float):
            Optical to X-ray index.
        alpha_uv (float):
            Low energy slope of the big blue bump.
        alpha_x (float):
            Slope of the X-ray power law.

    Returns:
        array, float:
            L_nu in erg/s/Hz per erg/s of bolometric luminosity.
    """
    nu_hz = (c / lam).to("Hz").value
    x = nu_hz / RYD_HZ

    # f_nu is the photon flux per unit energy times the energy
    lnu = x * agn_photon_flux(
        x, blue_bump_temperature, alpha_ox, alpha_uv, alpha_x
    )

    # Frequency decreases along the grid, hence the minus sign
    bolometric_luminosity = -np.trapezoid(lnu, nu_hz)

    return lnu / bolometric_luminosity


if __name__ == "__main__":
    axes_descriptions = {
        "blue_bump_temperature": "Temperature of the big blue bump",
        "alpha_ox": "Optical to X-ray index between 2500 A and 2 keV",
        "alpha_uv": "Low energy slope of the big blue bump",
        "alpha_x": "Slope of the X-ray power law",
    }

    axes_units = {
        "blue_bump_temperature": K,
        "alpha_ox": dimensionless,
        "alpha_uv": dimensionless,
        "alpha_x": dimensionless,
    }

    # Set log_on_read, i.e. which axes should be logged when extracted
    log_on_read = {
        "blue_bump_temperature": True,
        "alpha_ox": False,
        "alpha_uv": False,
        "alpha_x": False,
    }

    # Set up the command line arguments
    parser = Parser(description="cloudy agn continuum model creation.")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # get the arguments
    args = parser.parse_args()

    # open the config file and extract parameters
    with open(args.config_file, "r") as file:
        parameters = yaml.safe_load(file)

    model_name = parameters["model"]

    # Each parameter is an axis if given as a list and fixed otherwise
    values = {
        "blue_bump_temperature": 10
        ** np.array(parameters["log10blue_bump_temperature"], dtype=float),
        "alpha_ox": np.array(parameters["alpha_ox"], dtype=float),
        "alpha_uv": np.array(parameters.get("alpha_uv", -0.5), dtype=float),
        "alpha_x": np.array(parameters.get("alpha_x", -1.0), dtype=float),
    }
    axes_names = [name for name, value in values.items() if value.ndim]
    fixed = {
        name: float(value) for name, value in values.items() if not value.ndim
    }

    # Model definition dictionary, including any fixed parameters
    model = {
        "name": model_name,
        "type": "agn",
        "family": "cloudy_agn",
        **fixed,
    }

    # Define the grid filename and path
    out_filename = f"{args.grid_dir}/{model_name}.hdf5"

    # define axes dictionary which is saved to the HDF5 file
    axes = {name: values[name] * axes_units[name] for name in axes_names}

    lam = wavelength_grid()

    # create empty spectra grid
    axes_shape = [len(values[name]) for name in axes_names]
    spec = np.zeros((*axes_shape, len(lam)))

    # loop over every grid point
    for index in itertools.product(*[range(n) for n in axes_shape]):
        point = dict(fixed)
        for name, i in zip(axes_names, index):
            point[name] = values[name][i]
        spec[index] = agn_lnu(lam, **point)

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    # Write everything out that's common to all models
    out_grid.write_grid_common(
        model=model,
        axes=axes,
        descriptions={name: axes_descriptions[name] for name in axes_names},
        wavelength=lam,
        log_on_read={name: log_on_read[name] for name in axes_names},
        spectra={"incident": spec * erg / s / Hz},
        weight="bolometric_luminosities",
    )

    # Include the specific ionising photon luminosity
    print("Calculating and saving specific ionising luminosity")
    out_grid.add_specific_ionising_lum()
