"""
Create a synthesizer incident grid for the cloudy ``blackbody`` input shape.

This writes the spectrum produced by cloudy's ``blackbody`` command
(cloudy documentation, Hazy 1) as an incident grid, so it can be
used like the other incident spectra.

The functional form follows the cloudy c23.01 source code
(PlanckFunction in cont_ffun.cpp), which is the Planck function.

The grid axis ``blackbody_temperature`` is the temperature of the
blackbody (the first parameter of cloudy's ``blackbody`` command).
Synthesizer does not calculate it, so it needs to be set on the emitter
using this grid, or as a fixed parameter of the emission model.

Each spectrum is normalised to a bolometric luminosity of 1 erg/s,
integrated over cloudy's full energy range (10 MHz to 100 MeV).
"""

import numpy as np
import yaml
from synthesizer.utils.integrate import trapezoid
from unyt import Hz, K, c, erg, s

from syncretize.grid_io import GridFile
from syncretize.parser import Parser

# 1 Ryd in Hz and in K (cloudy's TE1RYD = h c R_inf / k)
RYD_HZ = 3.289841960250e15
TE1RYD = 157887.512

# cloudy's energy range (mesh.h): 10 MHz to 100 MeV, in Ryd
X_MIN = 1.0e7 / RYD_HZ
X_MAX = 1.0e8 / 13.605693122994

# The blackbody is set to zero once h nu / k T reaches EXP_LIMIT, the limit
# of cloudy's safe exponential sexp (cddefines.h).
EXP_LIMIT = 84.0

# Spacing of the wavelength grid in dex, as for the cloudy agn incident grid.
DEX_STEP = 0.005


def blackbody_photon_flux(x, blackbody_temperature):
    """
    Return the cloudy blackbody continuum in photons per unit energy.

    This is the quantity PlanckFunction returns, with an arbitrary
    normalisation.

    Args:
        x (array, float):
            Photon energy in Ryd.
        blackbody_temperature (float):
            Blackbody temperature in K.

    Returns:
        array, float:
            Photon flux per unit energy at x.
    """
    x = np.asarray(x, dtype=float)
    fac = TE1RYD * x / blackbody_temperature

    flux = np.zeros_like(x)
    keep = fac < EXP_LIMIT
    flux[keep] = x[keep] ** 2 / np.expm1(fac[keep])

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


def blackbody_lnu(lam, blackbody_temperature):
    """
    Return L_nu of the cloudy blackbody continuum for L_bol = 1 erg/s.

    Args:
        lam (unyt_array):
            Wavelengths, ascending.
        blackbody_temperature (float):
            Blackbody temperature in K.

    Returns:
        array, float:
            L_nu in erg/s/Hz per erg/s of bolometric luminosity.
    """
    nu_hz = (c / lam).to("Hz").value
    x = nu_hz / RYD_HZ

    # f_nu is the photon flux per unit energy times the energy
    lnu = x * blackbody_photon_flux(x, blackbody_temperature)

    # Frequency decreases along the grid, hence the minus sign
    bolometric_luminosity = -trapezoid(lnu, nu_hz)

    return lnu / bolometric_luminosity


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(description="cloudy blackbody model creation.")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # get the arguments
    args = parser.parse_args()

    # open the config file and extract parameters
    with open(args.config_file, "r") as file:
        parameters = yaml.safe_load(file)

    model_name = parameters["model"]

    temperatures = 10 ** np.array(
        parameters["log10blackbody_temperature"], dtype=float
    )

    # synthesizer needs the axis to increase, with no repeated values
    if not np.all(np.diff(temperatures) > 0):
        raise ValueError(
            "log10blackbody_temperature must be strictly increasing"
        )

    # Model definition dictionary
    model = {
        "name": model_name,
        "type": "agn",
        "family": "blackbody",
    }

    # Define the grid filename and path
    out_filename = f"{args.grid_dir}/{model_name}.hdf5"

    lam = wavelength_grid()

    # create the spectra grid, one spectrum per temperature
    spec = np.array(
        [blackbody_lnu(lam, temperature) for temperature in temperatures]
    )

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename)

    # Write everything out that's common to all models
    out_grid.write_grid_common(
        model=model,
        axes={"blackbody_temperature": temperatures * K},
        descriptions={"blackbody_temperature": "Temperature of the blackbody"},
        wavelength=lam,
        log_on_read={"blackbody_temperature": True},
        spectra={"incident": spec * erg / s / Hz},
        weight="bolometric_luminosities",
    )

    # Include the specific ionising photon luminosity
    print("Calculating and saving specific ionising luminosity")
    out_grid.add_specific_ionising_lum()
