"""A submodule containing photoinisation utilities."""

from dataclasses import dataclass

import numpy as np
from unyt import c, eV, h


@dataclass
class Ions:
    """A dataclass holding the ionisation energy of various ions.

    Used for calculating ionising photon luminosities (Q).

    Values taken from: \
        https://en.wikipedia.org/wiki/Ionization_energies_of_the_elements_(data_page)
    """

    energy = {
        "HI": 13.6 * eV,
        "HeI": 24.6 * eV,
        "HeII": 54.4 * eV,
        "CII": 24.4 * eV,
        "CIII": 47.9 * eV,
        "CIV": 64.5 * eV,
        "NI": 14.5 * eV,
        "NII": 29.6 * eV,
        "NIII": 47.4 * eV,
        "OI": 13.6 * eV,
        "OII": 35.1 * eV,
        "OIII": 54.9 * eV,
        "NeI": 21.6 * eV,
        "NeII": 41.0 * eV,
        "NeIII": 63.45 * eV,
    }


def calculate_Q_from_U(U_avg, n_h):
    """Calcualte Q for a given U assuming a n_h.

    Args:
        U_avg (float):
            Ionisation parameter
        n_h (float):
            Hyodrogen density (units: cm^-3)

    Returns:
        float:
            Ionising photon luminosity (units: s^-1)
    """
    alpha_B = 2.59e-13  # cm^3 s^-1
    c_cm = 2.99e8 * 100  # cm s^-1
    epsilon = 1.0

    return ((U_avg * c_cm) ** 3 / alpha_B**2) * (
        (4 * np.pi) / (3 * epsilon**2 * n_h)
    )


def calculate_U_from_Q(Q_avg, n_h=100):
    """Calcualte the ionisation parameter for given Q assuming a n_h.

    Args:
        Q_avg (float):
            Ionising photon luminosity (units: s^-1)
        n_h (float):
            Hyodrogen density (units: cm^-3)

    Returns:
        float:
            Ionisation parameter
    """
    alpha_B = 2.59e-13  # cm^3 s^-1
    c_cm = 2.99e8 * 100  # cm s^-1
    epsilon = 1.0

    return ((alpha_B ** (2.0 / 3)) / c_cm) * (
        (3 * Q_avg * (epsilon**2) * n_h) / (4 * np.pi)
    ) ** (1.0 / 3)


#: Ions whose ionisation edges are kept sharp in incident grids (#151).
EDGE_IONS = ("HI", "HeI", "HeII")


def edge_wavelength(ion):
    """Return the wavelength of an ion's ionisation edge in Angstrom.

    Uses Ions.energy, the same threshold the Q calculations use.

    Args:
        ion (str):
            The ion, a key of Ions.energy.

    Returns:
        float:
            The edge wavelength in Angstrom.
    """
    return float((h * c / Ions.energy[ion]).to("Angstrom").value)


def edge_is_resolved(lam, ion="HI", window=1.0):
    """Check that a wavelength grid samples an ionisation edge closely.

    Cloudy interpolates a tabulated SED log-log onto its own mesh, so an input
    with no points close to an edge has that edge smeared out (#151).

    Args:
        lam (np.ndarray):
            Wavelengths in Angstrom.
        ion (str):
            The ion whose edge to check.
        window (float):
            How close (Angstrom) a point must be on each side of the edge.

    Returns:
        bool:
            True if lam has a point within window of the edge on both sides,
            or doesn't span the edge at all.
    """
    lam = np.asarray(lam)
    edge = edge_wavelength(ion)
    if not lam.min() < edge < lam.max():
        return True
    blue = np.any((lam > edge - window) & (lam < edge))
    red = np.any((lam > edge) & (lam < edge + window))
    return bool(blue and red)
