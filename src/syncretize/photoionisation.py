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
