"""A submodule containing photoinisation utilities."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from spectres import spectres
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


#: Cloudy's (c23 and c25, default resolution) wavelength mesh below 1000 A
CLOUDY_MESH = Path(__file__).parent / "cloudy" / "cloudy_mesh.dat"


def rebin_ionising(lam, lnu, mesh=None):
    """Rebin the ionising part of an SED onto cloudy's mesh.

    Cloudy interpolates a tabulated SED log-log and samples it at the centres
    of its mesh's cells. Where the ionising spectrum is spikier than the mesh
    (FSPS 3.2's ~9 Angstrom steps, BPASS's 1 Angstrom ones) those samples miss
    up to ~5% of the input's ionising photons, and normalising the outputs to
    the input's Q then scales the whole spectrum up by as much. Giving cloudy
    the spectrum averaged over each of its cells (flux conserving, with
    spectres) makes its samples add up to the input's photons.

    Each stretch between ionisation edges (EDGE_IONS) is rebinned on its own,
    so no cell averages across an edge, and the points either side of an edge
    (see grid_io.add_ionisation_edges) are kept to keep it sharp.

    Args:
        lam (np.ndarray):
            Increasing wavelengths in Angstrom.
        lnu (np.ndarray):
            The SED at lam.
        mesh (np.ndarray):
            Cloudy's mesh (Angstrom); CLOUDY_MESH if None.

    Returns:
        (np.ndarray, np.ndarray)
            The wavelengths and the SED at them: the cell centres replacing
            the input's points between them, the input elsewhere.
    """
    lam, lnu = np.asarray(lam, dtype=float), np.asarray(lnu, dtype=float)
    if mesh is None:
        mesh = np.loadtxt(CLOUDY_MESH)
    edges = sorted(edge_wavelength(ion) for ion in EDGE_IONS)
    bounds = [lam[0]] + [e for e in edges if lam[0] < e < lam[-1]]
    keep = np.ones(lam.size, dtype=bool)
    new_lam, new_lnu = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        inside = (lam >= lo) & (lam <= hi)
        cells = mesh[(mesh > lo) & (mesh < hi)]
        if inside.sum() < 2 or cells.size < 2:
            continue
        # Cells reaching past the stretch's input can't be filled: drop them
        averaged = spectres(
            cells, lam[inside], lnu[inside], fill=np.nan, verbose=False
        )
        good = np.isfinite(averaged)
        if good.sum() < 2:
            continue
        cells, averaged = cells[good], averaged[good]
        keep &= ~((lam > cells[0]) & (lam < cells[-1]))
        new_lam.append(cells)
        new_lnu.append(averaged)
    lam = np.concatenate([lam[keep], *new_lam])
    lnu = np.concatenate([lnu[keep], *new_lnu])
    order = np.argsort(lam)
    return lam[order], lnu[order]


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
