"""Check syncretize uses its own cloudy modules."""

from pathlib import Path

import numpy as np

from syncretize import cloudy

SRC = Path(__file__).parents[1] / "src" / "syncretize"


def test_cloudy_modules_are_local():
    """The cloudy modules must not resolve to synthesizer's copies."""
    for module in (cloudy.cloudy17, cloudy.cloudy23, cloudy.cloudy25):
        assert module.__name__.startswith("syncretize.cloudy.")
    offenders = [
        str(p)
        for p in SRC.rglob("*.py")
        if "synthesizer.photoionisation import cloudy" in p.read_text()
        or "synthesizer.photoionisation.cloudy" in p.read_text()
    ]
    assert not offenders, offenders


def test_ionisation_edges_stay_sharp():
    """Inserted edge points copy their own side of the edge (#151)."""
    from unyt import Angstrom, Hz, erg, s

    from syncretize.grid_io import EDGE_OFFSET, add_ionisation_edges
    from syncretize.photoionisation import edge_is_resolved, edge_wavelength

    # A coarse grid like M13/FSPS 3.2: 905 | 915 around the HI edge
    lam = np.array([100.0, 905.0, 915.0, 1e4]) * Angstrom
    lnu = np.array([[1e-3, 1e-2, 1.0, 2.0]]) * erg / s / Hz
    assert not edge_is_resolved(lam.value)

    new_lam, new_spectra = add_ionisation_edges(lam, {"incident": lnu})
    edge = edge_wavelength("HI")
    assert edge_is_resolved(new_lam.value)
    assert np.all(np.diff(new_lam.value) > 0)
    assert new_spectra["incident"].units == lnu.units
    flux = dict(
        zip(np.round(new_lam.value, 4), new_spectra["incident"][0].value)
    )
    assert flux[round(edge - EDGE_OFFSET, 4)] == 1e-2
    assert flux[round(edge + EDGE_OFFSET, 4)] == 1.0
    # HeI and HeII edges are inside the grid too: 6 new points in total
    assert new_lam.size == lam.size + 6
    # The original points are untouched
    assert np.all(np.isin(lam.value, new_lam.value))
    # Inserted either side of the HeII, HeI and HI edges (hc/E, written out
    # here rather than taken from edge_wavelength)
    edges = np.array([227.9121, 504.0008, 911.6485])
    expected = np.sort(np.r_[edges - EDGE_OFFSET, edges + EDGE_OFFSET])
    inserted = np.setdiff1d(new_lam.value, lam.value)
    np.testing.assert_allclose(inserted, expected, atol=1e-3)

    # A sample already just blueward of the HI edge is kept, not duplicated
    close = np.array([100.0, 905.0, edge - EDGE_OFFSET / 2, 915.0, 1e4])
    close_lam, _ = add_ionisation_edges(
        close * Angstrom, {"incident": np.ones((1, 5)) * erg / s / Hz}
    )
    # Only the redward HI point and the four He points are added
    assert close_lam.size == close.size + 5
    assert np.min(np.abs(close_lam.value - (edge + EDGE_OFFSET))) < 1e-6
