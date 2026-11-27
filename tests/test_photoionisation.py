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


def test_ionising_sed_is_rebinned_for_cloudy():
    """Cloudy's samples of a spiky SED keep its ionising photons."""
    from scipy.integrate import trapezoid
    from unyt import Angstrom, Hz, erg, s

    from syncretize.grid_io import add_ionisation_edges
    from syncretize.photoionisation import (
        CLOUDY_MESH,
        edge_wavelength,
        rebin_ionising,
    )

    # A spiky ionising spectrum sampled every 9 Angstrom, like FSPS 3.2's,
    # with a step at each edge and the edge points grid_io adds
    lam = np.r_[np.arange(100.0, 912.0, 9.0), 2000.0, 5000.0]
    lnu = np.where(np.arange(lam.size) % 2, 3.0, 1.0)
    lnu = lnu * np.where(lam > edge_wavelength("HI"), 10.0, 1.0)
    lam, spectra = add_ionisation_edges(
        lam * Angstrom, {"incident": lnu * erg / s / Hz}
    )
    lam, lnu = lam.value, spectra["incident"].value

    new_lam, new_lnu = rebin_ionising(lam, lnu)
    assert np.all(np.diff(new_lam) > 0)
    # The edges stay sharp, and nothing redward of the HI edge changes
    edge = edge_wavelength("HI")
    near = np.abs(lam - edge) < 0.05
    assert np.all(np.isin(lam[near], new_lam))
    assert np.all(new_lam[new_lam > edge] == lam[lam > edge])

    # Ionising photons (a sum of lnu/lam) as cloudy sees them: the SED
    # interpolated log-log at its mesh, against the input's
    mesh = np.loadtxt(CLOUDY_MESH)
    mesh = mesh[(mesh > 100.0) & (mesh < edge)]

    def photons(x, y):
        seen = np.exp(np.interp(np.log(mesh), np.log(x), np.log(y)))
        return trapezoid(seen / mesh, mesh)

    ionising = lam < edge
    truth = trapezoid(lnu[ionising] / lam[ionising], lam[ionising])
    assert abs(photons(lam, lnu) / truth - 1) > 0.03
    assert abs(photons(new_lam, new_lnu) / truth - 1) < 0.01
