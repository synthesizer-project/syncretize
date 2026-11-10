"""Tests for creating Synthesizer grids from cloudy outputs."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from unyt import Angstrom, Hz, dimensionless, erg, s, unyt_array

from syncretize.grid_io import GridFile

# create_synthesizer_grid.py imports its sibling modules (utils,
# submission_scripts) so their directory must be importable.
CLOUDY_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "syncretize" / "cloudy"
)

# The incident grid's log10 Q for each ion, one value per grid point
INCIDENT_LOG10_Q = {
    "HI": np.array([40.0, 41.0]),
    "HeII": np.array([38.0, 39.0]),
}

# Synthesizer rescales luminosity weighted grids on load, but not others
# ("None" is the weight RELAGN and RELQSO incident grids use)
WEIGHTS = ("bolometric_luminosities", "initial_masses", "None")


@pytest.fixture
def create_empty_grid(monkeypatch):
    """Import create_empty_grid with its sibling modules importable.

    Args:
        monkeypatch (pytest.MonkeyPatch):
            Used to put the cloudy directory on sys.path for this test only.

    Returns:
        function:
            create_synthesizer_grid.create_empty_grid.
    """
    # Skip gracefully if Synthesizer is not installed in the environment.
    pytest.importorskip("synthesizer.grid")
    monkeypatch.syspath_prepend(str(CLOUDY_DIR))
    from syncretize.cloudy.create_synthesizer_grid import create_empty_grid

    return create_empty_grid


def _write_incident_grid(path, weight):
    """Write a minimal two point incident grid with known log10 Q.

    Args:
        path (pathlib.Path):
            The file path of the incident grid, including the file name.
        weight (str):
            The WeightVariable of the incident grid.
    """
    grid = GridFile(str(path))

    axes = {"alpha": unyt_array([-1.6, -1.2], dimensionless)}
    wavelength = unyt_array([500.0, 1000.0, 2000.0], Angstrom)
    spectra = {
        "incident": unyt_array(
            np.ones((len(axes["alpha"]), len(wavelength))),
            erg / s / Hz,
        )
    }

    grid.write_grid_common(
        axes=axes,
        wavelength=wavelength,
        spectra=spectra,
        log_on_read={"alpha": False},
        descriptions={"alpha": "Power-law slope"},
        model={"model": "unit-test"},
        weight=weight,
    )

    for ion, values in INCIDENT_LOG10_Q.items():
        grid.write_dataset(
            f"log10_specific_ionising_luminosity/{ion}",
            values * dimensionless,
            f"The specific ionising photon luminosity for {ion}",
            log_on_read=False,
        )


@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("extra_axis", [False, True])
def test_create_empty_grid_copies_stored_ionising_luminosity(
    tmp_path, create_empty_grid, weight, extra_axis
):
    """The new grid stores the incident file's log10 Q unchanged.

    Synthesizer rescales luminosity weighted grids to its internal
    luminosity unit when loading them, so log10 Q must not be taken from
    the loaded Grid.
    """
    _write_incident_grid(tmp_path / "incident.hdf5", weight)

    new_axes = ["alpha"]
    new_axes_values = {"alpha": np.array([-1.6, -1.2])}
    if extra_axis:
        new_axes.append("ionisation_parameter")
        new_axes_values["ionisation_parameter"] = np.array([0.01, 0.1, 1.0])

    create_empty_grid(
        str(tmp_path),
        "incident",
        "new.hdf5",
        new_axes,
        new_axes_values,
        {"cloudy_version": "c23.01"},
    )

    with h5py.File(tmp_path / "new.hdf5", "r") as hf:
        assert hf.attrs["WeightVariable"] == weight
        for ion, expected in INCIDENT_LOG10_Q.items():
            stored = hf["log10_specific_ionising_luminosity"][ion][...]
            if extra_axis:
                # Repeated along the new axis for every incident point
                n_extra = len(new_axes_values["ionisation_parameter"])
                expected = np.repeat(expected[:, np.newaxis], n_extra, axis=1)
            np.testing.assert_array_equal(stored, expected)
