"""Tests for the cloudy blackbody incident grid."""

import subprocess
import sys

import h5py
import numpy as np
import pytest
import yaml
from unyt import c

from syncretize.incident.blackholes.blackbody import (
    generate_incident_grid as blackbody,
)

SCRIPT = blackbody.__file__

# Incident nu F_nu from cloudy c23.01 for "blackbody 5e4 linear", as
# (energy in Ryd, nu F_nu relative to its peak), sampled from 1e-6 of the
# peak on the low energy side to 1e-6 of the peak on the high energy side
CLOUDY_NUFNU = np.array(
    [
        (5.363800e-03, 1.008016e-06),
        (6.910060e-03, 2.149457e-06),
        (8.872460e-03, 4.536685e-06),
        (1.143020e-02, 9.661685e-06),
        (1.467620e-02, 2.035326e-05),
        (1.890710e-02, 4.320652e-05),
        (2.427650e-02, 9.069293e-05),
        (3.127490e-02, 1.917120e-04),
        (4.015660e-02, 4.001359e-04),
        (5.173290e-02, 8.396739e-04),
        (6.664630e-02, 1.752717e-03),
        (8.557320e-02, 3.595109e-03),
        (1.102420e-01, 7.379076e-03),
        (1.415500e-01, 1.482337e-02),
        (1.823550e-01, 2.953804e-02),
        (2.341420e-01, 5.711957e-02),
        (3.016310e-01, 1.081522e-01),
        (3.872770e-01, 1.952446e-01),
        (4.989000e-01, 3.362772e-01),
        (6.405580e-01, 5.339674e-01),
        (8.251840e-01, 7.691576e-01),
        (1.063350e00, 9.592391e-01),
        (1.364500e00, 9.831522e-01),
        (1.756780e00, 7.752717e-01),
        (2.262070e00, 4.308424e-01),
        (2.903060e00, 1.543478e-01),
        (3.736770e00, 3.044837e-02),
        (4.812140e00, 2.805707e-03),
        (6.178390e00, 1.020109e-04),
        (7.959020e00, 1.015353e-06),
    ]
)


def test_matches_cloudy_continuum():
    """The shape matches cloudy's own incident continuum."""
    x, cloudy = CLOUDY_NUFNU.T
    nufnu = x**2 * blackbody.blackbody_photon_flux(x, 5.0e4)
    ratio = nufnu / cloudy
    np.testing.assert_allclose(ratio / np.median(ratio), 1.0, rtol=2e-3)


def test_normalised_to_unit_bolometric_luminosity():
    """Each spectrum integrates to 1 erg/s over cloudy's energy range."""
    lam = blackbody.wavelength_grid()
    nu_hz = (c / lam).to("Hz").value
    for temperature in (1.0e4, 1.0e5, 1.0e6):
        lnu = blackbody.blackbody_lnu(lam, temperature)
        assert -blackbody.trapezoid(lnu, nu_hz) == pytest.approx(1.0)


def test_grid_axis_and_ionising_luminosity(tmp_path):
    """The temperatures become the axis, with cloudy's Q(HI) per erg/s."""
    config = {
        "model": "blackbody-unit-test",
        "log10blackbody_temperature": [4.0, float(np.log10(5.0e4))],
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))

    subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--config-file",
            str(config_file),
            "--grid-dir",
            str(tmp_path),
        ],
        check=True,
    )

    with h5py.File(tmp_path / "blackbody-unit-test.hdf5", "r") as hf:
        assert list(hf.attrs["axes"]) == ["blackbody_temperature"]
        assert hf.attrs["WeightVariable"] == "bolometric_luminosities"
        np.testing.assert_allclose(
            hf["axes/blackbody_temperature"][...], [1.0e4, 5.0e4]
        )
        assert hf["spectra/incident"].shape[0] == 2

        # cloudy's blackbody command gives log10 Q(HI) = 6.645 and 10.247
        # per erg/s at these temperatures
        log10_q = hf["log10_specific_ionising_luminosity/HI"][...]
        np.testing.assert_allclose(log10_q, [6.645, 10.247], atol=2e-3)


def test_rejects_temperatures_not_increasing(tmp_path):
    """A repeated or decreasing temperature stops the script."""
    config = {
        "model": "blackbody-unit-test",
        "log10blackbody_temperature": [4.0, 5.0, 5.0],
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--config-file",
            str(config_file),
            "--grid-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "strictly increasing" in result.stderr
    assert not (tmp_path / "blackbody-unit-test.hdf5").exists()
