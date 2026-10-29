"""Tests for the cloudy agn continuum incident grid."""

import subprocess
import sys

import h5py
import numpy as np
import pytest
import yaml
from unyt import c

from syncretize.incident.blackholes.cloudy_agn import (
    generate_incident_grid as agn,
)

SCRIPT = agn.__file__

# The default cloudy AGN command: AGN T = 1e5 K, a(ox) = -1.4,
# a(uv) = -0.5, a(x) = -1
DEFAULTS = {
    "blue_bump_temperature": 1.0e5,
    "alpha_ox": -1.4,
    "alpha_uv": -0.5,
    "alpha_x": -1.0,
}

# Incident nu F_nu from cloudy c23.01 for the default command, as
# (energy in Ryd, nu F_nu relative to its peak), sampled from 0.01 to 100 Ryd
CLOUDY_DEFAULT_NUFNU = np.array(
    [
        (1.000353e-02, 1.091346e-01),
        (1.336859e-02, 1.614073e-01),
        (1.780611e-02, 2.228584e-01),
        (2.371664e-02, 2.930944e-01),
        (3.169459e-02, 3.720280e-01),
        (4.221526e-02, 4.567308e-01),
        (5.641577e-02, 5.480769e-01),
        (7.514241e-02, 6.420455e-01),
        (1.000851e-01, 7.364510e-01),
        (1.337524e-01, 8.277972e-01),
        (1.781499e-01, 9.077797e-01),
        (2.372850e-01, 9.685315e-01),
        (3.170926e-01, 1.000000e00),
        (4.223287e-01, 9.899476e-01),
        (5.624897e-01, 9.287587e-01),
        (7.516720e-01, 8.111888e-01),
        (1.001572e00, 6.481643e-01),
        (1.337551e00, 4.624126e-01),
        (1.780305e00, 2.933129e-01),
        (2.377810e00, 1.692308e-01),
        (3.165320e00, 1.037150e-01),
        (4.225695e00, 8.107517e-02),
        (5.628023e00, 7.867133e-02),
        (7.520690e00, 8.129371e-02),
        (1.001645e01, 8.395979e-02),
        (1.334044e01, 8.610140e-02),
        (1.782680e01, 8.771853e-02),
        (2.374266e01, 8.898601e-02),
        (3.172725e01, 8.990385e-02),
        (4.225597e01, 9.060315e-02),
        (5.627919e01, 9.117133e-02),
        (7.520752e01, 9.156469e-02),
        (9.983370e01, 9.187063e-02),
    ]
)

# A second command with every parameter shifted from the defaults:
# AGN T = 10^5.5 K, a(ox) = -1.0, a(uv) = -1.0, a(x) = -0.8
NON_DEFAULTS = {
    "blue_bump_temperature": 10**5.5,
    "alpha_ox": -1.0,
    "alpha_uv": -1.0,
    "alpha_x": -0.8,
}

# Incident nu F_nu from cloudy c23.01 for the second command, sampled as above
CLOUDY_NON_DEFAULT_NUFNU = np.array(
    [
        (1.000353e-02, 4.216410e-01),
        (1.332410e-02, 5.401026e-01),
        (1.780611e-02, 6.508718e-01),
        (2.371664e-02, 7.464615e-01),
        (3.158912e-02, 8.259487e-01),
        (4.221526e-02, 8.897436e-01),
        (5.622814e-02, 9.372308e-01),
        (7.489230e-02, 9.706667e-01),
        (1.000851e-01, 9.912821e-01),
        (1.333072e-01, 9.996923e-01),
        (1.775570e-01, 9.970256e-01),
        (2.372850e-01, 9.846154e-01),
        (3.160369e-01, 9.647179e-01),
        (4.223287e-01, 9.386667e-01),
        (5.624897e-01, 9.075897e-01),
        (7.491692e-01, 8.695385e-01),
        (9.996862e-01, 8.221538e-01),
        (1.333111e00, 7.644103e-01),
        (1.780305e00, 6.972308e-01),
        (2.369913e00, 6.268718e-01),
        (3.165320e00, 5.611282e-01),
        (4.211634e00, 5.123077e-01),
        (5.628023e00, 4.876923e-01),
        (7.495698e00, 4.904615e-01),
        (1.001645e01, 5.153846e-01),
        (1.334044e01, 5.526154e-01),
        (1.776751e01, 5.947692e-01),
        (2.374266e01, 6.390769e-01),
        (3.162167e01, 6.840000e-01),
        (4.211556e01, 7.300513e-01),
        (5.627919e01, 7.782564e-01),
        (7.495698e01, 8.277949e-01),
        (9.983370e01, 8.795897e-01),
    ]
)


def _nufnu(x, **params):
    """Return nu F_nu of the agn continuum at energies x in Ryd.

    Args:
        x (array, float):
            Photon energies in Ryd.
        **params (dict):
            Parameters passed to agn_photon_flux.

    Returns:
        array, float:
            nu F_nu, with an arbitrary normalisation.
    """
    return x**2 * agn.agn_photon_flux(x, **{**DEFAULTS, **params})


@pytest.mark.parametrize(
    "params, cloudy_nufnu",
    [
        (DEFAULTS, CLOUDY_DEFAULT_NUFNU),
        (NON_DEFAULTS, CLOUDY_NON_DEFAULT_NUFNU),
    ],
    ids=["default", "non-default"],
)
def test_matches_cloudy_continuum(params, cloudy_nufnu):
    """The shape matches cloudy's own incident continuum."""
    x, cloudy = cloudy_nufnu.T
    ratio = _nufnu(x, **params) / cloudy
    np.testing.assert_allclose(ratio / np.median(ratio), 1.0, rtol=2e-3)


def test_normalised_to_unit_bolometric_luminosity():
    """Each spectrum integrates to 1 erg/s over cloudy's energy range."""
    lam = agn.wavelength_grid()
    nu_hz = (c / lam).to("Hz").value
    for temperature in (1.0e4, 1.0e5, 10**6.5):
        params = {**DEFAULTS, "blue_bump_temperature": temperature}
        lnu = agn.agn_lnu(lam, **params)
        assert -np.trapezoid(lnu, nu_hz) == pytest.approx(1.0)


def test_grid_axes_and_fixed_parameters(tmp_path):
    """Lists in the config become axes and single values are fixed."""
    config = {
        "model": "cloudy_agn-unit-test",
        "log10blue_bump_temperature": [4.0, 5.0],
        "alpha_ox": [-1.4, -1.0],
        "alpha_uv": -0.5,
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

    with h5py.File(tmp_path / "cloudy_agn-unit-test.hdf5", "r") as hf:
        assert list(hf.attrs["axes"]) == ["blue_bump_temperature", "alpha_ox"]
        assert hf.attrs["WeightVariable"] == "bolometric_luminosities"
        np.testing.assert_allclose(
            hf["axes/blue_bump_temperature"][...], [1.0e4, 1.0e5]
        )
        assert hf["Model"].attrs["alpha_uv"] == -0.5
        assert hf["Model"].attrs["alpha_x"] == -1.0
        assert hf["spectra/incident"].shape[:2] == (2, 2)

        # cloudy's default command gives log10 Q(HI) = 9.497 per erg/s
        log10_q = hf["log10_specific_ionising_luminosity/HI"][...]
        assert log10_q[1, 0] == pytest.approx(9.497, abs=2e-3)
