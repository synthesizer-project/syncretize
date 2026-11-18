"""Tests for the Draine & Li dust opacity script (no downloads)."""

import math

import h5py
import numpy as np
import pytest
from scipy.interpolate import interp1d

import syncretize.dust.create_dustextcurve_draine_li as dl

# np.trapezoid needs numpy >= 2 (not available on Python 3.8)
trapezoid = getattr(np, "trapezoid", None) or np.trapz

GRAIN_RADII = np.geomspace(1e-3, 10.0, 20)
PAH_RADII = np.geomspace(3.548e-4, 1e-2, 12)


def _synthetic_table(material, data_dir):
    """Smooth stand-in for a Draine table, wavelengths decreasing."""
    radii = PAH_RADII if material.startswith("pah") else GRAIN_RADII
    wav = np.geomspace(1e3, 1e-3, 40)
    x = 2 * np.pi * radii[np.newaxis, :] / wav[:, np.newaxis]
    Qext = 2 * x**2 / (1 + x**2)
    return {
        "radii": radii,
        "wav": wav,
        "Qext": Qext,
        "Qabs": 0.4 * Qext,
        "Qsca": 0.6 * Qext,
        "g": np.full_like(Qext, 0.3),
    }


@pytest.fixture
def synthetic_tables(monkeypatch):
    monkeypatch.setattr(dl, "load_optical_table", _synthetic_table)


def _lognormal(a, sigma, materials=("silicate", "graphite"), **extra):
    return {
        "shape": "lognormal",
        "materials": list(materials),
        "axes": {"a": a, "sigma": sigma},
        **extra,
    }


def _powerlaw(a_min, a_max, materials=("silicate", "graphite"), **extra):
    return {
        "shape": "powerlaw",
        "materials": list(materials),
        "power": -3.5,
        "axes": {"a_min": a_min, "a_max": a_max},
        **extra,
    }


def test_restricted_weights_integrate_between_nodes_only():
    x = np.geomspace(1e-3, 1.0, 50)
    lo, hi = x[10], x[30]
    w = dl.restricted_trapezoid_weights(x, lo, hi)
    # trapezoid rule is exact for a linear integrand
    y = 2.0 * x * dl.MICRON_TO_CM
    expected = (hi**2 - lo**2) * dl.MICRON_TO_CM**2
    assert np.sum(w * y) == pytest.approx(expected, rel=1e-12)
    assert np.all(w[:10] == 0) and np.all(w[31:] == 0)
    # empty and NaN cells get no weight
    assert not np.any(dl.restricted_trapezoid_weights(x, hi, lo))
    assert not np.any(dl.restricted_trapezoid_weights(x, np.nan, hi))


def test_lognormal_a_axis_is_mass_median():
    a_m, sigma = 0.05, 0.7
    a = np.geomspace(1e-6, 1e4, 400001)
    mass_per_ln_a = a**4 * dl.lognormal_shape(a, a_m, sigma)
    cumulative = np.cumsum(mass_per_ln_a)
    median = a[np.searchsorted(cumulative, 0.5 * cumulative[-1])]
    assert median == pytest.approx(a_m, rel=1e-3)


def test_lognormal_mass_fraction_matches_numerical_integral():
    a_m, sigma, lo, hi = 0.004, 0.8, 1e-3, 1e-2
    a = np.geomspace(1e-7, 1e3, 200001)
    m = a**3 * dl.lognormal_shape(a, a_m, sigma)
    inside = (a >= lo) & (a <= hi)
    expected = trapezoid(m * inside, a) / trapezoid(m, a)
    frac = dl.lognormal_mass_fraction(lo, hi, a_m, sigma)
    assert frac == pytest.approx(expected, rel=1e-4)


def test_axis_spacing():
    a = dl.axis_values({"min": 0.001, "max": 1.0, "n": 4}, "a")
    np.testing.assert_allclose(a, [0.001, 0.01, 0.1, 1.0])
    s = dl.axis_values({"min": 0.1, "max": 0.4, "n": 4}, "sigma")
    np.testing.assert_allclose(s, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(dl.axis_values(0.75, "sigma"), [0.75])


def test_unit_mass_per_gram(synthetic_tables):
    res = dl.compute_opacities(_lognormal([0.01, 0.1], [0.5, 0.9]))
    mat = res["materials"]["silicate"]
    a_cm = mat["a_grid"] * dl.MICRON_TO_CM
    m = 4 / 3 * math.pi * mat["rho"] * a_cm**3
    mass = trapezoid(m * mat["n_per_gram"], a_cm, axis=-1)
    np.testing.assert_allclose(mass, 1.0, rtol=1e-12)


def test_grid_nodes_match_single_node_runs(synthetic_tables):
    a, sigmas = [0.003, 0.03, 0.3], [0.3, 0.7]
    grid = dl.compute_opacities(_lognormal(a, sigmas))["materials"]
    assert grid["graphite"]["kappa"]["extinction"].shape[:2] == (3, 2)
    for i, a_i in enumerate(a):
        for j, s_j in enumerate(sigmas):
            single = dl.compute_opacities(_lognormal(a_i, s_j))["materials"]
            for ct in dl.CURVE_TYPES:
                np.testing.assert_allclose(
                    grid["graphite"]["kappa"][ct][i, j],
                    single["graphite"]["kappa"][ct][0, 0],
                    rtol=1e-12,
                )


def test_powerlaw_nan_only_for_unphysical_cells(synthetic_tables):
    # includes a bin narrower than the integration spacing, a range
    # clipped by the table (a_max = 30 > 10 um) and ranges that miss it
    a_min, a_max = [0.005, 0.01, 20.0], [0.0101, 1.0, 30.0]
    res = dl.compute_opacities(_powerlaw(a_min, a_max))
    lo_grid, hi_grid = np.meshgrid(a_min, a_max, indexing="ij")
    physical = (lo_grid < hi_grid) & (lo_grid < 10.0)
    for mat in res["materials"].values():
        for ct in dl.CURVE_TYPES:
            finite = np.all(np.isfinite(mat["kappa"][ct]), axis=-1)
            np.testing.assert_array_equal(finite, physical)
            assert np.all(mat["kappa"][ct][physical] > 0)
        int_hi = mat["extra"]["integrated_a_max"][0]
        np.testing.assert_allclose(int_hi[physical & (hi_grid > 10)], 10.0)


def test_powerlaw_matches_direct_integral(synthetic_tables):
    a_min, a_max = 0.013, 0.37
    res = dl.compute_opacities(_powerlaw(a_min, a_max))
    kappa = res["materials"]["silicate"]["kappa"]["extinction"][0, 0]

    table = _synthetic_table("silicate", ".")
    Q = dl.efficiencies_on_grid(table, res["wav"])["extinction"]
    a = np.geomspace(a_min, a_max, 20001)
    Q_a = interp1d(table["radii"], Q, axis=1)(a)
    a_cm = a * dl.MICRON_TO_CM
    n = a**-3.5
    mass = trapezoid(4 / 3 * math.pi * 3.5 * a_cm**3 * n, a_cm)
    expected = trapezoid(math.pi * a_cm**2 * Q_a * n, a_cm, axis=-1) / mass
    np.testing.assert_allclose(kappa, expected, rtol=2e-3)


@pytest.mark.parametrize(
    "params, message",
    [
        ({**_lognormal(0.1, 0.5), "components": []}, "Unknown top-level"),
        ({**_lognormal(0.1, 0.5), "shape": "gaussian"}, "shape must be"),
        (_lognormal(0.1, 0.5, ["silicate", "iron"]), "Unknown materials"),
        (
            _lognormal(0.1, 0.5, ["silicate", "silicate"]),
            "must not repeat",
        ),
        (
            _lognormal(0.1, 0.5, ["silicate", "pah_neutral"]),
            "one family",
        ),
        (
            {**_lognormal(0.1, 0.5), "axes": {"a": 0.1}},
            "must be exactly",
        ),
        (_lognormal(0.1, 0.5, power=-3.5), "only used by shape powerlaw"),
        (
            {k: v for k, v in _powerlaw(0.01, 0.1).items() if k != "power"},
            "power is required",
        ),
        (_lognormal([0.1, 0.01], 0.5), "strictly increasing"),
        (_lognormal(0.1, [0.0, 0.5]), "must be > 0"),
        (
            _lognormal({"min": 0.01, "max": 0.1}, 0.5),
            "exactly min, max and n",
        ),
    ],
)
def test_invalid_params_raise(params, message):
    with pytest.raises(ValueError, match=message):
        dl.parse_params(params)


def _decode(values):
    return [v.decode() if isinstance(v, bytes) else v for v in values]


def test_write_grid_layout(synthetic_tables, tmp_path):
    params = _lognormal(
        {"min": 0.001, "max": 0.01, "n": 3},
        {"min": 0.2, "max": 0.8, "n": 4},
        ["pah_ionised", "pah_neutral"],
    )
    results = dl.compute_opacities(params)
    path = tmp_path / "opacities.hdf5"
    dl.write_grid(str(path), params, results)

    with h5py.File(path, "r") as hf:
        assert _decode(hf.attrs["axes"]) == ["a", "sigma"]
        assert hf["axes/a"].attrs["log_on_read"]
        assert not hf["axes/sigma"].attrs["log_on_read"]
        np.testing.assert_allclose(hf["axes/sigma"][()], [0.2, 0.4, 0.6, 0.8])
        assert (
            "mu = ln(a / 1 micron) - 3 sigma^2"
            in (hf["axes/a"].attrs["Description"])
        )
        n_wav = hf["wavelength"].shape[0]
        for ct, info in dl.CURVE_TYPES.items():
            group = hf[f"{ct}_curves"]
            assert group.attrs["efficiency"] == info["efficiency"]
            link = group.get("wavelength", getlink=True)
            assert isinstance(link, h5py.SoftLink)
            assert link.path == "/wavelength"
            assert set(group) == {"wavelength", "pahionised", "pahneutral"}
            assert group["pahneutral"].shape == (3, 4, n_wav)
            assert group["pahneutral"].attrs["Units"] == "cm**2/g"
        frac = hf["untruncated_mass_fraction/pahionised"]
        assert frac.shape == (3, 4)
        assert hf["Model"].attrs["shape"] == "lognormal"
        assert "paramfile" in hf["Model"].attrs
        truncation = hf["Model"].attrs["truncation"]
        assert "untruncated_mass_fraction" in truncation
        assert "[0.0003548, 0.01] um" in truncation


@pytest.mark.parametrize(
    "params",
    [
        _lognormal([0.01, 0.1], [0.5, 0.75, 1.0]),
        _lognormal(0.1, 0.75),
        _powerlaw([0.005, 0.01], [0.25, 1.0, 10.0]),
    ],
)
def test_synthesizer_grid_reads_output(synthetic_tables, tmp_path, params):
    Grid = pytest.importorskip("synthesizer.grid").Grid
    results = dl.compute_opacities(params)
    dl.write_grid(str(tmp_path / "opacities.hdf5"), params, results)

    grid = Grid("opacities", str(tmp_path), ignore_lines=True)
    assert grid.axes == list(results["axes"])
    sizes = tuple(v.size for v in results["axes"].values())
    for material in ("silicate", "graphite"):
        assert grid.spectra[material].shape == sizes + (results["wav"].size,)
        np.testing.assert_allclose(
            grid.spectra[material],
            results["materials"][material]["kappa"]["extinction"],
        )
