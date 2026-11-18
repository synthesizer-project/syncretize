"""Tests for the Draine & Li dust opacity script (no downloads)."""

import math

import h5py
import numpy as np
import pytest

import syncretize.dust.create_dustextcurve_draine_li as dl

# np.trapezoid needs numpy >= 2 (not available on Python 3.8)
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _synthetic_table(material, data_dir):
    """Smooth stand-in for a Draine table, wavelengths decreasing."""
    if material.startswith("pah"):
        radii = np.geomspace(3.548e-4, 1e-2, 12)
    else:
        radii = np.geomspace(1e-3, 10.0, 20)
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


def _run(components):
    return dl.compute_opacities({"components": components})["components"]


def test_normalise_per_gram_gives_unit_mass():
    a = np.geomspace(1e-3, 10.0, 300)
    shape = dl.lognormal_shape(a, [0.01, 0.05], [0.5, 0.9])
    n = dl.normalise_per_gram(a, shape, rho=3.0)
    a_cm = a * dl.MICRON_TO_CM
    mass = trapezoid(4 / 3 * math.pi * 3.0 * a_cm**3 * n, a_cm, axis=-1)
    np.testing.assert_allclose(mass, 1.0, rtol=1e-12)


def test_lognormal_mass_fraction_matches_numerical_integral():
    a_n, sigma, lo, hi = 0.002, 0.8, 1e-3, 1e-2
    a = np.geomspace(1e-7, 1e3, 200001)
    m = a**3 * dl.lognormal_shape(a, a_n, sigma)
    inside = (a >= lo) & (a <= hi)
    expected = trapezoid(m * inside, a) / trapezoid(m, a)
    frac = dl.lognormal_mass_fraction(lo, hi, a_n, sigma)
    assert frac == pytest.approx(expected, rel=1e-4)


def test_mass_and_number_median_conventions_agree(synthetic_tables):
    sigma = 0.6
    comps = _run(
        [
            {
                "material": "silicate",
                "kind": "lognormal",
                "mass_median": 0.1,
                "sigma": sigma,
            },
            {
                "material": "silicate",
                "kind": "lognormal",
                "number_median": 0.1 * math.exp(-3 * sigma**2),
                "sigma": sigma,
                "names": ["by_number"],
            },
        ]
    )
    for ct in dl.CURVE_TYPES:
        np.testing.assert_allclose(
            comps["by_number"]["kappa"][ct],
            comps["silicate_a0.1um"]["kappa"][ct],
            rtol=1e-12,
        )


def test_powerlaw_edges_expand_to_bins():
    specs = dl.expand_components(
        {
            "components": [
                {
                    "material": "graphite",
                    "kind": "powerlaw",
                    "edges": [0.001, 0.01, 0.1],
                    "power": -3.5,
                }
            ]
        }
    )
    assert [s["name"] for s in specs] == [
        "graphite_a0.001-0.01um",
        "graphite_a0.01-0.1um",
    ]
    assert [(s["a_min_um"], s["a_max_um"]) for s in specs] == [
        (0.001, 0.01),
        (0.01, 0.1),
    ]


@pytest.mark.parametrize(
    "entry, message",
    [
        (
            {
                "material": "silicate",
                "kind": "lognormal",
                "mass_median": 0.1,
                "number_median": 0.1,
                "sigma": 0.5,
            },
            "exactly one of number_median",
        ),
        (
            {"material": "silicate", "kind": "lognormal", "centre": 0.1},
            "unknown keys",
        ),
        (
            {"material": "iron", "kind": "lognormal", "mass_median": 0.1},
            "material must be one of",
        ),
        (
            {
                "material": "silicate",
                "kind": "powerlaw",
                "edges": [0.1, 0.01],
                "power": -3.5,
            },
            "strictly increasing",
        ),
        (
            {
                "material": "silicate",
                "kind": "powerlaw",
                "edges": [[0.005, 0.25], [0.01, 1.0]],
                "power": -3.5,
            },
            "use ranges for a list of",
        ),
        (
            {
                "material": "silicate",
                "kind": "powerlaw",
                "ranges": [0.001, 0.01, 0.1],
                "power": -3.5,
            },
            "use edges for a",
        ),
    ],
)
def test_invalid_components_raise(entry, message):
    with pytest.raises(ValueError, match=message):
        dl.expand_components({"components": [entry]})


def test_duplicate_names_raise():
    entry = {
        "material": "silicate",
        "kind": "lognormal",
        "mass_median": 0.1,
        "sigma": 0.5,
    }
    with pytest.raises(ValueError, match="Duplicate component names"):
        dl.expand_components({"components": [entry, entry]})


def test_table_nodes_match_fixed_components(synthetic_tables):
    log10_a_n = [-2.5, -2.0, -1.5]
    sigmas = [0.3, 0.7]
    comps = _run(
        [
            {
                "material": "graphite",
                "kind": "lognormal_table",
                "log10_number_median": log10_a_n,
                "sigma": sigmas,
            }
        ]
        + [
            {
                "material": "graphite",
                "kind": "lognormal",
                "number_median": 10**la,
                "sigma": s,
                "names": [f"n{i}_{j}"],
            }
            for i, la in enumerate(log10_a_n)
            for j, s in enumerate(sigmas)
        ]
    )
    table = comps["graphite_lognormal_table"]
    assert table["kappa"]["extinction"].shape[:2] == (3, 2)
    for i in range(3):
        for j in range(2):
            for ct in dl.CURVE_TYPES:
                np.testing.assert_allclose(
                    table["kappa"][ct][i, j],
                    comps[f"n{i}_{j}"]["kappa"][ct],
                    rtol=1e-12,
                )


def test_write_grid_layout(synthetic_tables, tmp_path):
    config = {
        "components": [
            {
                "material": "silicate",
                "kind": "lognormal",
                "mass_median": [0.01, 0.1],
                "sigma": 0.75,
            },
            {
                "material": "pah_neutral",
                "kind": "lognormal_table",
                "log10_number_median": {"min": -3.0, "max": -2.0, "n": 3},
                "sigma": {"min": 0.2, "max": 0.8, "n": 4},
            },
        ]
    }
    results = dl.compute_opacities(config)
    path = tmp_path / "opacities.hdf5"
    dl.write_grid(str(path), config, results)

    with h5py.File(path, "r") as hf:
        n_wav = hf["wavelength"].shape[0]
        fixed = hf["components/silicate_a0.1um"]
        assert list(fixed.attrs["axes"]) == []
        assert fixed.attrs["kind"] == "lognormal"
        assert fixed.attrs["mass_median_um"] == pytest.approx(0.1)
        assert fixed["extinction"].shape == (n_wav,)
        assert fixed["extinction"].attrs["Units"] == "cm**2/g"

        table = hf["components/pahneutral_lognormal_table"]
        axes = [
            a.decode() if isinstance(a, bytes) else a
            for a in table.attrs["axes"]
        ]
        assert axes == ["log10_number_median_um", "sigma"]
        assert table["attenuation"].shape == (3, 4, n_wav)
        assert table["untruncated_mass_fraction"].shape == (3, 4)
        assert "config" in hf["Model"].attrs
