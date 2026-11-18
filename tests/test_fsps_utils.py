import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from syncretize.incident.sps.fsps_utils import (
    expand_fsps_slopes,
    fsps_libraries,
    imf_values_at_axis,
    load_fsps_imf_config,
    parse_fsps_imf_config,
)
from syncretize.incident.sps.utils import get_model_filename

CONFIG = (
    Path(__file__).parents[1]
    / "src/syncretize/incident/sps/fsps/configs"
    / "variable_high_mass_slope.yaml"
)
SINGLE_SLOPE_CONFIG = CONFIG.with_name("variable_single_slope.yaml")


def _raw_config():
    """Return an independent copy of the documented example configuration."""
    with CONFIG.open() as stream:
        return yaml.safe_load(stream)


def _parse(raw, tmp_path):
    """Write a modified configuration and parse it like the script does."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_fsps_imf_config(path)


def test_explicit_imf_axis_config():
    config = load_fsps_imf_config(CONFIG)

    assert config["axis"]["axis_name"] == "imf_high_mass_slope"
    assert config["axis"]["axis_values"][0] == 1.3
    assert config["axis"]["axis_values"][-1] == 3.0
    assert 2.3 in config["axis"]["axis_values"]

    boundaries, slopes = imf_values_at_axis(config, 1.8)
    np.testing.assert_allclose(boundaries, [0.08, 0.5, 1.0, 120.0])
    np.testing.assert_allclose(slopes, [1.3, 2.3, 1.8])


def test_generated_log_imf_axis_config(tmp_path):
    raw = _raw_config()
    axis = raw["Axes"][0]
    axis.pop("values")
    axis.update(type="log-range", min=1.0, max=4.0, n=3)
    config = _parse(raw, tmp_path)

    np.testing.assert_allclose(config["axis"]["axis_values"], [1.0, 2.0, 4.0])


def test_outer_mass_boundary_can_be_axis(tmp_path):
    raw = _raw_config()
    raw["Axes"] = [
        {
            "name": "imf_upper_mass",
            "target": "imf_masses[3]",
            "type": "values",
            "values": [100.0, 120.0, 150.0],
        }
    ]
    config = _parse(raw, tmp_path)
    assert config["axis"]["name"] == "upper_mass"

    boundaries, slopes = imf_values_at_axis(config, 150.0)
    np.testing.assert_allclose(boundaries, [0.08, 0.5, 1.0, 150.0])
    np.testing.assert_allclose(slopes, [1.3, 2.3, 2.3])


def test_single_power_law_config():
    config = load_fsps_imf_config(SINGLE_SLOPE_CONFIG)

    boundaries, slopes = imf_values_at_axis(config, 2.3)
    np.testing.assert_allclose(boundaries, [0.08, 120.0])
    np.testing.assert_allclose(slopes, [2.3])
    np.testing.assert_allclose(expand_fsps_slopes(slopes), [2.3] * 3)
    assert config["axis"]["axis_name"] == "imf_slope"
    assert len(config["axis"]["axis_values"]) == 18


def test_rejects_multiple_imf_axes(tmp_path):
    raw = _raw_config()
    second_axis = copy.deepcopy(raw["Axes"][0])
    second_axis.update(name="imf_low_mass_slope", target="imf_slopes[0]")
    raw["Axes"].append(second_axis)

    with pytest.raises(ValueError, match="At most one IMF parameter"):
        _parse(raw, tmp_path)


def test_rejects_variable_internal_boundary(tmp_path):
    raw = _raw_config()
    raw["Axes"] = [
        {
            "name": "imf_low_mass_break",
            "target": "imf_masses[1]",
            "type": "values",
            "values": [0.4, 0.5, 0.6],
        }
    ]

    with pytest.raises(
        ValueError, match="cannot vary internal mass boundaries"
    ):
        _parse(raw, tmp_path)


def test_fixed_imf_config_has_no_axis():
    config = parse_fsps_imf_config(
        SimpleNamespace(
            model={"sps_version": "4.0", "imf_type": "bpl"},
            constants={"imf_masses": [0.08, 120.0], "imf_slopes": [2.3]},
            axes={},
            axis_info={},
        )
    )
    assert config["axis"] is None


def test_variable_axis_filename_follows_canonical_prefix():
    model = {
        "sps_name": "fsps",
        "sps_version": "3.2",
        "sps_variant": "mistmiles",
        "imf_type": "bpl",
        "imf_masses": [0.08, 0.5, 1.0, 120.0],
        "imf_slopes": [1.3, 2.3, 2.3],
        "alpha_enhancement": False,
    }

    assert get_model_filename(model, axis_name="imf_high_mass_slope") == (
        "fsps-3.2-mistmiles_bpl-0.08,0.5,1.0,120.0-1.3,2.3,2.3"
        "_axis-imf_high_mass_slope"
    )


def test_fsps_libraries_checks_expected():
    sp = SimpleNamespace(libraries=(b"mist", b"miles", b"DL07"))
    assert fsps_libraries(sp) == ["mist", "miles", "DL07"]
    assert fsps_libraries(sp, ["mist", "miles"]) == ["mist", "miles", "DL07"]
    with pytest.raises(ValueError, match="compiled with"):
        fsps_libraries(sp, ["mist", "c3k_lr"])
