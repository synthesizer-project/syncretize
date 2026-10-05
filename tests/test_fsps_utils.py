import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from syncretize.incident.sps.fsps_utils import (
    expand_fsps_slopes,
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


def test_explicit_imf_axis_config():
    config = load_fsps_imf_config(CONFIG)

    assert config["axis"]["axis_name"] == "imf_high_mass_slope"
    assert config["axis"]["axis_values"][0] == 1.3
    assert config["axis"]["axis_values"][-1] == 3.0
    assert 2.3 in config["axis"]["axis_values"]

    boundaries, slopes = imf_values_at_axis(config, 1.8)
    np.testing.assert_allclose(boundaries, [0.08, 0.5, 1.0, 120.0])
    np.testing.assert_allclose(slopes, [1.3, 2.3, 1.8])


def test_generated_log_imf_axis_config():
    raw = _raw_config()
    axis = raw["imf"]["slopes"][-1]["axis"]
    axis.pop("values")
    axis["range"] = {
        "minimum": 1.0,
        "maximum": 4.0,
        "number": 3,
        "spacing": "log",
    }
    config = parse_fsps_imf_config(raw)

    np.testing.assert_allclose(config["axis"]["axis_values"], [1.0, 2.0, 4.0])


def test_outer_mass_boundary_can_be_axis():
    raw = _raw_config()
    raw["imf"]["slopes"][-1]["vary_axis"] = False
    raw["imf"]["slopes"][-1].pop("axis")
    boundary = raw["imf"]["mass_boundaries"][-1]
    boundary["vary_axis"] = True
    boundary["axis"] = {
        "name": "imf_upper_mass",
        "values": [100.0, 120.0, 150.0],
    }
    config = parse_fsps_imf_config(raw)

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


def test_rejects_multiple_imf_axes():
    raw = _raw_config()
    second_axis = raw["imf"]["slopes"][0]
    second_axis["vary_axis"] = True
    second_axis["axis"] = copy.deepcopy(raw["imf"]["slopes"][-1]["axis"])
    second_axis["axis"]["name"] = "imf_low_mass_slope"

    with pytest.raises(ValueError, match="At most one IMF parameter"):
        parse_fsps_imf_config(raw)


def test_rejects_variable_internal_boundary():
    raw = _raw_config()
    raw["imf"]["slopes"][-1]["vary_axis"] = False
    raw["imf"]["slopes"][-1].pop("axis")
    boundary = raw["imf"]["mass_boundaries"][1]
    boundary["vary_axis"] = True
    boundary["axis"] = {
        "name": "imf_low_mass_break",
        "values": [0.4, 0.5, 0.6],
    }

    with pytest.raises(
        ValueError, match="cannot vary internal mass boundaries"
    ):
        parse_fsps_imf_config(raw)


def test_variable_axis_filename_follows_canonical_prefix():
    model = {
        "sps_name": "fsps",
        "sps_version": "3.2",
        "sps_variant": "mistmiles",
        "imf_type": "bpl",
        "imf_masses": [0.08, 0.5, 1.0, 120.0],
        "imf_slopes": [1.3, 2.3, 2.3],
        "alpha": False,
    }

    assert get_model_filename(model, axis_name="imf_high_mass_slope") == (
        "fsps-3.2-mistmiles_bpl-0.08,0.5,1.0,120.0-1.3,2.3,2.3"
        "_axis-imf_high_mass_slope"
    )
