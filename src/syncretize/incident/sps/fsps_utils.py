"""Configuration helpers for YAML-driven FSPS IMF grids."""

from pathlib import Path

import numpy as np
import yaml


def _numeric_value(value, field):
    """Return a finite float or raise a configuration error."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _axis_values(axis, field):
    """Build and validate an axis from explicit values or a range."""
    if not isinstance(axis, dict):
        raise ValueError(f"{field}.axis must be a mapping")

    has_values = "values" in axis
    has_range = "range" in axis
    if has_values == has_range:
        raise ValueError(
            f"{field}.axis must contain exactly one of 'values' or 'range'"
        )

    if has_values:
        if not isinstance(axis["values"], list) or len(axis["values"]) < 2:
            raise ValueError(
                f"{field}.axis.values must contain at least 2 values"
            )
        values = np.array(
            [
                _numeric_value(value, f"{field}.axis.values")
                for value in axis["values"]
            ]
        )
    else:
        range_config = axis["range"]
        if not isinstance(range_config, dict):
            raise ValueError(f"{field}.axis.range must be a mapping")
        required = {"minimum", "maximum", "number", "spacing"}
        missing = required - set(range_config)
        if missing:
            raise ValueError(
                f"{field}.axis.range is missing: {', '.join(sorted(missing))}"
            )

        minimum = _numeric_value(
            range_config["minimum"], f"{field}.axis.range.minimum"
        )
        maximum = _numeric_value(
            range_config["maximum"], f"{field}.axis.range.maximum"
        )
        number = range_config["number"]
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number < 2
        ):
            raise ValueError(
                f"{field}.axis.range.number must be an integer >= 2"
            )
        if maximum <= minimum:
            raise ValueError(f"{field}.axis.range.maximum must exceed minimum")

        spacing = range_config["spacing"]
        if spacing == "linear":
            values = np.linspace(minimum, maximum, number)
        elif spacing == "log":
            if minimum <= 0:
                raise ValueError(
                    f"{field}.axis log spacing requires positive limits"
                )
            values = np.geomspace(minimum, maximum, number)
        else:
            raise ValueError(
                f"{field}.axis.range.spacing must be 'linear' or 'log'"
            )

    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{field}.axis values must be unique and increasing")
    return values


def _parse_parameters(parameters, group):
    """Validate one ordered group of IMF parameters."""
    if not isinstance(parameters, list) or not parameters:
        raise ValueError(f"imf.{group} must be a non-empty list")

    parsed = []
    for index, parameter in enumerate(parameters):
        field = f"imf.{group}[{index}]"
        if not isinstance(parameter, dict):
            raise ValueError(f"{field} must be a mapping")
        for required in ("name", "fixed_value", "vary_axis"):
            if required not in parameter:
                raise ValueError(f"{field} is missing '{required}'")
        if not isinstance(parameter["name"], str) or not parameter["name"]:
            raise ValueError(f"{field}.name must be a non-empty string")
        if not isinstance(parameter["vary_axis"], bool):
            raise ValueError(f"{field}.vary_axis must be true or false")

        item = {
            "name": parameter["name"],
            "fixed_value": _numeric_value(
                parameter["fixed_value"], f"{field}.fixed_value"
            ),
            "vary_axis": parameter["vary_axis"],
            "group": group,
            "index": index,
        }
        if item["vary_axis"]:
            axis = parameter.get("axis")
            if not isinstance(axis, dict):
                raise ValueError(
                    f"{field}.axis is required when vary_axis is true"
                )
            if not isinstance(axis.get("name"), str) or not axis["name"]:
                raise ValueError(
                    f"{field}.axis.name must be a non-empty string"
                )
            item["axis_name"] = axis["name"]
            item["axis_values"] = _axis_values(axis, field)
            item["log_on_read"] = axis.get("log_on_read", False)
            if not isinstance(item["log_on_read"], bool):
                raise ValueError(
                    f"{field}.axis.log_on_read must be true or false"
                )
            if item["axis_name"] in {"ages", "metallicities"}:
                raise ValueError(
                    f"{field}.axis.name conflicts with a standard axis"
                )
            if not (
                item["axis_values"][0]
                <= item["fixed_value"]
                <= item["axis_values"][-1]
            ):
                raise ValueError(
                    f"{field}.fixed_value must lie within the axis"
                )
            if item["log_on_read"] and item["axis_values"][0] <= 0:
                raise ValueError(
                    f"{field}.axis.log_on_read requires positive values"
                )
        elif "axis" in parameter:
            raise ValueError(
                f"{field}.axis is only valid when vary_axis is true"
            )
        parsed.append(item)
    return parsed


def parse_fsps_imf_config(config):
    """Validate an FSPS IMF configuration and return normalized values.

    The current generator uses FSPS ``imf_type=2``. It supports any one of
    the three slopes or either outer mass limit as an axis. FSPS fixes its two
    internal boundaries at 0.5 and 1.0 solar masses, so varying those entries
    is rejected until custom ``imf_type=5`` files can be supplied safely.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    model = config.get("model")
    imf = config.get("imf")
    if not isinstance(model, dict) or not isinstance(imf, dict):
        raise ValueError("Configuration requires 'model' and 'imf' mappings")
    if model.get("imf_type") != "bpl":
        raise ValueError("Only model.imf_type='bpl' is currently supported")
    if "sps_version" not in model:
        raise ValueError("model.sps_version is required")

    boundaries = _parse_parameters(
        imf.get("mass_boundaries"), "mass_boundaries"
    )
    slopes = _parse_parameters(imf.get("slopes"), "slopes")
    if len(boundaries) != len(slopes) + 1:
        raise ValueError(
            "IMF requires exactly one more mass boundary than slope"
        )
    if (len(boundaries), len(slopes)) not in {(2, 1), (4, 3)}:
        raise ValueError(
            "FSPS imf_type=2 requires either 2 boundaries and 1 slope or "
            "4 boundaries and 3 slopes"
        )

    names = [item["name"] for item in boundaries + slopes]
    if len(names) != len(set(names)):
        raise ValueError("IMF parameter names must be unique")

    active = [item for item in boundaries + slopes if item["vary_axis"]]
    if len(active) > 1:
        raise ValueError("At most one IMF parameter may have vary_axis=true")
    active = active[0] if active else None

    # A single power law spans both outer limits, so FSPS's internal breaks
    # are irrelevant. A three-part IMF must match the hard-coded breaks.
    if len(boundaries) == 4 and (
        not np.isclose(boundaries[1]["fixed_value"], 0.5)
        or not np.isclose(boundaries[2]["fixed_value"], 1.0)
    ):
        raise ValueError(
            "FSPS imf_type=2 requires internal boundaries 0.5 and 1.0"
        )
    if (
        len(boundaries) == 4
        and active is not None
        and active["group"] == "mass_boundaries"
        and active["index"] in (1, 2)
    ):
        raise ValueError(
            "FSPS imf_type=2 cannot vary internal mass boundaries; "
            "custom imf_type=5 support is required"
        )

    fixed_boundaries = np.array([item["fixed_value"] for item in boundaries])
    fixed_slopes = np.array([item["fixed_value"] for item in slopes])
    axis_values = [None] if active is None else active["axis_values"]
    for value in axis_values:
        candidate = fixed_boundaries.copy()
        if active is not None and active["group"] == "mass_boundaries":
            candidate[active["index"]] = value
        if candidate[0] <= 0 or np.any(np.diff(candidate) <= 0):
            raise ValueError("Mass boundaries must remain strictly increasing")

    return {
        "sps_version": str(model["sps_version"]),
        "imf_type": model["imf_type"],
        "boundaries": fixed_boundaries,
        "slopes": fixed_slopes,
        "axis": active,
    }


def load_fsps_imf_config(path):
    """Load and validate an FSPS IMF YAML configuration file."""
    path = Path(path)
    with path.open() as stream:
        config = yaml.safe_load(stream)
    return parse_fsps_imf_config(config)


def imf_values_at_axis(config, value=None):
    """Return mass boundaries and slopes at one configured axis value."""
    boundaries = config["boundaries"].copy()
    slopes = config["slopes"].copy()
    axis = config["axis"]
    if axis is None:
        if value is not None:
            raise ValueError(
                "Axis value supplied for a fixed IMF configuration"
            )
        return boundaries, slopes
    if value is None:
        raise ValueError(
            "Axis value is required for a variable IMF configuration"
        )

    target = boundaries if axis["group"] == "mass_boundaries" else slopes
    target[axis["index"]] = value
    return boundaries, slopes


def expand_fsps_slopes(slopes):
    """Expand a single power-law slope onto FSPS's three IMF intervals."""
    slopes = np.asarray(slopes)
    if len(slopes) == 1:
        return np.repeat(slopes, 3)
    if len(slopes) == 3:
        return slopes.copy()
    raise ValueError("FSPS requires either one or three IMF slopes")
