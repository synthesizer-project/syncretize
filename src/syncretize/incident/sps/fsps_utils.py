"""Configuration helpers for YAML-driven FSPS IMF grids."""

import re

import numpy as np

from syncretize.parser import load_params

#: The name of each IMF parameter, by position, for each number of entries
PARAMETER_NAMES = {
    "mass_boundaries": {
        2: ["lower_mass", "upper_mass"],
        4: ["lower_mass", "low_mass_break", "high_mass_break", "upper_mass"],
    },
    "slopes": {
        1: ["slope"],
        3: ["low_mass_slope", "intermediate_mass_slope", "high_mass_slope"],
    },
}

#: The Constants list each axis target indexes into
TARGET_GROUPS = {"imf_masses": "mass_boundaries", "imf_slopes": "slopes"}

#: The python-fsps release that provides each FSPS version. Each release
#: bundles one FSPS commit: 0.4.7 bundles 82a8735 (May 2024, the build the
#: published fsps-3.2 grids were made with) and 0.5.0 bundles v4.0.
FSPS_VERSIONS = {"3.2": "0.4.7", "4.0": "0.5.0"}

#: Where to find how to build FSPS for a given version and set of libraries
FSPS_README = "src/syncretize/incident/sps/fsps/README.md"


def check_fsps_model(model):
    """Check a config's Model section gives the FSPS version and libraries.

    Args:
        model (dict)
            The Model section of the parameter file.
    """
    for key in ("sps_version", "libraries"):
        if key not in model:
            raise ValueError(f"Model.{key} is required. See {FSPS_README}.")
    libraries = model["libraries"]
    if (
        not isinstance(libraries, list)
        or not libraries
        or not all(isinstance(name, str) and name for name in libraries)
    ):
        raise ValueError(
            "Model.libraries must be a non-empty list of library names, "
            f"e.g. [mist, c3k_lr], got {libraries!r}. See {FSPS_README}."
        )


def check_fsps_version(sps_version, installed):
    """Check the installed python-fsps provides the FSPS version asked for.

    FSPS is compiled into python-fsps, so the version can't be chosen at run
    time; a different installation raises rather than silently making a
    grid with another FSPS.

    Args:
        sps_version (str)
            The FSPS version the parameter file asks for, e.g. "4.0".
        installed (str)
            The installed python-fsps version (``fsps.__version__``).
    """
    if sps_version not in FSPS_VERSIONS:
        raise ValueError(
            f"Model.sps_version must be one of {list(FSPS_VERSIONS)}, got "
            f"{sps_version!r}. See {FSPS_README}."
        )
    if installed != FSPS_VERSIONS[sps_version]:
        raise ValueError(
            f"FSPS {sps_version} grids need python-fsps "
            f"{FSPS_VERSIONS[sps_version]}, but {installed} is installed. "
            f"See {FSPS_README} for how to install it."
        )


def parse_fsps_imf_config(params):
    """Validate an FSPS IMF configuration and return normalized values.

    The IMF is given by ``Constants.imf_masses`` and ``Constants.imf_slopes``.
    At most one axis may vary one of them, named by its ``target``, e.g.
    ``imf_slopes[2]``; the value in Constants is then the nominal one used in
    the filename.

    The current generator uses FSPS ``imf_type=2``. It supports any one of
    the three slopes or either outer mass limit as an axis. FSPS fixes its two
    internal boundaries at 0.5 and 1.0 solar masses, so varying those entries
    is rejected until custom ``imf_type=5`` files can be supplied safely.

    Args:
        params (SimpleNamespace)
            The parameters, as returned by ``load_params``.
    """
    model = params.model
    if model.get("imf_type") != "bpl":
        raise ValueError("Only Model.imf_type='bpl' is currently supported")
    check_fsps_model(model)

    fixed = {}
    for key in TARGET_GROUPS:
        if key not in params.constants:
            raise ValueError(f"Constants.{key} is required")
        try:
            fixed[key] = np.array(params.constants[key], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Constants.{key} must be numeric") from exc
        if fixed[key].ndim != 1 or not np.all(np.isfinite(fixed[key])):
            raise ValueError(f"Constants.{key} must be a list of numbers")
    fixed_boundaries = fixed["imf_masses"]
    fixed_slopes = fixed["imf_slopes"]
    if (len(fixed_boundaries), len(fixed_slopes)) not in {(2, 1), (4, 3)}:
        raise ValueError(
            "FSPS imf_type=2 requires either 2 boundaries and 1 slope or "
            "4 boundaries and 3 slopes"
        )

    if len(params.axis_info) > 1:
        raise ValueError("At most one IMF parameter may be an axis")
    active = None
    for axis_name, entry in params.axis_info.items():
        match = re.fullmatch(
            r"(imf_masses|imf_slopes)\[(\d+)\]", str(entry.get("target"))
        )
        if match is None:
            raise ValueError(
                f"Axis {axis_name} needs a target such as imf_slopes[2]"
            )
        group = TARGET_GROUPS[match.group(1)]
        index = int(match.group(2))
        if index >= len(fixed[match.group(1)]):
            raise ValueError(f"Axis {axis_name} target is out of range")
        if axis_name in {"ages", "metallicities"}:
            raise ValueError(
                f"Axis {axis_name} conflicts with a standard axis"
            )
        values = params.axes[axis_name]
        if len(values) < 2:
            raise ValueError(f"Axis {axis_name} needs at least 2 values")
        fixed_value = fixed[match.group(1)][index]
        if not values[0] <= fixed_value <= values[-1]:
            raise ValueError(
                f"The {match.group(1)} value the axis {axis_name} varies "
                "must lie within the axis"
            )
        log_on_read = entry.get("log_on_read", False)
        if not isinstance(log_on_read, bool):
            raise ValueError(f"Axis {axis_name} log_on_read must be a bool")
        if log_on_read and values[0] <= 0:
            raise ValueError(
                f"Axis {axis_name} log_on_read requires positive values"
            )
        active = {
            "name": PARAMETER_NAMES[group][len(fixed[match.group(1)])][index],
            "fixed_value": fixed_value,
            "group": group,
            "index": index,
            "axis_name": axis_name,
            "axis_values": values,
            "log_on_read": log_on_read,
        }

    # A single power law spans both outer limits, so FSPS's internal breaks
    # are irrelevant. A three-part IMF must match the hard-coded breaks.
    if len(fixed_boundaries) == 4 and (
        not np.isclose(fixed_boundaries[1], 0.5)
        or not np.isclose(fixed_boundaries[2], 1.0)
    ):
        raise ValueError(
            "FSPS imf_type=2 requires internal boundaries 0.5 and 1.0"
        )
    if (
        len(fixed_boundaries) == 4
        and active is not None
        and active["group"] == "mass_boundaries"
        and active["index"] in (1, 2)
    ):
        raise ValueError(
            "FSPS imf_type=2 cannot vary internal mass boundaries; "
            "custom imf_type=5 support is required"
        )

    axis_values = [None] if active is None else active["axis_values"]
    for value in axis_values:
        candidate = fixed_boundaries.copy()
        if active is not None and active["group"] == "mass_boundaries":
            candidate[active["index"]] = value
        if candidate[0] <= 0 or np.any(np.diff(candidate) <= 0):
            raise ValueError("Mass boundaries must remain strictly increasing")

    return {
        "sps_version": str(model["sps_version"]),
        "libraries": model.get("libraries"),
        "imf_type": model["imf_type"],
        "boundaries": fixed_boundaries,
        "slopes": fixed_slopes,
        "axis": active,
    }


def fsps_libraries(sp, expected=None):
    """Return the libraries FSPS was compiled with, checking them.

    FSPS output depends on its compiled isochrone, spectral and dust
    libraries, which cannot be chosen at run time. A parameter file can
    list the ones it expects, in that order (trailing ones may be left
    out); a different installation then raises rather than silently
    making a different grid.

    Args:
        sp (fsps.StellarPopulation)
            The stellar population; its ``libraries`` are read.
        expected (list of str, optional)
            The libraries the parameter file expects.

    Returns:
        list of str
            The isochrone, spectral and dust library names.
    """
    libraries = [
        lib.decode("utf-8") if isinstance(lib, bytes) else str(lib)
        for lib in sp.libraries
    ]
    if expected and [str(e) for e in expected] != libraries[: len(expected)]:
        raise ValueError(
            f"FSPS was compiled with libraries {libraries}, but the "
            f"parameter file expects {list(expected)}. Rebuild FSPS with "
            f"the expected libraries; see {FSPS_README}."
        )
    return libraries


def load_fsps_imf_config(path):
    """Load and validate an FSPS IMF YAML configuration file."""
    return parse_fsps_imf_config(
        load_params(
            path,
            model_keys=["sps_version", "imf_type", "libraries"],
            constant_keys=list(TARGET_GROUPS),
            axis_names=None,
        )
    )


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
