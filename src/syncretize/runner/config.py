"""Load and check a run configuration (``run.yaml``).

A run config has these sections:

    Run       where things go and how the jobs are sized (required)
    Incident  the incident model and its parameter file, or an existing
              incident grid (required)
    Cloudy    a Cloudy config and executable to run on the incident grid
              (optional; leave it out to make only the incident grid)
    Cloudy2, Cloudy3, ...
              more Cloudy set-ups for the same incident grid, each making
              its own grid (Cloudy is the same as Cloudy1)

See ``parameter_files/runs/run_template.yaml`` for every key.
"""

import importlib
import os
import re
from types import SimpleNamespace

import yaml

from syncretize.cloudy.utils import resolve_param_file
from syncretize.parser import YAMLLoader

#: Marks a key that must be given
REQUIRED = object()

#: Every key of the Run section, with its default
RUN_DEFAULTS = {
    "name": REQUIRED,
    "work_dir": REQUIRED,
    "grid_dir": REQUIRED,
    "input_dir": None,
    "environment": "",
    "slurm": {"partition": None, "account": None},
    "incident": {"cores": 1, "walltime": "12:00:00", "memory": None},
    "cloudy": {
        "workers": 250,
        "cores_per_worker": 4,
        "walltime": "12:00:00",
        "walltime_growth": 1.5,
        "max_walltime": "3-00:00:00",
        "margin": "00:15:00",
        "claim": 1,
        "max_retries": 2,
    },
    "assemble": {"cores": 1, "walltime": "12:00:00", "memory": None},
}

#: Every key of the Incident section, with its default
INCIDENT_DEFAULTS = {"model": None, "config": None, "grid": None}

#: Every key of the Cloudy section, with its default
CLOUDY_DEFAULTS = {
    "config": REQUIRED,
    "extra_config": None,
    "executable": REQUIRED,
    "include_spectra": True,
    "include_intrinsic_lines": False,
    # Models that fail every attempt are run again either side of their
    # parameters and interpolated (see syncretize.runner.perturb); off
    # unless given
    "perturb": None,
}

#: Every key of Cloudy.perturb, with its default
PERTURB_DEFAULTS = {"step": 0.1, "log_step": 0.05, "max_steps": 3}

#: How long before a worker's time limit it stops the Cloudy runs left
STOP_BEFORE = "00:02:00"

#: Keys that must be whole numbers of at least one
POSITIVE_INTEGERS = {
    "incident": ["cores"],
    "cloudy": ["workers", "cores_per_worker", "claim"],
    "assemble": ["cores"],
}

#: Keys that are Slurm time limits
WALLTIMES = {
    "incident": ["walltime"],
    "cloudy": ["walltime", "max_walltime", "margin"],
    "assemble": ["walltime"],
}

#: Paths resolved against the run config's directory when relative
RUN_PATHS = ("work_dir", "grid_dir", "input_dir")
INCIDENT_PATHS = ("config", "grid")


#: A Cloudy section's key: Cloudy (the first) or Cloudy<n>
CLOUDY_KEY = re.compile(r"Cloudy(\d*)")


def load_run(run_file):
    """Read and check a run config.

    Args:
        run_file (str)
            The path to ``run.yaml``.

    Returns:
        SimpleNamespace
            ``path`` (the absolute path of ``run_file``), ``run`` and
            ``incident`` (dicts with every default filled in), and
            ``sections`` (each Cloudy section, likewise filled in, by its
            number: 1 for Cloudy or Cloudy1; empty without Cloudy).
    """
    run_file = os.path.abspath(run_file)
    with open(run_file) as file:
        raw = yaml.load(file, Loader=YAMLLoader) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{run_file} must be a mapping of sections")
    unknown = {
        key
        for key in set(raw) - {"Run", "Incident"}
        if CLOUDY_KEY.fullmatch(key) is None
        or CLOUDY_KEY.fullmatch(key).group(1).startswith("0")
    }
    if unknown or "Run" not in raw or "Incident" not in raw:
        raise ValueError(
            f"{run_file} needs Run and Incident sections and may have "
            f"Cloudy sections (Cloudy, Cloudy2, ...; unknown "
            f"{sorted(unknown)})"
        )

    run = _fill(run_file, "Run", raw["Run"], RUN_DEFAULTS)
    incident = _fill(run_file, "Incident", raw["Incident"], INCIDENT_DEFAULTS)
    sections = {}
    for key in sorted(set(raw) - {"Run", "Incident"}):
        if raw[key] is None:
            continue
        index = int(CLOUDY_KEY.fullmatch(key).group(1) or 1)
        if index in sections:
            raise ValueError(
                f"{run_file}: Cloudy and Cloudy1 are the same section; give "
                "one of them"
            )
        sections[index] = _fill(run_file, key, raw[key], CLOUDY_DEFAULTS)
    sections = dict(sorted(sections.items()))

    for group, keys in POSITIVE_INTEGERS.items():
        for key in keys:
            value = run[group][key]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or (value < 1)
            ):
                raise ValueError(
                    f"{run_file}: Run.{group}.{key} must be a whole number "
                    f"of at least 1, got {value!r}"
                )
    for group, keys in WALLTIMES.items():
        for key in keys:
            try:
                seconds(run[group][key])
            except ValueError as error:
                raise ValueError(f"{run_file}: Run.{group}.{key}: {error}")
    if seconds(run["cloudy"]["walltime"]) > seconds(
        run["cloudy"]["max_walltime"]
    ):
        raise ValueError(
            f"{run_file}: Run.cloudy.walltime is longer than max_walltime"
        )
    if seconds(run["cloudy"]["margin"]) >= seconds(run["cloudy"]["walltime"]):
        raise ValueError(
            f"{run_file}: Run.cloudy.margin leaves the workers no time"
        )
    if seconds(run["cloudy"]["margin"]) <= seconds(STOP_BEFORE):
        raise ValueError(
            f"{run_file}: Run.cloudy.margin must be longer than "
            f"{STOP_BEFORE}, when the workers stop the Cloudy runs left"
        )
    growth = run["cloudy"]["walltime_growth"]
    if not isinstance(growth, (int, float)) or growth < 1:
        raise ValueError(
            f"{run_file}: Run.cloudy.walltime_growth must be at least 1"
        )

    # Exactly one of model + config (make the incident grid) or grid (use
    # an existing one, which only makes sense with Cloudy)
    if incident["grid"] is not None:
        if incident["model"] is not None or incident["config"] is not None:
            raise ValueError(
                f"{run_file}: give Incident.grid or Incident.model and "
                "Incident.config, not both"
            )
        if not sections:
            raise ValueError(
                f"{run_file}: Incident.grid needs a Cloudy section (there is "
                "nothing else to do)"
            )
    elif incident["model"] is None or incident["config"] is None:
        raise ValueError(
            f"{run_file}: Incident needs model and config (or grid, to use "
            "an existing incident grid)"
        )
    else:
        incident_module(incident["model"])

    base = os.path.dirname(run_file)
    for key in RUN_PATHS:
        run[key] = _resolve(base, run[key])
    for key in INCIDENT_PATHS:
        incident[key] = _resolve(base, incident[key])
    names = {}
    for index, cloudy in sections.items():
        where = f"Cloudy{index}"
        cloudy["executable"] = _resolve(base, cloudy["executable"])
        # Cloudy configs are a path or the name of a shipped config
        for key in ("config", "extra_config"):
            if cloudy[key] is None:
                continue
            relative = _resolve(base, cloudy[key])
            if os.path.isfile(relative):
                cloudy[key] = relative
            elif not resolve_param_file(cloudy[key]).is_file():
                raise ValueError(
                    f"{run_file}: {where}.{key} {cloudy[key]!r} is neither a "
                    "file nor a shipped Cloudy config"
                )
        _check_cloudy_version(run_file, where, cloudy)
        if cloudy["perturb"] is not None:
            _check_perturb(run_file, where, cloudy)
        # Each section's grid (and directory) is named after its configs
        name = section_name(cloudy)
        if name in names:
            raise ValueError(
                f"{run_file}: Cloudy{names[name]} and {where} would both "
                f"make the grid {name} (each needs different configs)"
            )
        names[name] = index

    return SimpleNamespace(
        path=run_file, run=run, incident=incident, sections=sections
    )


def section_name(cloudy):
    """A Cloudy section's name, from its config files.

    It is the Cloudy grid's name after "_cloudy-" (e.g. "c23.01-sps", or
    "c23.01-sps-density-1000" with an extra config), so each section of a
    run makes a differently named grid.

    Args:
        cloudy (dict)
            The Cloudy section of the run config.

    Returns:
        str
            The name.
    """
    return "-".join(
        resolve_param_file(cloudy[key]).stem
        for key in ("config", "extra_config")
        if cloudy[key] is not None
    )


def _check_perturb(run_file, where, cloudy):
    """Fill in a section's perturb defaults (``true`` takes them all) and
    check it.

    Args:
        run_file (str)
            The run config, for errors.
        where (str)
            The section's key, e.g. "Cloudy2", for errors.
        cloudy (dict)
            The section, whose perturb is filled in.
    """
    given = cloudy["perturb"]
    if given is True:
        given = {}
    perturb = _fill(run_file, "perturb", given, PERTURB_DEFAULTS, f"{where}.")
    cloudy["perturb"] = perturb
    for key in ("step", "log_step"):
        value = perturb[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{run_file}: {where}.perturb.{key} must be a number"
            )
        if value <= 0:
            raise ValueError(
                f"{run_file}: {where}.perturb.{key} must be positive"
            )
    steps = perturb["max_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(
            f"{run_file}: {where}.perturb.max_steps must be a whole number "
            "of at least 1"
        )


def incident_module(model):
    """Import an incident model by its path under ``syncretize.incident``.

    Args:
        model (str)
            e.g. ``sps/bc03/generate_incident_grid``.

    Returns:
        module
            The model's module, which has ``setup`` and ``generate_point``.
    """
    name = "syncretize.incident." + model.removesuffix(".py").replace("/", ".")
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        # Only a missing model is the config's fault; a model's own missing
        # dependency (e.g. fsps) should say so
        if not name.startswith(error.name or ""):
            raise
        raise ValueError(f"Incident.model {model!r}: no module {name}")
    if not hasattr(module, "setup") or not hasattr(module, "generate_point"):
        raise ValueError(
            f"Incident.model {model!r} has no setup and generate_point"
        )
    return module


def seconds(walltime):
    """A Slurm time limit, "[D-]HH:MM:SS", in seconds.

    YAML reads an unquoted 12:00:00 as the number 43200 (base 60), so a
    whole number is taken as seconds already.
    """
    if isinstance(walltime, int) and not isinstance(walltime, bool):
        return walltime
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d\d):(\d\d)", str(walltime))
    if match is None:
        raise ValueError(f"{walltime!r} isn't a time like [D-]HH:MM:SS")
    days, hours, minutes, secs = (int(x or 0) for x in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + secs


def slurm_time(seconds):
    """Seconds as a Slurm time limit, "D-HH:MM:SS"."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days}-{hours:02}:{minutes:02}:{secs:02}"


def _check_cloudy_version(run_file, where, cloudy):
    """Stop when the Cloudy build is a different version from the config's.

    Only a build directory named like a version (e.g. c23.01 or c25) is
    checked; its major version must match the config's cloudy_version.

    Args:
        run_file (str)
            The run config, for errors.
        where (str)
            The section's key, e.g. "Cloudy2", for errors.
        cloudy (dict)
            The section.
    """
    build = re.match(r"c(\d+)", os.path.basename(cloudy["executable"]))
    if build is None:
        return
    version = None
    for key in ("config", "extra_config"):
        if cloudy[key] is not None:
            with open(resolve_param_file(cloudy[key])) as file:
                params = yaml.load(file, Loader=YAMLLoader) or {}
            version = params.get("cloudy_version", version)
    wanted = re.match(r"c(\d+)", str(version))
    if wanted is not None and wanted.group(1) != build.group(1):
        raise ValueError(
            f"{run_file}: {where}.executable {cloudy['executable']} is Cloudy "
            f"{build.group(0)} but the Cloudy config is for {version}"
        )


def _fill(run_file, section, values, defaults, prefix=""):
    """Check a section's keys and fill in its defaults."""
    values = values or {}
    where = f"{run_file}: {prefix}{section}"
    if not isinstance(values, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(
            f"{where} has unknown keys {sorted(unknown)} (allowed: "
            f"{list(defaults)})"
        )
    out = {}
    for key, default in defaults.items():
        if isinstance(default, dict):
            out[key] = _fill(
                run_file, key, values.get(key), default, f"{prefix}{section}."
            )
        elif values.get(key) is not None:
            out[key] = values[key]
        elif default is REQUIRED:
            raise ValueError(f"{where} needs {key}")
        else:
            out[key] = default
    return out


def _resolve(base, path):
    """Make a relative path relative to base."""
    if path is None:
        return None
    return os.path.normpath(os.path.join(base, os.path.expanduser(path)))
