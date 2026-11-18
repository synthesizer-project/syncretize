"""A module for parsing command line arguments in a standardised way.

Incident grid scripts take a YAML parameter file describing the grid, and
command line arguments for where to read and write files:

    parser = incident_parser("A description", input_dir=True)
    args = parser.parse_args()
    params = load_param_file(args.param_file)
    grid = GridFile(f"{args.grid_dir}/name.hdf5", param_file=args.param_file)

Cloudy scripts use the Parser class:

    parser = Parser("A description of the program", cloudy_args=True)
    args = parser.parse_args()

"""

import argparse
import re
from types import SimpleNamespace

import numpy as np
import yaml
from scipy.stats import qmc


def incident_parser(description, input_dir=False, download=False):
    """
    Create the command line parser for an incident grid script.

    Args:
        description (str):
            A description of the program.
        input_dir (bool):
            Does the script read source data from --input-dir?
        download (bool):
            Can the script download its source data (--download)?

    Returns:
        argparse.ArgumentParser:
            The parser, so scripts can add their own arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "param_file",
        help="Path to the YAML parameter file describing the grid.",
    )
    parser.add_argument(
        "--grid-dir",
        required=True,
        help="The directory to write the grid to.",
    )
    if input_dir:
        parser.add_argument(
            "--input-dir",
            required=True,
            help="The directory containing (or to contain) the source data.",
        )
    if download:
        parser.add_argument(
            "--download",
            action="store_true",
            help="Download the source data into --input-dir first.",
        )
    return parser


def load_param_file(param_file):
    """
    Read a YAML parameter file.

    Args:
        param_file (str):
            The path to the parameter file.

    Returns:
        dict:
            The parameters.
    """
    with open(param_file) as file:
        return yaml.safe_load(file) or {}


#: The sections every incident parameter file has, in order
SECTIONS = ("Model", "Constants", "Axes", "Sampling")

#: How an axis entry can be sampled
AXIS_TYPES = ("values", "linear-range", "log-range", "sobol")


class _Loader(yaml.SafeLoader):
    """A YAML loader that reads 1e6 and 1.0e6 as floats, not strings."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^[-+]?(?:[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?"
        r"|\.[0-9_]+(?:[eE][-+]?[0-9]+)?"
        r"|[0-9][0-9_]*[eE][-+]?[0-9]+"
        r"|\.(?:inf|Inf|INF)|[-+]\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


def load_params(
    param_file,
    model_keys=(),
    constant_keys=(),
    axis_names=(),
    allow_sobol=False,
):
    """
    Read an incident parameter file in the Model/Constants/Axes/Sampling form.

    Every section must be present, but may be empty (a blank key). Keys a
    script doesn't list are rejected, so a typo can't silently fall back to a
    default.

    Args:
        param_file (str):
            The path to the parameter file.
        model_keys, constant_keys, axis_names (iterable of str):
            The keys the script accepts in Model and Constants, and the axes
            it can build (None accepts any axis name).
        allow_sobol (bool):
            Can the script build a grid from Sobol-sampled points?

    Returns:
        SimpleNamespace:
            ``model`` and ``constants`` (dicts), ``axes`` (name -> array of
            values; for Sobol axes these are the sampled points, one per
            model), ``axis_info`` (name -> the axis entry) and ``sampling``.
    """
    with open(param_file) as file:
        raw = yaml.load(file, Loader=_Loader) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{param_file} must be a mapping of sections")

    unknown = set(raw) - set(SECTIONS)
    missing = [section for section in SECTIONS if section not in raw]
    if unknown or missing:
        raise ValueError(
            f"{param_file} needs exactly the sections {list(SECTIONS)} "
            f"(missing {missing}, unknown {sorted(unknown)}); leave a section "
            "blank to use the defaults"
        )

    model = raw["Model"] or {}
    constants = raw["Constants"] or {}
    entries = raw["Axes"] or []
    sampling = raw["Sampling"] or {}
    _check_keys(param_file, "Model", model, model_keys)
    _check_keys(param_file, "Constants", constants, constant_keys)

    axis_info = {}
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str) or (
            axis_names is not None and name not in axis_names
        ):
            allowed = "a name" if axis_names is None else list(axis_names)
            raise ValueError(
                f"{param_file}: axis {name!r} must be one of {allowed}"
            )
        if name in axis_info or name in constants:
            raise ValueError(f"{param_file}: {name} is given more than once")
        if entry.get("type") not in AXIS_TYPES:
            raise ValueError(
                f"{param_file}: axis {name} type must be one of "
                f"{list(AXIS_TYPES)}, got {entry.get('type')!r}"
            )
        axis_info[name] = entry

    sobol = [name for name, e in axis_info.items() if e["type"] == "sobol"]
    if sobol and not allow_sobol:
        raise ValueError(f"{param_file}: this script can't use Sobol axes")
    # Sobol points can't yet be combined with regular axes (that would need
    # the product of the point set and the regular axes)
    if sobol and len(sobol) != len(axis_info):
        raise ValueError(
            f"{param_file}: either every axis is Sobol-sampled or none is"
        )

    axes = (
        _sobol_axes(param_file, axis_info, sampling)
        if sobol
        else {
            name: _axis_values(param_file, entry)
            for name, entry in axis_info.items()
        }
    )
    return SimpleNamespace(
        model=model,
        constants=constants,
        axes=axes,
        axis_info=axis_info,
        sampling=sampling,
    )


def _check_keys(param_file, section, values, allowed):
    """Reject keys in a section that the script doesn't accept."""
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(
            f"{param_file}: unknown {section} keys {sorted(unknown)}; "
            f"allowed keys are {list(allowed)}"
        )


def _axis_values(param_file, entry):
    """Return the values of a regular axis entry."""
    kind = entry["type"]
    if kind == "values":
        values = np.asarray(entry["values"], dtype=float)
    else:
        lo, hi, n = entry["min"], entry["max"], int(entry["n"])
        if kind == "log-range":
            if lo <= 0 or hi <= 0:
                raise ValueError(
                    f"{param_file}: log-range axis {entry['name']} needs "
                    "positive limits"
                )
            values = np.geomspace(lo, hi, n)
        else:
            values = np.linspace(lo, hi, n)
    if values.size == 0 or np.any(np.diff(values) <= 0):
        raise ValueError(
            f"{param_file}: axis {entry['name']} must be strictly increasing"
        )
    return values


def _sobol_axes(param_file, axis_info, sampling):
    """Sample every axis together from one Sobol sequence."""
    if "n_samples" not in sampling:
        raise ValueError(f"{param_file}: Sobol axes need Sampling.n_samples")
    n_samples = int(sampling["n_samples"])
    sampler = qmc.Sobol(
        d=len(axis_info),
        scramble=sampling.get("scramble", True),
        seed=sampling.get("seed"),
    )
    # Sobol points balance best in powers of two; keep the first n_samples
    unit = sampler.random(n=2 ** int(np.ceil(np.log2(n_samples))))[:n_samples]

    axes = {}
    for column, (name, entry) in zip(unit.T, axis_info.items()):
        lo, hi = entry["min"], entry["max"]
        if entry.get("scale", "linear") == "log":
            axes[name] = 10 ** (
                np.log10(lo) + column * (np.log10(hi) - np.log10(lo))
            )
        else:
            axes[name] = lo + column * (hi - lo)
    return axes


def get_choice(params, key, options):
    """
    Return a required parameter, checking it is one of the allowed options.

    Args:
        params (dict):
            The parameters.
        key (str):
            The parameter to return.
        options (iterable):
            The allowed values.

    Returns:
        The value of the parameter.
    """
    value = params.get(key)
    if value not in options:
        raise ValueError(
            f"{key} must be one of {list(options)}, got {value!r}"
        )
    return value


class Parser(argparse.ArgumentParser):
    """
    A class for parsing command line arguments in a standardised way.

    NOTE: This class is a wrapper around the argparse.ArgumentParser class
    adding some standardised arguments and quality of life methods.

    Attributes:
        parser (argparse.ArgumentParser):
            The parser instance.
    """

    def __init__(self, description, cloudy_args=False):
        """
        Create the parser instance.

        Args:
            description (str):
                A description of the program.
            cloudy_args (bool):
                Add the cloudy arguments instead of the incident ones.
        """
        # Create the parser
        super(Parser, self).__init__(description=description)

        # First add the common arguments

        # Add filepath argument for the grids.
        self.add_argument(
            "--grid-dir",
            type=str,
            help="The directory containing (or to contain) the grids.",
            required=True,
        )

        # Are we talking?
        self.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Are we talking?",
        )

        # Now add the cloudy specific arguments or incident grid arguments
        if not cloudy_args:
            self._add_incident_args()
        else:
            self._add_cloudy_args()

    def _add_incident_args(self):
        """Add arguments for incident grids."""
        # Add filepath argument for the input files used to create
        # SPS incident grids.
        self.add_argument(
            "--input-dir",
            type=str,
            help="The input file path",
            required=False,
        )

        # Add download flag
        self.add_argument(
            "--download",
            "-d",
            action="store_true",
            help="Should the data be downloaded?",
        )

    def _add_cloudy_args(self):
        """Add arguments for cloudy grids."""
        # Path to directory where cloudy runs are
        self.add_argument(
            "--cloudy-output-dir",
            type=str,
            required=True,
            help="The directory containing each "
            "of the individual cloudy run outputs.",
        )

        # The name of the incident grid
        self.add_argument(
            "--incident-grid",
            type=str,
            required=True,
            help="The name of the incident grid (the grid "
            "used to generate the cloudy runs).",
        )

        # The cloudy parameters, including any grid axes
        self.add_argument(
            "--cloudy-paramfile",
            type=str,
            required=False,
            default="c23.01-sps",
            help="The path to the cloudy parameter file.",
        )

        # A second cloudy parameter set which supersedes the above
        self.add_argument(
            "--cloudy-paramfile-extra",
            type=str,
            required=False,
            help="The path to the cloudy 'extra' parameter file.",
        )

        # # Should we include the spectra in the grid?
        # self.add_argument(
        #     "--include-spectra",
        #     type=bool,
        #     action="store_true",
        #     help="Should the spectra be included in the grid?",
        # )

        # # Boolean flag as to whether to attempt to replace missing files
        # self.add_argument(
        #     "--replace",
        #     action="store_true",
        #     help="Should missing files be replaced?",
        # )

        # # Define the line calculation method.
        # self.add_argument(
        #     "--line-calc-method",
        #     type=str,
        #     default="lines",
        #     required=False,
        #     help="The method used to calculate the line fluxes "
        #     "(either 'lines' or 'linelist')",
        # )

        # # Define the machine (for rerunning cloudy runs)
        # self.add_argument(
        #     "--machine",
        #     type=str,
        #     default=None,
        #     required=False,
        #     help="The machine used to run the cloudy runs "
        #     "(currently only supports apollo)",
        # )

        # # Should we normalise by the specific ionising luminosity?
        # self.add_argument(
        #     "--norm-by-Q",
        #     "-Q",
        #     default=True,
        #     required=False,
        #     help="Should the grid be normalised by the specific "
        #     "ionising luminosity? (default is True)",
        # )

    def __str__(self):
        """Summarise the command line arguments."""
        self.print_help()
        return ""
