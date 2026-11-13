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

import yaml


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

    def __init__(
        self,
        description,
        cloudy_args=False,
        incident_grid_required=True,
    ):
        """
        Create the parser instance.

        Args:
            description (str):
                A description of the program.
            cloudy_args (bool):
                Add the cloudy arguments instead of the incident ones.
            incident_grid_required (bool):
                Is --incident-grid required (cloudy arguments only)?
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
            self._add_cloudy_args(incident_grid_required)

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

    def _add_cloudy_args(self, incident_grid_required=True):
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
            required=incident_grid_required,
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
