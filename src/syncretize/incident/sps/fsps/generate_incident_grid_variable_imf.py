"""Create an FSPS grid with an optional YAML-configured IMF axis.

Example:
    python generate_incident_grid_variable_imf.py \
        --grid-dir path/to/grid/dir \
        --config-file \
        parameter_files/incident/sps/fsps/variable_high_mass_slope.yaml \
        [--cores 4]
"""

import fsps
from unyt import Hz, Msun, angstrom, dimensionless, erg, s, yr

from syncretize.incident.builder import GridSpec, Output, build_grid
from syncretize.incident.sps.fsps.generate_incident_grid import (
    stellar_population,
)
from syncretize.incident.sps.fsps_utils import (
    check_fsps_version,
    expand_fsps_slopes,
    fsps_libraries,
    imf_values_at_axis,
    load_fsps_imf_config,
)
from syncretize.incident.sps.utils import get_model_filename
from syncretize.parser import Parser


def _axis_description(config):
    """Describe the configured IMF axis and its power-law convention."""
    axis = config["axis"]
    if axis["group"] == "slopes":
        lower = config["boundaries"][axis["index"]]
        upper = config["boundaries"][axis["index"] + 1]
        return (
            "IMF power-law slope alpha in dN/dM proportional to M^-alpha "
            f"over {lower:g} <= M/Msun < {upper:g}"
        )
    return f"IMF mass boundary corresponding to '{axis['name']}', in Msun"


def _set_imf_parameters(sp, boundaries, slopes):
    """Apply one broken-power-law IMF definition to python-FSPS."""
    sp.params["imf_lower_limit"] = boundaries[0]
    sp.params["imf_upper_limit"] = boundaries[-1]
    for index, slope in enumerate(expand_fsps_slopes(slopes), start=1):
        sp.params[f"imf{index}"] = slope


def setup(config_file, input_dir=None):
    """Read the config and describe the grid (see syncretize.incident.builder).

    Args:
        config_file (str)
            The IMF parameter file.
        input_dir (str)
            Unused; FSPS has no source data to read.

    Returns:
        GridSpec
            The grid, with one point per (IMF axis value, metallicity), or
            per metallicity without an IMF axis.
    """
    config = load_fsps_imf_config(config_file)

    # The FSPS version is fixed when FSPS is built, so check the installation
    # matches the config before making anything
    check_fsps_version(config["sps_version"], fsps.__version__)

    fixed_boundaries = config["boundaries"]
    fixed_slopes = config["slopes"]
    fsps_slopes = expand_fsps_slopes(fixed_slopes)
    sp_args = {
        "imf_type": 2,
        "imf_lower_limit": float(fixed_boundaries[0]),
        "imf_upper_limit": float(fixed_boundaries[-1]),
        "imf1": float(fsps_slopes[0]),
        "imf2": float(fsps_slopes[1]),
        "imf3": float(fsps_slopes[2]),
    }
    sp = stellar_population(sp_args)

    libraries = fsps_libraries(sp, config["libraries"])
    variant = "".join(libraries[:2])
    axis = config["axis"]
    model = {
        "sps_name": "fsps",
        "sps_version": config["sps_version"],
        "sps_variant": variant,
        "imf_type": config["imf_type"],
        "imf_masses": fixed_boundaries,
        "imf_slopes": fixed_slopes,
        "alpha_enhancement": False,
        "pyfsps_version": str(fsps.__version__),
        "fsps_libraries": ",".join(libraries),
        "imf_config": str(config_file),
    }
    if axis is not None:
        model["variable_imf_parameter"] = axis["name"]
        model["variable_imf_axis"] = axis["axis_name"]

    axes = {
        "ages": 10**sp.log_age * yr,
        "metallicities": sp.zlegend * dimensionless,
    }
    descriptions = {}
    log_on_read = {"ages": True, "metallicities": False}
    point_axes = ("metallicities",)
    if axis is not None:
        axis_units = (
            Msun if axis["group"] == "mass_boundaries" else dimensionless
        )
        axes[axis["axis_name"]] = axis["axis_values"] * axis_units
        descriptions[axis["axis_name"]] = _axis_description(config)
        log_on_read[axis["axis_name"]] = axis["log_on_read"]
        # The IMF axis outermost, so FSPS rebuilds its SSPs once per value
        point_axes = (axis["axis_name"], "metallicities")

    return GridSpec(
        grid_name=get_model_filename(
            model, axis_name=None if axis is None else axis["axis_name"]
        ),
        model=model,
        axes=axes,
        wavelength=sp.wavelengths * angstrom,
        outputs={
            "incident": Output(units=erg / s / Hz),
            "star_fraction": Output(
                spectra=False,
                description="Remaining stellar mass fraction with "
                "dimensions matching the grid axes",
            ),
        },
        point_axes=point_axes,
        log_on_read=log_on_read,
        descriptions=descriptions,
        weight="initial_masses",
        data={"sp_args": sp_args, "config": config},
    )


def generate_point(spec, index):
    """Return every age's spectrum and stellar fraction at one point."""
    config = spec.data["config"]
    axis = config["axis"]
    sp = stellar_population(spec.data["sp_args"])
    value = None
    if axis is not None:
        value = spec.axes[axis["axis_name"]].value[index[axis["axis_name"]]]
    boundaries, slopes = imf_values_at_axis(config, value)
    _set_imf_parameters(sp, boundaries, slopes)
    spectra = sp.get_spectrum(zmet=index["metallicities"] + 1)[1]
    return {
        "incident": spectra * 3.826e33,
        "star_fraction": sp.stellar_mass / sp.formed_mass,
    }


if __name__ == "__main__":
    parser = Parser(description="YAML-configured FSPS IMF grid creation")
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to an FSPS IMF YAML configuration file",
    )
    args = parser.parse_args()

    if args.download:
        print("FSPS is a Python package; no separate download is required")

    build_grid(
        setup(args.config_file),
        generate_point,
        args.grid_dir,
        param_file=args.config_file,
        cores=args.cores,
    )
