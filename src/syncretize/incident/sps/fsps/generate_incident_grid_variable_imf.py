"""Create an FSPS grid with an optional YAML-configured IMF axis.

Example:
    python generate_incident_grid_variable_imf.py \
        --grid-dir path/to/grid/dir \
        --config-file fsps/configs/variable_high_mass_slope.yaml
"""

import fsps
import numpy as np
from unyt import Hz, Msun, angstrom, dimensionless, erg, s, yr

from syncretize.grid_io import GridFile
from syncretize.incident.sps.fsps_utils import (
    expand_fsps_slopes,
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


def generate_grid(config, grid_dir, config_file):
    """Generate an FSPS grid from a validated IMF configuration."""
    fixed_boundaries = config["boundaries"]
    fixed_slopes = config["slopes"]
    fsps_slopes = expand_fsps_slopes(fixed_slopes)
    sp = fsps.StellarPopulation(
        imf_type=2,
        imf_lower_limit=fixed_boundaries[0],
        imf_upper_limit=fixed_boundaries[-1],
        imf1=fsps_slopes[0],
        imf2=fsps_slopes[1],
        imf3=fsps_slopes[2],
    )

    variant = "".join(library.decode("utf-8") for library in sp.libraries[:2])
    axis = config["axis"]
    model = {
        "sps_name": "fsps",
        "sps_version": config["sps_version"],
        "sps_variant": variant,
        "imf_type": config["imf_type"],
        "imf_masses": fixed_boundaries,
        "imf_slopes": fixed_slopes,
        "alpha": False,
        "pyfsps_version": str(fsps.__version__),
        "imf_config": str(config_file),
    }
    if axis is not None:
        model["variable_imf_parameter"] = axis["name"]
        model["variable_imf_axis"] = axis["axis_name"]

    model_name = get_model_filename(
        model, axis_name=None if axis is None else axis["axis_name"]
    )
    out_filename = f"{grid_dir}/{model_name}.hdf5"

    lam = sp.wavelengths
    log10ages = sp.log_age
    ages = 10**log10ages
    metallicities = sp.zlegend

    axis_values = np.array([0.0]) if axis is None else axis["axis_values"]
    grid_shape = (len(ages), len(metallicities))
    if axis is not None:
        grid_shape += (len(axis_values),)
    spec = np.zeros((*grid_shape, len(lam)))
    stellar_fraction = np.zeros(grid_shape)

    for axis_index, axis_value in enumerate(axis_values):
        if axis is not None:
            print(
                f"IMF axis {axis_index + 1}/{len(axis_values)}: "
                f"{axis['axis_name']}={axis_value:g}",
                flush=True,
            )
        boundaries, slopes = imf_values_at_axis(
            config, None if axis is None else axis_value
        )
        _set_imf_parameters(sp, boundaries, slopes)

        for metallicity_index in range(len(metallicities)):
            spectra = sp.get_spectrum(zmet=metallicity_index + 1)[1]
            remaining_fraction = sp.stellar_mass / sp.formed_mass
            output_index = (slice(None), metallicity_index)
            if axis is not None:
                output_index += (axis_index,)
            spec[output_index] = spectra * 3.826e33
            stellar_fraction[output_index] = remaining_fraction

    axes = {
        "ages": ages * yr,
        "metallicities": metallicities * dimensionless,
    }
    descriptions = {}
    log_on_read = {"ages": True, "metallicities": False}
    if axis is not None:
        axis_units = (
            Msun if axis["group"] == "mass_boundaries" else dimensionless
        )
        axes[axis["axis_name"]] = axis_values * axis_units
        descriptions[axis["axis_name"]] = _axis_description(config)
        log_on_read[axis["axis_name"]] = axis["log_on_read"]

    out_grid = GridFile(out_filename)
    print(f"Writing {out_filename}", flush=True)
    out_grid.write_grid_common(
        model=model,
        axes=axes,
        wavelength=lam * angstrom,
        spectra={"incident": spec * erg / s / Hz},
        descriptions=descriptions,
        log_on_read=log_on_read,
        weight="initial_masses",
    )
    out_grid.write_dataset(
        "star_fraction",
        stellar_fraction * dimensionless,
        "Remaining stellar mass fraction with dimensions matching "
        "the grid axes",
        log_on_read=False,
    )
    out_grid.add_specific_ionising_lum()


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

    generate_grid(
        load_fsps_imf_config(args.config_file),
        args.grid_dir,
        args.config_file,
    )
