"""A script to make a reduced grids from incident SPS grid files.

This tool is used to create small subsampled grids for testing purposes. It
can be used to reduce the number of models in a grid by specifying the
desired ages and metallicities for the reduced axes. The script will then
find the nearest models in the original grid to the specified values and
create a new grid with only those models.

Example Usage:

    ```bash
    python create_reduced_grid.py reduced_grid.yaml --grid-dir /path/to/grid
    ```

    where reduced_grid.yaml contains:

    ```yaml
    original_grid: original_grid_name
    ages: [1.0e6, 1.0e7, 1.0e8]       # yr, optional
    max_age: 1.0e8                    # yr, optional
    metallicities: [0.001, 0.01]      # optional
    ```
"""

import numpy as np
from synthesizer.grid import Grid
from unyt import Hz, erg, s, yr

from syncretize.grid_io import GridFile
from syncretize.parser import incident_parser, load_param_file


def reduce_grid(original_grid, param_file=None, **axes):
    """
    Reduce the size of a grid by sampling the original axes.

    This will use a NGP sampling method to match the input axes to those
    on the original grid.
    """
    # Get all the axes on the original gripre-commit run --all-filesd
    orig_axes_names = original_grid.axes

    # Loop over axes and get the indices for the reduction (any axis not in the
    # input axes will be kept as is)
    new_axes_mask = {
        axis: np.zeros(original_grid.shape[ind], dtype=bool)
        for ind, axis in enumerate(orig_axes_names)
    }

    for ind, axis in enumerate(orig_axes_names):
        if axis not in axes:
            new_axes_mask[axis] = np.ones(original_grid.shape[ind], dtype=bool)
        else:
            print(f"Reducing axis {axis}")
            values = axes[axis]
            for value in values:
                nearest_index = np.argmin(
                    np.abs(getattr(original_grid, axis) - value)
                )
                new_axes_mask[axis][nearest_index] = True

    # Get the new axes
    new_axes = {
        axis: getattr(original_grid, axis)[mask]
        for axis, mask in new_axes_mask.items()
    }

    # Get the spectra
    new_spectra = {
        spec_type: original_grid.spectra[spec_type] * erg / s / Hz
        for spec_type in original_grid.spectra
    }

    # Apply the masks along each axis of the spectra grid
    for i, axis in enumerate(orig_axes_names):
        for spec_type in original_grid.spectra:
            new_spectra[spec_type] = np.take(
                new_spectra[spec_type],
                np.nonzero(new_axes_mask[axis])[0],
                axis=i,
            )

    # Set up the new grid output. The selected axis values are stored as
    # the grid's axes, so the name only needs the reduced flag.
    new_name = original_grid.grid_name + "_reduced"

    out_path = f"{original_grid.grid_dir}/{new_name}.hdf5"
    print(f"Writing reduced grid to {out_path}")

    # Collect model metadata
    model_metadata = {k: v for k, v in original_grid._model_metadata.items()}

    # Which axes should be logged when read?
    # TODO: this is maintaining some backwards compatibility which won't be
    # needed in the future. The logged_axes attribute doesn't exist in new
    # grids.
    if hasattr(original_grid, "_logged_axes"):
        log_on_read = {
            axis: log
            for axis, log in zip(orig_axes_names, original_grid._logged_axes)
        }
    else:
        log_on_read = {
            axis: "log10" in log
            for axis, log in zip(orig_axes_names, original_grid._extract_axes)
        }

    # GridFile opens its file in write mode, so a grid_name matching the
    # source grid would truncate the source
    if param_file is not None and (
        load_param_file(param_file).get("grid_name") == original_grid.grid_name
    ):
        raise ValueError(
            f"grid_name in {param_file} is the name of the grid being "
            "reduced; choose a different name."
        )

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_path, param_file=param_file)

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model_metadata,
        axes=new_axes,
        wavelength=original_grid.lam,
        spectra=new_spectra,
        log_on_read=log_on_read,
        weight=original_grid._weight_var,
    )

    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    parser = incident_parser("Reduce a grid")
    args = parser.parse_args()
    param_file = args.param_file
    params = load_param_file(param_file)
    # PyYAML reads 1e8 (no decimal point) as a string, hence the floats
    max_age = params.get("max_age") and float(params["max_age"])
    new_ages = params.get("ages")
    new_metallicities = params.get("metallicities")

    # open the parent incident grid
    original_grid = Grid(
        params["original_grid"],
        grid_dir=args.grid_dir,
    )

    # Initialise the new axes
    ages = None
    metallicities = None

    if new_ages:
        ages = np.array(new_ages, dtype=float) * yr
        print(f"New ages: {ages}")

    elif max_age:
        ages = original_grid.ages[original_grid.ages <= max_age]

    if new_metallicities:
        metallicities = np.array(
            new_metallicities,
            dtype=float,
        )
        print(f"New metallicities: {metallicities}")

    # We can also impose a maximum age if new_age were set... bit pointless
    # but it's possible
    if max_age:
        ages = ages[ages <= max_age]

    if ages is None and metallicities is None:
        raise ValueError("No reduction for any axis specified")

    # Collect which axes we are modifying
    axes = {}
    if ages is not None:
        axes["ages"] = ages
    if metallicities is not None:
        axes["metallicities"] = metallicities

    # Just do it...
    reduce_grid(original_grid, param_file=param_file, **axes)
