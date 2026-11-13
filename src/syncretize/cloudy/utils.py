from pathlib import Path

import numpy as np
import yaml

# Ready-made Cloudy parameter files shipped with syncretize
PARAM_DIR = Path(__file__).parent / "params"


def resolve_param_file(param_file, param_dir=None):
    """
    Find a Cloudy parameter file.

    The ".yaml" extension is optional. The first of these that exists is
    used: the path as given, the file in param_dir (e.g. the copy saved
    beside a grid's Cloudy runs), then the ready-made parameter file of that
    name (cloudy/params).

    Arguments:
        param_file (str or Path)
            path or name of the parameter file
        param_dir (str, optional)
            directory to look in after the path as given

    Returns:
        path (Path)
            path to the parameter file. If none exists, the last location
            tried in param_dir (or cloudy/params), so the error names it.
    """
    name = str(param_file)
    if Path(name).is_file():
        return Path(name)
    if not name.endswith(".yaml"):
        name += ".yaml"
    candidates = [Path(name)]
    if param_dir is not None:
        candidates.append(Path(param_dir) / name)
    candidates.append(PARAM_DIR / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[1] if param_dir is not None else candidates[-1]


def get_cloudy_params(param_file="c23.01-test", param_dir=None):
    """
    Read parameters from a yaml parameter file

    Arguments:
        param_file (str or Path)
            path or name of the parameter file
            (see resolve_param_file)
        param_dir (str)
            directory containing the parameter file

    Returns:
        fixed_params (dict)
            dictionary of parameters that are fixed
        grid_params (dict)
            dictionary of parameters that vary on the grid
    """

    # open paramter file
    with open(resolve_param_file(param_file, param_dir), "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    grid_params = {}
    fixed_params = {}

    # loop over parameters
    for k, v in params.items():
        # if parameter is a list store it in the grid_parameters dictionary
        # and convert to a numpy array
        if isinstance(v, list):
            grid_params[k] = np.array(list(map(float, v)))

        # if a dictionary collect any parameters that are also lists
        elif isinstance(v, dict):
            for k_, v_ in v.items():
                if isinstance(v_, list):
                    grid_params[f"{k}.{k_}"] = np.array(list(map(float, v_)))
                else:
                    fixed_params[f"{k}.{k_}"] = v_

        # otherwise store it in fixed_params dictionary
        else:
            fixed_params[k] = v

    return fixed_params, grid_params


def get_grid_props_cloudy(axes, axes_values, verbose=True):
    """
    Get the cloudy related properties of a grid.

    This returns some basic properties needed for processing an incident
    grid through CLOUDY.

    Args:
        axes (list)
            List of axes to vary.
        axes_values (dict)
            Dictionary of axis values.
        verbose (bool)
            Are we talking or not?

    Returns:
        n_axes (int)
            Number of axes.
        shape (list)
            Shape of the grid.
        n_models (int)
            Number of models.
        mesh (numpy.ndarray)
            Mesh of the grid.
        model_list (numpy.ndarray)
            List of models.
        index_list (numpy.ndarray)
            List of indices of each model within the grid.
    """
    # The grid axes
    if verbose:
        print(f"axes: {axes}")

    # Number of axes
    n_axes = len(axes)
    if verbose:
        print(f"number of axes: {n_axes}")

    # The shape of the grid (useful for creating outputs)
    shape = list([len(axes_values[axis]) for axis in axes])
    if verbose:
        print(f"shape: {shape}")

    # Determine number of models
    n_models = np.prod(shape)
    if verbose:
        print(f"number of models to run: {n_models}")

    # Create the mesh of the grid
    mesh = np.array(
        np.meshgrid(*[np.array(axes_values[axis]) for axis in axes])
    )

    # Create the list of the models
    model_list = mesh.T.reshape(n_models, n_axes)
    if verbose:
        print("model list:")
        print(model_list)

    # Create a list of the indices
    index_mesh = np.array(np.meshgrid(*[range(n) for n in shape]))
    index_list = index_mesh.T.reshape(n_models, n_axes)
    if verbose:
        print("index list:")
        print(index_list)

    return n_axes, shape, n_models, mesh, model_list, index_list
