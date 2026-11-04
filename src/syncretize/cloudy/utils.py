from pathlib import Path

import numpy as np
import yaml

from syncretize.cloudy import cloudy17, cloudy23, cloudy25

# The cloudy package directory, holding the bundled line lists, and its
# ready-made Cloudy parameter files
CLOUDY_DIR = Path(__file__).parent
PARAM_DIR = CLOUDY_DIR / "params"

# What Cloudy prints at the end of a successful run
CLOUDY_SUCCESS = "Cloudy exited OK"


def get_cloudy_module(cloudy_version):
    """
    Return the interface module for a Cloudy version.

    Arguments:
        cloudy_version (str)
            The Cloudy version, e.g. "c23.01".

    Returns:
        module
            cloudy17, cloudy23 or cloudy25.
    """
    if cloudy_version.startswith("c17"):
        return cloudy17
    if cloudy_version.startswith("c23"):
        return cloudy23
    if cloudy_version.startswith("c25"):
        return cloudy25
    raise ValueError(f"Unknown Cloudy version: {cloudy_version}")


def resolve_linelist(linelist):
    """
    Find a line list: an existing path, or one bundled with syncretize.

    Arguments:
        linelist (str or Path)
            Path or name of the line list.

    Returns:
        path (Path)
            Path to the line list.
    """
    if Path(linelist).exists():
        return Path(linelist)
    return CLOUDY_DIR / linelist


def cloudy_run_status(model):
    """
    Say whether a Cloudy run succeeded, from the end of its output.

    Arguments:
        model (str or Path)
            The model path without extension, e.g. <grid>/<i>/<j>.

    Returns:
        reason (str or None)
            None if Cloudy exited OK, otherwise why the run failed.
    """
    out_file = Path(f"{model}.out")
    if not out_file.exists():
        return "not run (no .out file)"
    with open(out_file, "rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(handle.tell() - 8192, 0))
        tail = handle.read().decode("utf-8", errors="ignore")
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if any(CLOUDY_SUCCESS in line for line in lines[-20:]):
        return None
    if not lines:
        return "empty .out file"
    for token, reason in (
        ("ABORT", "aborted"),
        ("did not converge", "did not converge"),
        ("DISASTER", "disaster"),
    ):
        if any(token in line for line in lines[-20:]):
            return f"{reason}: {lines[-1]}"
    return f"did not finish: {lines[-1]}"


def resolve_param_file(param_file, param_dir=None):
    """
    Find a Cloudy parameter file.

    The ".yaml" extension is optional. Without param_dir, an existing path
    is used as given; otherwise the name is looked up in the ready-made
    parameter files (cloudy/params).

    Arguments:
        param_file (str or Path)
            path or name of the parameter file
        param_dir (str)
            directory containing the parameter file

    Returns:
        path (Path)
            path to the parameter file
    """
    name = str(param_file)
    if not name.endswith(".yaml"):
        name += ".yaml"
    if param_dir is not None:
        return Path(param_dir) / name
    if Path(name).exists():
        return Path(name)
    return PARAM_DIR / name


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
