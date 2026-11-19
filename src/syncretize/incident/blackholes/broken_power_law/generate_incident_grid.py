"""
Create a synthesizer incident grid for a broken power-law SED.
"""

import numpy as np
from synthesizer.utils.integrate import trapezoid
from unyt import Angstrom, Hz, c, dimensionless, erg, s

from syncretize.incident.builder import GridSpec, Output, build_grid
from syncretize.parser import Parser, load_params


def broken_power_law(x, edges, indices, normalisations=False, normalise=True):
    """
    Function to produce a broken power-law spectrum.

    Arguments
        x (array, float)
            x-coordinate
        edges (list, float)
            List of floats describing the edges of the individual power-law
            components.
        indices (list, float)
            List of the power-law indices. Length should be one less than
            edges.
        normalisations (list, float)
            List of normalisations. Length should be one less than
            edges.
        normalise (bool)
            Flag to decide whether to normalise to unity.

    """

    y = np.zeros(len(x))

    # if normalisations not calculated, go ahead an
    if not normalisations:
        normalisations = [
            1.0,
        ]

        # calcualte remaining normalisations
        for i, (edge, ind1, ind2) in enumerate(
            zip(edges[1:], indices, indices[1:])
        ):
            norm = normalisations[i] * (edge**ind1) / (edge**ind2)
            normalisations.append(norm)

    # now construct spectra
    for e1, e2, ind, norm in zip(
        edges[0:], edges[1:], indices, normalisations
    ):
        # identify indices within the wavelength range

        s = (x >= e1) & (x < e2)

        y[s] = norm * x[s] ** ind

    # normalise
    if normalise:
        y /= trapezoid(y[::-1], x=x[::-1])

    return y


def setup(config_file, input_dir=None):
    """Read the config and describe the grid (see syncretize.incident.builder).

    Args:
        config_file (str)
            The parameter file.
        input_dir (str)
            Unused; the model needs no source data.

    Returns:
        GridSpec
            The grid, with one point per combination of the slopes.
    """
    # open the config file and extract parameters
    config = load_params(
        config_file,
        model_keys=["grid_name"],
        constant_keys=["edges"],
        axis_names=["alpha1", "alpha2", "alpha3"],
    )
    parameters = {**config.constants, **config.axes}

    model_name = config.model["grid_name"]
    indices = [parameters[f"alpha{i}"] for i in (1, 2, 3)]
    edges_lam = np.array(parameters["edges"]) * Angstrom

    # Model defintion dictionary
    model = {
        "name": model_name,
        "type": "agn",
        "family": "broken_power_law",
    }

    # Define axes names
    axes_names = [f"alpha{i + 1}" for i, _ in enumerate(indices)]

    # Define axes descriptions
    axes_descriptions = {}
    for i, axis_name in enumerate(axes_names):
        axes_descriptions[axis_name] = (
            rf"The power-law slope between {edges_lam[i]} "
            rf"\le \lambda/\AA < {edges_lam[i + 1]}"
        )

    # In this case the incident axes values are just the indices
    # but we also convert lists to arrays
    axes_values = [np.array(index) * dimensionless for index in indices]

    # axes dictionary, save as part of output
    axes = dict(zip(axes_names, axes_values))

    # convert the value of edges for use
    edges_nu = c / edges_lam
    edges = edges_nu.to("THz").value

    # define wavelength grid using the edges
    lam = np.arange(edges_lam.value[0], edges_lam.value[-1], 10.0) * Angstrom

    return GridSpec(
        grid_name=model_name,
        model=model,
        axes=axes,
        descriptions=axes_descriptions,
        wavelength=lam,
        outputs={"incident": Output(units=erg / s / Hz)},
        point_axes=tuple(axes_names),
        # Define which axes are logged
        log_on_read={"alpha1": False, "alpha2": False, "alpha3": False},
        weight="bolometric_luminosities",
        data={"edges": edges},
    )


def generate_point(spec, index):
    """Return the broken power-law spectrum for one set of slopes."""
    # define the indices of the current model
    indices_ = [spec.axes[name][i].value for name, i in index.items()]

    # frequency grid
    x = (c / spec.wavelength).to("THz").value

    # generate the broken power-law spectra
    return {
        "incident": broken_power_law(
            x, spec.data["edges"][::-1], indices_[::-1]
        )
    }


if __name__ == "__main__":
    """
    Create incident AGN spectra assuming a broken power-law model.

    $L_{\nu} = \nu^{\alpha_i}

    """

    # define indices. Where this is a single value implies this is fixed.
    # TODO: this could be set by a separate parameter file, but this model
    # is unlikely to be used much since it should be deprecated by UnifiedAGN.

    # Set up the command line arguments
    parser = Parser(description="Broken power-law AGN model creation.")

    # parameter file to use
    parser.add_argument("--config-file", type=str, required=True)

    # get the arguments
    args = parser.parse_args()

    build_grid(
        setup(args.config_file),
        generate_point,
        args.grid_dir,
        param_file=args.config_file,
        cores=args.cores,
    )
