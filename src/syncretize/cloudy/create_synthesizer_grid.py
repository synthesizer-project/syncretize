"""A module for reading in cloudy outputs and creating a synthesizer grid.

This module reads in cloudy outputs and creates a synthesizer grid. The
synthesizer grid is a HDF5 file that contains the spectra and line luminosities
from cloudy outputs.

Example usage:

"""

import json
import os
import re

import h5py
import numpy as np
from synthesizer.emissions import Sed
from synthesizer.grid import Grid
from synthesizer.units import has_units
from synthesizer.utils.integrate import trapezoid
from unyt import (
    Angstrom,
    Hz,
    cm,
    dimensionless,
    erg,
    eV,
    km,
    s,
    unyt_array,
    yr,
)

from syncretize.cloudy import (
    cloudy17,
    cloudy23,
    cloudy25,
    submission_scripts,
)
from syncretize.cloudy.utils import (
    THRESHOLD_KEY,
    below_ionising_threshold,
    get_cloudy_params,
    get_grid_props_cloudy,
    resolve_param_file,
)
from syncretize.grid_io import GridFile, write_cloudy_build
from syncretize.parser import Parser
from syncretize.provenance import find_out_file

#: Non-ionising band (Angstrom) used to check the normalised incident spectra.
CHECK_BAND = (1500.0, 5000.0)

#: Largest acceptable fractional offset of the normalised incident continuum.
CHECK_TOLERANCE = 0.02

# Create dictionary of commonly used units for grid axes
axes_units = {
    "reference_ionisation_parameter": dimensionless,
    "fixed_ionisation_parameter": dimensionless,
    "depletion_scale": dimensionless,
    "z": dimensionless,
    "ionisation_parameter": dimensionless,
    "ages": yr,
    "metallicities": dimensionless,
    "gas_metallicity": dimensionless,
    "hydrogen_density": cm ** (-3),
    "stop_column_density": dimensionless,
    "alpha_enhancement": dimensionless,
    "abundance_scalings.nitrogen_to_oxygen": dimensionless,
    "abundance_scalings.carbon_to_oxygen": dimensionless,
    "turbulence": km / s,
}

log_on_read_dict = {
    "reference_ionisation_parameter": True,
    "fixed_ionisation_parameter": True,
    "depletion_scale": False,
    "z": False,
    "ionisation_parameter": True,
    "gas_metallicity": True,
    "hydrogen_density": True,
    "stop_column_density": False,
    "alpha_enhancement": False,
    "abundance_scalings.nitrogen_to_oxygen": False,
    "abundance_scalings.carbon_to_oxygen": False,
}

pluralisation_of_axes = {
    "z": "redshifts",
    "metallicities": "metallicities",
    "gas_metallicity": "gas_metallicities",
    "ionisation_parameter": "ionisation_parameters",
    "turbulence": "turbulences",
    "hydrogen_density": "hydrogen_densities",
    "depletion_scale": "depletion_scales",
    "stop_column_density": "column_densities",
    "alpha_enhancement": "alpha_enhancements",
    "reference_ionisation_parameter": "reference_ionisation_parameters",
    "abundance_scalings.nitrogen_to_oxygen": (
        "abundance_scalings.nitrogen_to_oxygen"
    ),
    "abundance_scalings.carbon_to_oxygen": (
        "abundance_scalings.carbon_to_oxygen"
    ),
}


def create_empty_grid(
    grid_dir,
    incident_grid_name,
    new_grid_name,
    new_axes,
    new_axes_values,
    photoionisation_parameters,
    incident_grid_dir=None,
    param_files=(),
):
    """
    Create an empty Synthesizer grid file based on an existing incident grid.

    The new grid holds a copy of the incident grid's provenance (see
    ``GridFile.copy_provenance``), so it can be remade from itself alone.

    Args:
        grid_dir (str):
            Directory where the grid file will be saved.
        incident_grid_name (str):
            Name of the incident grid file we ran through cloudy.
        new_grid_name (str):
            Name of the new grid file.
        new_axes (list):
            List of the new axes.
        new_axes_values (dict):
            Dictionary of the new axes values (including incident).
        photoionisation_parameters (dict):
            Dictionary of photoionisation modelling parameters.
        incident_grid_dir (str, optional):
            Directory holding the incident grid, if not grid_dir.
        param_files (iterable of str, optional):
            The cloudy parameter files, stored verbatim in CloudyParams.

    Returns:
        out_grid (GridFile):
            The newly created grid file.
        incident_grid (GridFile):
            The incident grid file.
        shape (tuple)
            The shape of the new grid.
    """
    # Open the parent incident grid
    incident_grid = Grid(
        incident_grid_name,
        grid_dir=incident_grid_dir or grid_dir,
        ignore_lines=True,
    )

    # Make the new grid
    out_grid = GridFile(f"{grid_dir}/{new_grid_name}")

    # Add attribute with the original incident grid axes
    out_grid.write_attribute(
        group="/",
        attr_key="incident_axes",
        data=incident_grid.axes,
    )

    # Copy over the model metadata, and everything recorded about how the
    # incident grid was made
    out_grid.write_model_metadata(incident_grid._model_metadata)
    out_grid.copy_provenance(incident_grid.grid_filename)

    # Get properties of the grid
    (
        n_axes,
        shape,
        n_models,
        mesh,
        model_list,
        index_list,
    ) = get_grid_props_cloudy(new_axes, new_axes_values, verbose=False)

    # We want to copy over log10_specific_ionising_luminosity from the
    # incident grid to allow us to normalise the cloudy outputs.
    # However, the axes of the incident grid may be different from the
    # cloudy grid due to additional parameters, in which case we need to
    # extend the axes of log10_specific_ionising_luminosity.

    # Read log10_specific_ionising_luminosity directly from the incident
    # grid file rather than from the loaded Grid. synthesizer rescales
    # luminosity weighted grids to its internal luminosity unit on load, but
    # the cloudy outputs must be normalised to the values stored in the file.
    with h5py.File(incident_grid.grid_filename, "r") as hf:
        log10_specific_ionising_lum = {
            ion: hf["log10_specific_ionising_luminosity"][ion][...]
            for ion in hf["log10_specific_ionising_luminosity"].keys()
        }

        # The fractions of the formed mass left in stars and remnants
        # (synthesizer reads star_fraction for surviving masses), as stored
        fractions = {
            key: (
                hf[key][...],
                hf[key].attrs.get("Units", "dimensionless"),
                hf[key].attrs.get("Description", key),
            )
            for key in ("star_fraction", "remnant_fraction")
            if key in hf
        }

    # Compute the amount we may have to expand the axes (if we don't need to
    # expand this is harmless)
    expansion = int(
        np.prod(shape) / np.prod(log10_specific_ionising_lum["HI"].shape)
    )

    # Loop over ions
    for ion in log10_specific_ionising_lum.keys():
        # If there are no additional axes simply copy over the incident
        # .log10_specific_ionising_lum .
        if len(new_axes) == len(incident_grid.axes):
            out_grid.write_dataset(
                key=f"log10_specific_ionising_luminosity/{ion}",
                data=log10_specific_ionising_lum[ion] * dimensionless,
                description="The specific ionising photon luminosity for"
                f" {ion}",
                log_on_read=False,
            )

        # Otherwise, we need to expand the axis to account for these additional
        # parameters.
        else:
            # Create new array with repeated elements
            log10_specific_ionising_luminosity = np.repeat(
                log10_specific_ionising_lum[ion] * dimensionless,
                expansion,
                axis=-1,
            )
            out_grid.write_dataset(
                f"log10_specific_ionising_luminosity/{ion}",
                np.reshape(log10_specific_ionising_luminosity, shape)
                * dimensionless,
                description="The specific ionising photon luminosity"
                f" for {ion}",
                log_on_read=False,
            )

    # Copy over the stellar and remnant fractions, which depend only on the
    # incident model, unchanged
    for key, (values, units, description) in fractions.items():
        out_grid.write_dataset(
            key=key,
            data=unyt_array(values, units),
            description=description,
            log_on_read=False,
        )

    # Copy over the weight attribute from the incident grid
    weight_var = getattr(incident_grid, "_weight_var", None)
    if weight_var is None:
        weight_var = "None"
    out_grid.write_attribute(
        group="/",
        attr_key="WeightVariable",
        data=weight_var,
    )

    # Now write out the grid axes, we first do the incident grid axes so we
    # can extract their metadata and then any extras
    for axis, extract_axis in zip(
        incident_grid.axes, incident_grid._extract_axes
    ):
        # Write out this axis (we can get everything we need from the incident)
        if extract_axis != axis:
            log_on_read = True
        else:
            log_on_read = False

        out_grid.write_dataset(
            key=f"axes/{axis}",
            data=getattr(incident_grid, axis),
            description=f"Grid axes {axis} (taken from incident grid)",
            log_on_read=log_on_read,
        )

    # Now write out the new grid axes if there are any
    # List of new grid axes including pluralised form
    new_axes_for_grid = []
    for axis in new_axes:
        # Skip the one we did above
        if axis in incident_grid.axes:
            new_axes_for_grid.append(axis)
            continue

        if axis in pluralisation_of_axes.keys():
            pluralised_axis = pluralisation_of_axes[axis]
            new_axes_for_grid.append(pluralised_axis)
        else:
            raise ValueError(f"No pluralisation set for {axis} axis")

        # If we have units in the grid_params dictionary already then use
        # these, otherwise use the axes_units dictionary, if we don't have
        # units for the axis then raise an error.

        # Determine log_on_read safety
        log_on_read = log_on_read_dict.get(axis, False)

        if has_units(new_axes_values[axis]):
            description = f"grid axes {axis} (introduced during cloudy run)"
            out_grid.write_dataset(
                key=f"axes/{pluralised_axis}",
                data=new_axes_values[axis],
                description=description,
                log_on_read=log_on_read,
            )

        # Ok, maybe its just a common axis that we have units for
        elif axis in axes_units:
            # Multiply values with the corresponding unit
            values_with_units = new_axes_values[axis] * axes_units[axis]

            out_grid.write_dataset(
                key=f"axes/{pluralised_axis}",
                data=values_with_units,
                description=f"grid axes {axis} (introduced during cloudy run)",
                log_on_read=log_on_read,
            )

        else:
            raise ValueError(
                f"Unit for axis '{axis}' needs to be added to "
                f"create_synthesizer_grid.py"
            )

    # Add attribute with the full grid axes
    out_grid.write_attribute(
        group="/", attr_key="axes", data=new_axes_for_grid
    )

    # Write out cloudy parameters
    out_grid.write_cloudy_metadata(photoionisation_parameters, param_files)

    return out_grid, incident_grid


def get_grid_properties_hf(hf, verbose=True):
    """
    Get the properties of a grid from an HDF5 file.

    Arguments:
        hf : h5py.File
            The HDF5 file containing the grid.
        verbose : boolean
            Print out the properties of the grid.

    Returns:
        axes : list
            List of axes in the correct order.
        n_axes : int
            Number of axes in the grid.
        shape : tuple
            Shape of the grid.
        n_models : int
            Number of models in the grid.
        mesh : tuple
            Mesh of the grid.
        model_list : list
            List of model parameters.
        index_list : list
            List of model indices.
    """
    # Unpack the axes
    axes = hf.attrs["axes"]
    axes_values = {axis: hf[f"axes/{axis}"][:] for axis in axes}

    # Get the properties of the grid including the dimensions etc.
    return axes, *get_grid_props_cloudy(axes, axes_values, verbose=verbose)


def required_extensions(include_spectra, include_intrinsic_lines, geometry):
    """
    The cloudy output files a model needs to count as successful.

    Args:
        include_spectra (bool)
            Whether the grid stores the spectra.
        include_intrinsic_lines (bool)
            Whether the grid stores the intrinsic line luminosities.
        geometry (str)
            The cloudy geometry (e.g. "spherical").

    Returns:
        list of str
            The file extensions.
    """
    extensions = ["emergent_elin"]
    if include_intrinsic_lines:
        extensions.append("intrinsic_elin")

    # Open (non-spherical) geometries need the reflected continuum (.refl) to
    # separate the reflected incident light from the nebular emission. Without
    # it nebular would include scattered incident light, so treat a missing
    # .refl as a failed model rather than storing an incorrect spectrum.
    # Cloudy cannot write .refl for 'sphere', so it is not required there.
    if include_spectra and geometry != "spherical":
        extensions.append("refl")
    return extensions


def check_if_failed(
    model_to_check,
    extensions_to_check=["emergent_elin"],
):
    """
    Function to check if a particular model has failed.

    A model has failed unless cloudy ended its .out file with "Cloudy exited
    OK" and wrote each output file with at least 100 bytes.

    Args:
        model_to_check (str)
            The model, including path, to check.
        extensions_to_check (list)
            List of file extensions to check to check for each model.

    Returns:
        str or None
            Why the model failed (e.g. cloudy's last line, or "no
            .emergent_elin"), or None if it didn't.
    """
    # Cloudy writes "Cloudy exited OK" only when it ends without an error
    try:
        with open(model_to_check + ".out", "rb") as file:
            file.seek(0, os.SEEK_END)
            file.seek(max(0, file.tell() - 1000))
            tail = file.read().decode(errors="replace")
    except FileNotFoundError:
        return "no .out"
    if "Cloudy exited OK" not in tail:
        lines = tail.strip().splitlines()
        return lines[-1].strip() if lines else "empty .out"

    for ext in extensions_to_check:
        path = model_to_check + "." + ext
        if not os.path.isfile(path):
            return f"no .{ext}"
        if os.path.getsize(path) < 100:
            return f".{ext} under 100 bytes"
    return None


def _below(below_threshold, incident_index_tuple):
    """Whether an incident point is below the ionising luminosity threshold
    (and so wasn't run through cloudy)."""
    return below_threshold is not None and bool(
        below_threshold[tuple(incident_index_tuple)]
    )


def _first_run(cloudy_dir, new_grid_name, incident_index_list, n_photo):
    """The first model with a continuum (.cont), for the wavelength grid."""
    for incident_index in range(len(incident_index_list)):
        for photoionisation_index in range(n_photo):
            path = (
                f"{cloudy_dir}/{new_grid_name}/"
                f"{incident_index}/{photoionisation_index}"
            )
            if os.path.exists(path + ".cont"):
                return path
    raise ValueError(f"No cloudy continuum in {cloudy_dir}/{new_grid_name}")


def _interpolations(cloudy_dir, new_grid_name, extensions_to_check):
    """How to fill failed models from runs perturbed around them.

    Reads the plan the Runner keeps (perturb.json, see
    syncretize.runner.perturb). For each parameter a failed model was
    perturbed along whose two sides both ran, the model is the linear
    interpolation between them (in log10 of the parameter where the plan
    says so).

    Returns:
        dict
            Each model ("<i>_<j>") with at least one such pair: a list of
            (lower run, upper run, weight of the upper run) paths, without
            extensions.
    """
    path = f"{cloudy_dir}/{new_grid_name}/perturb.json"
    if not os.path.exists(path):
        return {}
    with open(path) as file:
        plan = json.load(file)
    interpolations = {}
    for model, parameters in plan.items():
        incident_index = model.partition("_")[0]
        pairs = []
        for parameter in parameters.values():
            lower, upper = parameter.get("minus"), parameter.get("plus")
            if lower is None or upper is None:
                continue
            paths = [
                f"{cloudy_dir}/{new_grid_name}/{incident_index}/{run['name']}"
                for run in (lower, upper)
            ]
            if any(check_if_failed(p, extensions_to_check) for p in paths):
                continue  # one sided isn't enough
            weight = (parameter["x"] - lower["x"]) / (upper["x"] - lower["x"])
            pairs.append((*paths, weight))
        if pairs:
            interpolations[model] = pairs
    return interpolations


def _normalised_run(cloudy, path, geometry, lam, log10_q):
    """A run's spectra, and the factor normalising it to the grid point.

    Cloudy runs a model at a total ionising luminosity; the grid stores it
    per unit of the incident model (e.g. per solar mass), so each run is
    scaled by the grid point's specific ionising luminosity over the run's.

    Args:
        log10_q (float)
            The grid point's log10_specific_ionising_luminosity/HI, or None
            to leave the run unscaled.

    Returns:
        (dict, float)
            The run's spectra (unscaled) and the normalisation.
    """
    spec_dict = cloudy.read_continuum(
        path, return_dict=True, geometry=geometry
    )
    if log10_q is None:
        return spec_dict, 1.0
    sed = Sed(lam=lam * Angstrom, lnu=spec_dict["incident"] * erg / s / Hz)
    ionising_photon_production_rate = (
        sed.calculate_ionising_photon_production_rate(
            ionisation_energy=13.6 * eV, limit=100
        )
    )
    return spec_dict, 10 ** (
        log10_q - np.log10(ionising_photon_production_rate)
    )


def check_cloudy_runs(
    new_grid_name,
    cloudy_dir,
    incident_index_list,
    photoionisation_index_list,
    extensions_to_check=["emergent_elin"],
    machine=None,
    cloudy_executable_path=None,
    below_threshold=None,
):
    """
    Check that all the cloudy runs have run properly.

    Args:
        new_grid_name
            The name of the new grid (also the subdirectory wher the cloudy
            runs are stored)
        cloudy_dir (str)
            Parent directory for the cloudy runs.
        incident_index_list (list)
            List of incident grid points
        photoionisation_index_list (list)
            List of photoionisation grid points
        replace (boolean)
            If a run has failed simply replace the model with the previous one.
        extensions_to_check (list)
            List of files to check for each model.

    Returns:
        failed_list : list
            List of models that have failed.
    """

    # List of tuples describing failed models
    failed_list = []

    for incident_index, incident_index_tuple in enumerate(incident_index_list):
        for photoionisation_index, photoionisation_index_tuple in enumerate(
            photoionisation_index_list
        ):
            # Points below the ionising threshold weren't run
            if _below(below_threshold, incident_index_tuple):
                continue

            # The model to check, including the full path
            model_to_check = (
                f"{cloudy_dir}/{new_grid_name}/"
                f"{incident_index}/{photoionisation_index}"
            )

            # Check if it failed (and why)
            failed = check_if_failed(
                model_to_check,
                extensions_to_check=extensions_to_check,
            )

            # Record models that have failed
            if failed:
                # Append the incident and photoionisation index to an array
                failed_list.append([incident_index, photoionisation_index])

                # If at least one model fails print a header
                if len(failed_list) == 1:
                    print("Failed models:")
                    print("incident_index, photoionisation_index, why")

                # Print the indices and why the model failed
                print("    ", incident_index, photoionisation_index, failed)

    number_of_failed_models = len(failed_list)

    if number_of_failed_models > 0:
        print("number of failed cloudy runs:", number_of_failed_models)

        # Save list of failed runs, beside the cloudy outputs
        failed_filename = f"{cloudy_dir}/{new_grid_name}.failures"
        np.savetxt(failed_filename, np.array(failed_list).T, fmt="%d")

        # If a specific machine is specified run the function in
        # submission_scripts to generate a submission script.
        if machine:
            getattr(submission_scripts, machine)(
                new_grid_name,
                number_of_jobs=number_of_failed_models,
                cloudy_output_dir=cloudy_dir,
                cloudy_executable_path=cloudy_executable_path,
                memory="4G",
                from_list=failed_filename,
            )

    # Return number_of_failed_models
    return number_of_failed_models


# Spectra stored in the grid by default. All are read from Cloudy by
# read_continuum (in cloudy17/cloudy23) and include the light leaving
# both faces of the cloud, consistent with the emergent line luminosities:
#   incident           : the incident spectrum
#   transmitted        : the attenuated incident spectrum (outward)
#   nebular            : nebular emission from both faces, lines included
#                        (DiffOut + reflc - reflected_incident)
#   linecont           : line contribution from both faces (outlin + reflin)
#   reflected_incident : incident light scattered back towards the source
#                        (open geometries; zero for 'sphere')
# so that total = transmitted + nebular + reflected_incident and
# nebular_continuum = nebular - linecont. Every grid stores the same set, so
# grids are read the same way whatever the geometry. read_continuum also
# provides "reflected" (reflc) and "reflected_linecont" (reflin), from which
# the outward-only spectra can be recovered, but these are not stored by
# default (they would add two full spectral grids). To keep them, add them to
# spec_names when calling add_spectra.
SPEC_NAMES = (
    "incident",
    "transmitted",
    "nebular",
    "linecont",
    "reflected_incident",
)

# Spectra that add_lines needs to compute the continuum quantities at each
# line (incident, transmitted, nebular_continuum = nebular - linecont and
# total_continuum = nebular_continuum + transmitted + reflected_incident).
# add_spectra always stores these, whatever spec_names it is given.
REQUIRED_SPEC_NAMES = (
    "incident",
    "transmitted",
    "nebular",
    "linecont",
    "reflected_incident",
)


def _cloudy_module(cloudy_version):
    """Return the syncretize module reading this Cloudy version's outputs."""
    major = cloudy_version.split(".")[0]
    if major == "c17":
        return cloudy17
    if major == "c23":
        return cloudy23
    if major == "c25":
        return cloudy25
    raise ValueError(f"Unsupported Cloudy version: {cloudy_version}")


def add_spectra(
    new_grid,
    new_grid_name,
    cloudy_dir,
    incident_index_list,
    photoionisation_index_list,
    new_shape,
    spec_names=SPEC_NAMES,
    norm_by_q=True,
    incident_grid_file=None,
    extensions_to_check=["emergent_elin"],
    geometry=None,
    below_threshold=None,
    interpolations=None,
):
    """
    Open cloudy spectra and add them to the grid.

    Args:
        new_grid (GridFile)
            The grid file to add the spectra to.
        new_grid_name (str)
            The name of the new grid.
        cloudy_dir (str)
            Parent directory for the cloudy runs.
        incident_index_list (list)
            List of incident grid points
        photoionisation_index_list (list)
            List of photoionisation grid points
        spec_names (list)
            The names of the spectra to save (default SPEC_NAMES: incident,
            transmitted, nebular, linecont and reflected_incident).
            The spectra in REQUIRED_SPEC_NAMES (incident, transmitted,
            nebular, linecont and reflected_incident) are always included,
            even if not listed, since add_lines needs them for the line
            continuum quantities (and reflected_incident for total =
            transmitted + nebular + reflected_incident).
        shape (ndarray)
            The shape of the new grid.
        norm_by_q (bool)
            If True, the spectra are normalised by the specific ionising
            photon luminosity calculated from the incident spectrum.
        incident_grid_file (str)
            Path to the incident grid file. If given, the normalised incident
            spectra are checked against it over CHECK_BAND.
        extensions_to_check (list)
            List of file extensions checked to decide if a model failed.
        geometry (str)
            The Cloudy geometry of the runs.
        below_threshold (ndarray of bool)
            The incident points below the ionising luminosity threshold,
            which weren't run: their incident light is transmitted as it is,
            with no nebular emission (needs incident_grid_file).
        interpolations (dict)
            Failed models to fill from runs perturbed around them (see
            _interpolations); they are marked in the "interpolated" dataset
            rather than "failures".
    """

    # Always store the spectra add_lines relies on. reflected_incident is
    # also needed to reconstruct the total spectrum for open geometries (it
    # is zero for 'sphere').
    spec_names = tuple(spec_names)
    spec_names += tuple(
        name for name in REQUIRED_SPEC_NAMES if name not in spec_names
    )

    # Determine the cloudy version...
    cloudy_version = new_grid.read_attribute(
        "cloudy_version",
        group="CloudyParams",
    )

    # ... and use to select the correct module.
    cloudy = _cloudy_module(cloudy_version)

    # Read first spectra from the first grid point to get length and
    # wavelength grid.
    lam = cloudy.read_wavelength(
        _first_run(
            cloudy_dir,
            new_grid_name,
            incident_index_list,
            len(photoionisation_index_list),
        )
    )

    # Write the spectra names... for some reason (not sure why since .keys()
    # gives the same result)
    new_grid.write_attribute(group="/", attr_key="spec_names", data=spec_names)

    # Number of wavelength points
    nlam = len(lam)

    # Set up the spectra dictionary
    spectra = {}

    # Make spectral grids and set them to zero
    for spec_name in spec_names:
        spectra[spec_name] = np.zeros((*new_shape, nlam))

    # Array for holding the normalisation which is calculated below and
    # used by lines
    spectra["normalisation"] = np.ones(new_shape)

    # Array for recording cloudy failures, and the failures interpolated
    # from runs perturbed around them
    failures = np.zeros(new_shape)
    interpolated = np.zeros(new_shape)
    interpolations = interpolations or {}

    # Ratio of the normalised incident continuum to the input SED, per run.
    # The input is read from the file, not a loaded Grid, since synthesizer
    # rescales luminosity weighted grids on load.
    band = (lam > CHECK_BAND[0]) & (lam < CHECK_BAND[1])
    continuum_ratios = []
    input_spectra = None
    quiet = below_threshold is not None and below_threshold.any()
    if incident_grid_file is not None and (band.any() or quiet):
        with h5py.File(incident_grid_file, "r") as incident_file:
            # Use the stored units (older grids without them are in
            # Angstrom and erg/s/Hz)
            dset = incident_file["spectra/wavelength"]
            input_lam = (
                unyt_array(dset[...], dset.attrs.get("Units", "Angstrom"))
                .to("Angstrom")
                .value
            )
            dset = incident_file["spectra/incident"]
            input_spectra = (
                unyt_array(dset[...], dset.attrs.get("Units", "erg/s/Hz"))
                .to("erg/s/Hz")
                .value
            )

    for incident_index, incident_index_tuple in enumerate(incident_index_list):
        input_lnu = None
        if input_spectra is not None:
            input_lnu = input_spectra[tuple(incident_index_tuple)]

        for photoionisation_index, photoionisation_index_tuple in enumerate(
            photoionisation_index_list
        ):
            # Get the indices of the grid_point
            indices = tuple(incident_index_tuple) + tuple(
                photoionisation_index_tuple
            )

            # The infile
            infile = (
                f"{cloudy_dir}/{new_grid_name}/"
                f"{incident_index}/{photoionisation_index}"
            )

            # Below the ionising threshold there is nothing to reprocess:
            # the incident light is transmitted, with no nebular emission
            if _below(below_threshold, incident_index_tuple):
                incident = np.interp(lam, input_lam, input_lnu)
                spectra["incident"][indices] = incident
                spectra["transmitted"][indices] = incident
                continue

            # Check to see if the model failed or not
            failed = check_if_failed(
                infile, extensions_to_check=extensions_to_check
            )

            # The grid point's specific ionising luminosity, which the runs
            # are normalised to
            log10_q = (
                new_grid.read_dataset(
                    "log10_specific_ionising_luminosity/HI", indices=indices
                )
                if norm_by_q
                else None
            )

            model = f"{incident_index}_{photoionisation_index}"
            if failed and model in interpolations:
                # Interpolate between the runs perturbed either side of it
                # (averaging over the parameters perturbed along)
                for spec_name in spec_names:
                    spectra[spec_name][indices] = np.zeros(nlam)
                for lower, upper, weight in interpolations[model]:
                    for path, share in ((lower, 1 - weight), (upper, weight)):
                        spec_dict, normalisation = _normalised_run(
                            cloudy, path, geometry, lam, log10_q
                        )
                        for spec_name in spec_names:
                            spectra[spec_name][indices] += (
                                share
                                * spec_dict[spec_name]
                                * normalisation
                                / len(interpolations[model])
                            )
                interpolated[indices] = 1

            # If the model has failed save the spectra as an array of zeros
            elif failed:
                # Set all the spectra to zero but with the correct shape
                for spec_name in spec_names:
                    spectra[spec_name][indices] = np.zeros(nlam)

                # Record failures in array
                failures[indices] = 1

            else:
                # Read the continuum file containing the spectra and
                # renormalise it by the specific ionising photon luminosity
                spec_dict, normalisation = _normalised_run(
                    cloudy, infile, geometry, lam, log10_q
                )

                # Save normalisation for later use (rescaling lines)
                spectra["normalisation"][indices] = normalisation

                # Save the normalised spectrum to the correct grid point
                for spec_name in spec_names:
                    spectra[spec_name][indices] = (
                        spec_dict[spec_name]
                        * spectra["normalisation"][indices]
                    )

                # Compare the normalised incident continuum with the input
                if input_lnu is not None:
                    input_band = np.interp(lam[band], input_lam, input_lnu)
                    output_band = spectra["incident"][indices][band]
                    input_flux = trapezoid(input_band, lam[band])
                    # An input with no flux in the band can't be compared
                    if input_flux > 0:
                        continuum_ratios.append(
                            trapezoid(output_band, lam[band]) / input_flux
                        )

    # Warn if the normalisation has moved the stellar continuum (#151)
    continuum_ratios = np.array(continuum_ratios)
    continuum_ratios = continuum_ratios[np.isfinite(continuum_ratios)]
    if input_spectra is not None and not continuum_ratios.size:
        print(
            "Could not compare the normalised incident continuum with "
            f"the input SED over {CHECK_BAND} A: no successful run has "
            "input flux there."
        )
    elif continuum_ratios.size:
        offsets = np.abs(continuum_ratios - 1)
        print(
            f"Normalised incident vs input over {CHECK_BAND} A: max offset "
            f"{offsets.max():.3f}, median {np.median(offsets):.3f}."
        )
        if offsets.max() > CHECK_TOLERANCE:
            print(
                f"WARNING: {np.sum(offsets > CHECK_TOLERANCE)} runs have a "
                "normalised incident continuum more than "
                f"{CHECK_TOLERANCE:.0%} from the input SED. The incident grid "
                "probably samples the ionisation edges too coarsely; "
                "regenerate it (see syncretize#151)."
            )

    # Apply units to the spectra
    for spec in spectra:
        spectra[spec] *= erg / s / Hz

    # Write the spectra out
    new_grid.write_spectra(
        spectra,
        wavelength=lam * Angstrom,
    )

    # Need to write the dataset containing the failures
    new_grid.write_dataset(
        "failures",
        failures * dimensionless,
        "array of model failures",
        False,
        verbose=False,
    )
    new_grid.write_dataset(
        "interpolated",
        interpolated * dimensionless,
        "1 where the model failed and was interpolated between runs "
        "perturbed either side of it",
        False,
        verbose=False,
    )

    return lam, spectra


def add_lines(
    new_grid,
    new_grid_name,
    cloudy_dir,
    incident_index_list,
    photoionisation_index_list,
    new_shape,
    lam=None,
    spectra=None,
    line_type="linelist",
    include_intrinsic=False,
    extensions_to_check=["emergent_elin"],
    below_threshold=None,
    geometry=None,
    interpolations=None,
):
    """
    Open cloudy lines and add them to the HDF5 grid

    Arguments:
        new_grid (GridFile)
            The grid file to add the spectra to.
        new_grid_name (str)
            The name of the new grid.
        cloudy_dir (str)
            Parent directory for the cloudy runs.
        incident_index_list (list)
            List of incident grid points
        photoionisation_index_list (list)
            List of photoionisation grid points
        new_shape (ndarray)
            The shape of the new grid.
        lam (ndarray)
            The wavelength grid of the spectra.
        spectra (dict)
            The spectra returned by add_spectra (or None to skip the
            continuum quantities). Must contain REQUIRED_SPEC_NAMES.
        line_type (str)
            The type of line file to use (linelist, lines)
        include_intrinsic (bool)
            If True, also read the intrinsic line luminosities (the
            'intrinsic_elin' Cloudy output) and store them as
            lines/luminosity_intrinsic.
        extensions_to_check (list)
            List of file extensions checked to decide if a model failed.
        below_threshold (ndarray of bool)
            The incident points below the ionising luminosity threshold.
        geometry (str)
            The Cloudy geometry of the runs.
        interpolations (dict)
            Failed models to fill from runs perturbed around them (see
            _interpolations).
    """
    interpolations = interpolations or {}

    def read_lines(path, extension):
        """A run's line IDs, wavelengths and luminosities, by wavelength."""
        ids, wavelengths, luminosities = cloudy.read_linelist(
            path, extension=extension
        )
        order = np.argsort(wavelengths)
        return ids[order], wavelengths[order], luminosities[order]

    if spectra is not None:
        calculate_continuum = True

        # Check we have every spectrum needed for the continuum quantities
        # (add_spectra always provides these).
        missing_spectra = [
            name for name in REQUIRED_SPEC_NAMES if name not in spectra
        ]
        if missing_spectra:
            raise ValueError(
                "add_lines needs the spectra "
                f"{', '.join(REQUIRED_SPEC_NAMES)} to compute the line "
                f"continuum quantities; missing: {', '.join(missing_spectra)}"
            )
    else:
        calculate_continuum = False

    # Determine the cloudy version...
    cloudy_version = new_grid.read_attribute(
        "cloudy_version",
        group="CloudyParams",
    )

    # ... and use to select the correct module.
    cloudy = _cloudy_module(cloudy_version)

    # Open the first linelist and use it to populate the keys
    with open(f"{cloudy_dir}/{new_grid_name}/0/linelist.dat") as file:
        linelist = file.readlines()
        # strip escape character
        lines_to_include = [
            re.sub(r"[\x00-\x1F\x7F]", "", s) for s in linelist
        ]

    # Calculate number of lines
    nlines = len(lines_to_include)

    print(f"number of lines included: {nlines}")

    # Set up the lines dictionary
    lines = {}

    # Define output quantities

    # We always save luminosity...
    lines["luminosity"] = np.empty((*new_shape, nlines))

    # ... optionally the intrinsic line luminosity ...
    if include_intrinsic:
        lines["luminosity_intrinsic"] = np.empty((*new_shape, nlines))

    # ... but only save continuum values if spectra are provided.
    continuum_quantities = [
        "transmitted",
        "incident",
        "nebular_continuum",
        "total_continuum",
    ]

    if calculate_continuum:
        spectra_ = {}

        # Calculate incideted_contiuum
        spectra_["incident"] = spectra["incident"]

        # Calculate transmitted_contiuum
        spectra_["transmitted"] = spectra["transmitted"]

        # Calculate nebular_contiuum
        spectra_["nebular_continuum"] = (
            spectra["nebular"] - spectra["linecont"]
        )

        # Calculate total_continuum. Note this is not the same as adding
        #  nebular and transmitted since that would include emission lines.
        # The incident light scattered back towards the source is continuum
        # leaving the cloud too (zero in spherical geometry).
        spectra_["total_continuum"] = (
            spectra_["nebular_continuum"]
            + spectra_["transmitted"]
            + spectra["reflected_incident"]
        )

        # Define output arrays
        for continuum_quantity in continuum_quantities:
            lines[continuum_quantity] = np.empty((*new_shape, nlines))

    # Array for recording cloudy failures
    failures = np.zeros(new_shape)

    # Points below the ionising threshold, filled once the line wavelengths
    # are known
    quiet_points = []

    # Loop over incident models and photoionisation models
    for incident_index, incident_index_tuple in enumerate(incident_index_list):
        for photoionisation_index, photoionisation_index_tuple in enumerate(
            photoionisation_index_list
        ):
            # Get the indices of the grid_point
            indices = tuple(incident_index_tuple) + tuple(
                photoionisation_index_tuple
            )

            # The infile
            infile = (
                f"{cloudy_dir}/{new_grid_name}/"
                f"{incident_index}/{photoionisation_index}"
            )

            if _below(below_threshold, incident_index_tuple):
                quiet_points.append(indices)
                continue

            # Check to see if the model failed or not
            failed = check_if_failed(
                infile, extensions_to_check=extensions_to_check
            )

            model = f"{incident_index}_{photoionisation_index}"
            if failed and model in interpolations:
                # Interpolate between the runs perturbed either side of it,
                # each normalised to the grid point (they ran at slightly
                # different ionising luminosities)
                log10_q = (
                    new_grid.read_dataset(
                        "log10_specific_ionising_luminosity/HI",
                        indices=indices,
                    )
                    if calculate_continuum
                    else None
                )
                extensions = ["emergent_elin"]
                if include_intrinsic:
                    extensions.append("intrinsic_elin")
                totals = {extension: 0.0 for extension in extensions}
                pairs = interpolations[model]
                for lower, upper, weight in pairs:
                    for path, share in ((lower, 1 - weight), (upper, weight)):
                        normalisation = (
                            _normalised_run(
                                cloudy, path, geometry, lam, log10_q
                            )[1]
                            if calculate_continuum
                            else 1.0
                        )
                        for extension in extensions:
                            ids, line_wavelengths, luminosities = read_lines(
                                path, extension
                            )
                            totals[extension] = totals[extension] + (
                                share
                                * luminosities
                                * normalisation
                                / len(pairs)
                            )
                if "wavelength" not in lines:
                    lines["wavelength"], lines["id"] = line_wavelengths, ids
                lines["luminosity"][indices] = totals["emergent_elin"]
                if include_intrinsic:
                    lines["luminosity_intrinsic"][indices] = totals[
                        "intrinsic_elin"
                    ]
                if calculate_continuum:
                    for continuum_quantity in continuum_quantities:
                        lines[continuum_quantity][indices] = np.interp(
                            line_wavelengths,
                            lam,
                            spectra_[continuum_quantity][indices],
                        )

            # If the model has failed save the luminosities as an array of
            # zeros
            elif failed:
                # Set the line luminosities to zero
                lines["luminosity"][indices] = np.zeros(nlines)
                if include_intrinsic:
                    lines["luminosity_intrinsic"][indices] = np.zeros(nlines)

                # Set the continuum luminosities to zero
                if calculate_continuum:
                    for continuum_quantity in continuum_quantities:
                        lines[continuum_quantity][indices] = np.zeros(nlines)

                # Record failures in array
                failures[indices] = 1

            else:
                # Read line quantities
                ids, line_wavelengths, luminosities = cloudy.read_linelist(
                    infile, extension="emergent_elin"
                )

                # re-order by wavelength
                sorted_indices = np.argsort(line_wavelengths)
                ids = ids[sorted_indices]
                luminosities = luminosities[sorted_indices]
                line_wavelengths = line_wavelengths[sorted_indices]

                # Save the wavelength grid from the first successful model
                # (the first grid point may have failed)
                if "wavelength" not in lines:
                    lines["wavelength"] = line_wavelengths
                    # for id, lam in zip(ids, sorted_indices):
                    #     print(id, lam)

                    # Record list of IDs
                    lines["id"] = ids

                # If spectra have been calculated extract the normalisation ..
                if calculate_continuum:
                    normalisation = spectra["normalisation"][indices]

                # Otherwise set the normalisation to unity
                else:
                    normalisation = 1.0

                # Calculate line luminosity and save it. Uses normalisation
                # from spectra. Units are erg/s (written out in Lsun)
                lines["luminosity"][indices] = luminosities * normalisation

                # Read, sort and save the intrinsic line luminosities in the
                # same way, using the same normalisation.
                if include_intrinsic:
                    (
                        ids_intrinsic,
                        line_wavelengths_intrinsic,
                        luminosities_intrinsic,
                    ) = cloudy.read_linelist(
                        infile, extension="intrinsic_elin"
                    )
                    sorted_indices_intrinsic = np.argsort(
                        line_wavelengths_intrinsic
                    )
                    ids_intrinsic = ids_intrinsic[sorted_indices_intrinsic]
                    luminosities_intrinsic = luminosities_intrinsic[
                        sorted_indices_intrinsic
                    ]

                    # Both files use the same linelist so the ordering
                    # should match; guard against silent misalignment.
                    if not np.array_equal(ids_intrinsic, ids):
                        raise ValueError(
                            "Line IDs in intrinsic_elin do not match "
                            f"emergent_elin for model {infile}"
                        )

                    lines["luminosity_intrinsic"][indices] = (
                        luminosities_intrinsic * normalisation
                    )

                # Calculate continuum luminosities. Units are erg/s/Hz.
                if calculate_continuum:
                    for continuum_quantity in continuum_quantities:
                        lines[continuum_quantity][indices] = np.interp(
                            line_wavelengths,
                            lam,
                            spectra_[continuum_quantity][indices],
                        )

    # Without a single successful model there are no line wavelengths or IDs
    if "wavelength" not in lines:
        raise ValueError(
            f"All cloudy models in {cloudy_dir}/{new_grid_name} failed "
            f"(checked extensions: {', '.join(extensions_to_check)}); "
            "cannot build the lines grid."
        )

    # Below the ionising threshold there are no lines, and the continua are
    # the (transmitted) incident light's
    for indices in quiet_points:
        lines["luminosity"][indices] = 0.0
        if include_intrinsic:
            lines["luminosity_intrinsic"][indices] = 0.0
        if calculate_continuum:
            for continuum_quantity in continuum_quantities:
                lines[continuum_quantity][indices] = np.interp(
                    lines["wavelength"],
                    lam,
                    spectra_[continuum_quantity][indices],
                )

    # Apply units to the wavelength
    lines["wavelength"] *= Angstrom

    # Apply units to the lines
    lines["luminosity"] *= erg / s
    if include_intrinsic:
        lines["luminosity_intrinsic"] *= erg / s

    # If continuum values are calculated add units
    for continuum_quantity in continuum_quantities:
        if continuum_quantity in lines:
            lines[continuum_quantity] *= erg / s / Hz

    # Write the lines out
    new_grid.write_lines(
        lines,
    )

    # Need to write the dataset containing the failures
    # This is repeated if spectra were made.
    if spectra is not None:
        print("Dataset of failed models already written out!")
    else:
        new_grid.write_dataset(
            "failures",
            failures * dimensionless,
            "array of model failures",
            False,
            verbose=False,
        )
        interpolated = np.zeros(new_shape)
        for model in interpolations:
            i, j = (int(index) for index in model.split("_"))
            interpolated[
                tuple(incident_index_list[i])
                + tuple(photoionisation_index_list[j])
            ] = 1
        new_grid.write_dataset(
            "interpolated",
            interpolated * dimensionless,
            "1 where the model failed and was interpolated between runs "
            "perturbed either side of it",
            False,
            verbose=False,
        )


def create_synthesizer_grid(
    grid_dir,
    cloudy_dir,
    incident_grid_name,
    cloudy_param_file,
    extra_cloudy_param_file=None,
    include_spectra=False,
    include_intrinsic_lines=False,
    machine=None,
    cloudy_executable_path=None,
    incident_grid_dir=None,
    verbose=False,
):
    """
    Make a synthesizer grid from the cloudy outputs of an incident grid.

    Failed models are recorded in the grid's failures dataset, with zeros in
    their spectra and lines.

    Args:
        grid_dir (str):
            The directory the new grid is written to (and holding the
            incident grid, unless incident_grid_dir is given).
        cloudy_dir (str):
            The directory holding the cloudy outputs, in
            <new_grid_name>/<incident index>/<photoionisation index>.
        incident_grid_name (str):
            The incident grid's name (with or without .hdf5).
        cloudy_param_file (str):
            The cloudy parameter file (see resolve_param_file).
        extra_cloudy_param_file (str, optional):
            A parameter file whose parameters supersede cloudy_param_file's.
        include_spectra (bool):
            Store the spectra as well as the lines.
        include_intrinsic_lines (bool):
            Also store the intrinsic line luminosities.
        machine (str, optional):
            Write a submission script for this machine rerunning failures.
        cloudy_executable_path (str, optional):
            The cloudy directory, for the submission script.
        incident_grid_dir (str, optional):
            The directory holding the incident grid, if not grid_dir.
        verbose (bool):
            Print more.

    Returns:
        str:
            The new grid's path.
    """
    cloudy_param_file = resolve_param_file(cloudy_param_file)
    if extra_cloudy_param_file is not None:
        extra_cloudy_param_file = resolve_param_file(extra_cloudy_param_file)

    print(" " * 80)
    print("-" * 80)
    print("incident_grid_name:", incident_grid_name)
    print("cloudy_param_file:", cloudy_param_file)
    print("extra_cloudy_param_file:", extra_cloudy_param_file)

    # Names can hold dots (e.g. "...-0.1,100"), so only drop the extension
    incident_grid_name = incident_grid_name.removesuffix(".hdf5")

    # Get name of new grid (concatenation of incident_grid and cloudy
    # parameter file)
    new_grid_name = f"{incident_grid_name}_cloudy-{cloudy_param_file.stem}"

    # If an additional parameter set append this to the new grid name
    if extra_cloudy_param_file:
        new_grid_name += "-" + extra_cloudy_param_file.stem

    new_grid_file = new_grid_name + ".hdf5"

    # Open the incident grid using synthesizer
    incident_grid = Grid(
        incident_grid_name,
        grid_dir=incident_grid_dir or grid_dir,
        ignore_lines=True,
    )

    # Extract axes and axes values from the Grid
    incident_axes = incident_grid.axes
    incident_axes_values = incident_grid.axes_values

    # Get properties of the incident grid
    (
        incident_n_axes,
        incident_shape,
        incident_n_models,
        incident_mesh,
        incident_model_list,
        incident_index_list,
    ) = get_grid_props_cloudy(
        incident_axes, incident_axes_values, verbose=False
    )

    # Load the cloudy parameters you are going to run
    fixed_photoionisation_params, variable_photoionisation_params = (
        get_cloudy_params(cloudy_param_file)
    )

    # If an additional parameter set is provided supersede the default
    # parameters with these.
    if extra_cloudy_param_file:
        (photoionisation_fixed_params_, photoionisation_variable_params_) = (
            get_cloudy_params(extra_cloudy_param_file)
        )
        fixed_photoionisation_params |= photoionisation_fixed_params_
        variable_photoionisation_params |= photoionisation_variable_params_

    print("fixed photoionisation parameters:")
    for k, v in fixed_photoionisation_params.items():
        print(f"    {k}: {v}")
    print("variable photoionisation parameters:")
    for k, v in variable_photoionisation_params.items():
        print(f"    {k}: {v}")

    # If we have photoionisation parameters that vary we need to calculate the
    # model list
    if len(variable_photoionisation_params) > 0:
        # Sort the axes to ensure consistent ordering
        photoionisation_axes = sorted(variable_photoionisation_params.keys())

        # get properties of the photoionsation grid
        (
            photoionisation_n_axes,
            photoionisation_shape,
            photoionisation_n_models,
            photoionisation_mesh,
            photoionisation_model_list,
            photoionisation_index_list,
        ) = get_grid_props_cloudy(
            photoionisation_axes,
            variable_photoionisation_params,
            verbose=False,
        )

    # Else, we need to specify that the only photonionisation model is 0
    else:
        photoionisation_index_list = [[]]
        photoionisation_shape = ()

    # Define the axes and values of the output grid
    new_axes = list(incident_axes)
    new_axes_values = incident_axes_values.copy()

    new_shape = tuple(incident_shape) + tuple(photoionisation_shape)
    if verbose:
        print("new shape:", new_shape)

    # If there are photoionisation axes add these as well
    if len(variable_photoionisation_params) > 0:
        new_axes += photoionisation_axes
        new_axes_values |= variable_photoionisation_params

    for key, value in new_axes_values.items():
        print(key, value)

    # Create empty synthesizer grid, this returns a GridFile object we can
    # add spectra and lines to below
    new_grid, incident_grid = create_empty_grid(
        grid_dir,
        incident_grid_name,
        new_grid_file,
        new_axes,
        new_axes_values,
        fixed_photoionisation_params,
        incident_grid_dir=incident_grid_dir,
        param_files=[
            path
            for path in (cloudy_param_file, extra_cloudy_param_file)
            if path is not None
        ],
    )

    # Cloudy output files that must exist for a model to count as successful
    geometry = fixed_photoionisation_params.get("geometry")
    extensions_to_check = required_extensions(
        include_spectra, include_intrinsic_lines, geometry
    )

    # Incident points with too few ionising photons weren't run
    below_threshold = below_ionising_threshold(
        incident_grid.grid_filename,
        fixed_photoionisation_params.get(THRESHOLD_KEY),
    )

    # Failed models that can be interpolated from runs perturbed around them
    interpolations = _interpolations(
        cloudy_dir, new_grid_name, extensions_to_check
    )
    if interpolations:
        print(
            f"{len(interpolations)} failed models are interpolated from runs "
            "perturbed around them"
        )

    # Check cloudy runs and potentially replace them by the nearest grid point
    # if they fail.
    check_cloudy_runs(
        new_grid_name,
        cloudy_dir,
        incident_index_list,
        photoionisation_index_list,
        extensions_to_check=extensions_to_check,
        machine=machine,
        cloudy_executable_path=cloudy_executable_path,
        below_threshold=below_threshold,
    )

    # Even if a grid point has failed the code will now still create a grid,
    # recording failures with zeros in the spectra and line quantities. The
    # code will also save an array of failure flags.

    # Now add spectra.
    if include_spectra:
        lam, spectra = add_spectra(
            new_grid,
            new_grid_name,
            cloudy_dir,
            incident_index_list,
            photoionisation_index_list,
            new_shape,
            spec_names=SPEC_NAMES,
            norm_by_q=True,
            incident_grid_file=incident_grid.grid_filename,
            extensions_to_check=extensions_to_check,
            geometry=geometry,
            below_threshold=below_threshold,
            interpolations=interpolations,
        )
        print("Added spectra")
        print(spectra.keys())
    else:
        lam = None
        spectra = None

    # Now add lines
    add_lines(
        new_grid,
        new_grid_name,
        cloudy_dir,
        incident_index_list,
        photoionisation_index_list,
        new_shape,
        lam,
        spectra,
        line_type="linelist",
        include_intrinsic=include_intrinsic_lines,
        extensions_to_check=extensions_to_check,
        below_threshold=below_threshold,
        geometry=geometry,
        interpolations=interpolations,
    )
    print("Added lines")

    # Mark the points below the ionising threshold, over every
    # photoionisation point
    extra = len(new_shape) - below_threshold.ndim
    new_grid.write_dataset(
        "below_threshold",
        np.broadcast_to(
            below_threshold.reshape(below_threshold.shape + (1,) * extra),
            new_shape,
        ).astype(float)
        * dimensionless,
        "1 where the incident ionising luminosity is below the "
        f"threshold ({THRESHOLD_KEY}), so cloudy wasn't run",
        False,
        verbose=False,
    )

    # The Cloudy version the models ran with, from one of their outputs
    out_file = find_out_file(os.path.join(cloudy_dir, new_grid_name))
    if out_file is not None:
        write_cloudy_build(new_grid.filepath, out_file=out_file)

    return new_grid.filepath


if __name__ == "__main__":
    # Set up the command line arguments
    parser = Parser(
        description="Create Synthesizer HDF5 grid files from cloudy outputs.",
        cloudy_args=True,
    )

    # Add additional parameters which are specific to this script

    # Machine (for submission script generation for failed models)
    parser.add_argument("--machine", type=str, required=False, default=None)

    # Path to cloudy directory (not the executable; this is assumed to
    # {cloudy}/{cloudy_version}/source/cloudy.exe)
    parser.add_argument(
        "--cloudy-executable-path", type=str, required=False, default=None
    )

    # Should we include the spectra in the grid?
    parser.add_argument(
        "--include-spectra",
        action="store_true",
        help="Should the spectra be included in the grid?",
        default=False,
        required=False,
    )

    # Should we include the intrinsic line luminosities in the grid?
    parser.add_argument(
        "--include-intrinsic-lines",
        action="store_true",
        help="Also store intrinsic line luminosities (from the cloudy "
        "'intrinsic_elin' output) as lines/luminosity_intrinsic.",
        default=False,
        required=False,
    )

    # Should we normalise by the specific ionising luminosity?
    parser.add_argument(
        "--norm-by-Q",
        "-Q",
        default=True,
        required=False,
        help="Should the grid be normalised by the specific "
        "ionising luminosity? (default is True)",
    )

    # Parse arguments
    args = parser.parse_args()

    create_synthesizer_grid(
        grid_dir=args.grid_dir,
        cloudy_dir=args.cloudy_output_dir,
        incident_grid_name=args.incident_grid,
        cloudy_param_file=args.cloudy_paramfile,
        extra_cloudy_param_file=args.cloudy_paramfile_extra,
        include_spectra=args.include_spectra,
        include_intrinsic_lines=args.include_intrinsic_lines,
        machine=args.machine,
        cloudy_executable_path=args.cloudy_executable_path,
        verbose=args.verbose,
    )
