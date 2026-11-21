"""Build an incident grid from a model's per-point function.

Every incident model provides two functions:

    setup(config, input_dir) -> GridSpec
        Read the parameter file (and any source data) and describe the grid:
        its axes, wavelengths, outputs and metadata. This is cheap and its
        result is picklable, so it can be sent to other processes.

    generate_point(spec, index) -> dict[str, numpy.ndarray]
        Compute one point of the grid. ``index`` maps each of the spec's
        ``point_axes`` to an index along that axis. The result maps each of
        the spec's outputs to an array over the remaining (block) axes, in the
        spec's axis order, with wavelength last for spectra.

A point is the smallest piece of the grid a model computes at once: FSPS
computes every age for one metallicity, so its point axis is metallicity and
ages come back as a block; a table model reads everything at once, so it has
no point axes and a single point. ``build_grid`` loops over the points, in
parallel with ``cores`` processes when there is more than one, and writes the
grid with ``GridFile``.
"""

import itertools
import multiprocessing
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from unyt import dimensionless

from syncretize import provenance
from syncretize.grid_io import GridFile

# The source data an incident model reads, in its setup or as it makes each
# point, are recorded in the grid (see GridFile.write_source_data), so the
# reads are noted from the moment the builder is imported (before any setup)
provenance.record_reads()


@dataclass
class Output:
    """One array the model produces at every point.

    Attributes:
        units (unyt.Unit)
            The units of the values ``generate_point`` returns.
        spectra (bool)
            True for spectra (written under ``spectra/``, with wavelength as
            the last dimension), False for a dataset over the grid axes.
        description (str)
            Description of a dataset (spectra use the standard one).
    """

    units: object = dimensionless
    spectra: bool = True
    description: str = ""


@dataclass
class GridSpec:
    """Everything needed to build and write an incident grid.

    Attributes:
        grid_name (str)
            The grid's file name, without ``.hdf5``.
        model (dict)
            Model metadata, written to the ``Model`` group.
        axes (dict[str, unyt_array])
            The grid axes, in array order.
        wavelength (unyt_array)
            The wavelengths of the spectra.
        outputs (dict[str, Output])
            What ``generate_point`` returns at each point.
        point_axes (tuple of str)
            The axes looped over, one ``generate_point`` call per
            combination. The other axes come back as a block.
        log_on_read (dict[str, bool])
            Whether synthesizer interpolates each axis in log space.
        descriptions (dict[str, str])
            Descriptions of non-standard axes.
        alt_axes (tuple of str)
            Alternative axis names, if any.
        weight (str)
            What one unit of the spectra corresponds to.
        ionising_luminosity (bool)
            Whether to add log10 specific ionising luminosities (needs an
            ``incident`` spectra output).
        data (dict)
            Anything else (picklable) ``generate_point`` needs.
    """

    grid_name: str
    model: dict
    axes: dict
    wavelength: object
    outputs: dict
    point_axes: tuple = ()
    log_on_read: dict = field(default_factory=dict)
    descriptions: dict = field(default_factory=dict)
    alt_axes: tuple = ()
    weight: str = "initial_masses"
    ionising_luminosity: bool = True
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        """Check the point axes are grid axes."""
        unknown = set(self.point_axes) - set(self.axes)
        if unknown:
            raise ValueError(f"Point axes {sorted(unknown)} aren't grid axes")


def points(spec):
    """Every point of the grid, as dicts of point-axis indices."""
    ranges = [range(len(spec.axes[axis])) for axis in spec.point_axes]
    for indices in itertools.product(*ranges):
        yield dict(zip(spec.point_axes, indices))


def _location(spec, index):
    """The slice of a full-grid array that one point's block fills."""
    return tuple(index.get(axis, slice(None)) for axis in spec.axes)


def _generate(generate_point, spec, index):
    """Run generate_point for one point, keeping the index with its result."""
    return index, generate_point(spec, index)


def build_grid(
    spec, generate_point, grid_dir, param_file=None, cores=1, verbose=True
):
    """Compute every point of a grid and write it.

    Args:
        spec (GridSpec)
            The grid, from the model's ``setup``.
        generate_point (callable)
            The model's ``generate_point``. It must be a module-level function
            of an importable module (or of the script being run), so other
            processes can import it.
        grid_dir (str)
            The directory to write the grid to.
        param_file (str, optional)
            The parameter file, stored in the grid's ``ParameterFile`` group.
        cores (int)
            The number of processes to compute points with. Points are
            independent, so any number works; only models with more than one
            point gain from more than one.
        verbose (bool)
            Print progress.

    Returns:
        str
            The path of the grid written (``grid_name`` in the parameter
            file can change it).
    """
    shape = tuple(len(values) for values in spec.axes.values())
    nlam = len(spec.wavelength)
    arrays = {
        name: np.zeros(shape + ((nlam,) if output.spectra else ()))
        for name, output in spec.outputs.items()
    }

    todo = list(points(spec))
    if verbose:
        print(f"{spec.grid_name}: {len(todo)} points on {cores} core(s)")
    work = partial(_generate, generate_point, spec)
    if cores > 1 and len(todo) > 1:
        # spawn rather than fork, which can deadlock once numpy or HDF5 have
        # started threads; generate_point must therefore be importable
        context = multiprocessing.get_context("spawn")
        with context.Pool(min(cores, len(todo))) as pool:
            results = pool.imap_unordered(work, todo)
            _fill(spec, arrays, results, len(todo), verbose)
    else:
        _fill(spec, arrays, map(work, todo), len(todo), verbose)

    out_grid = GridFile(f"{grid_dir}/{spec.grid_name}.hdf5", param_file)
    out_grid.write_grid_common(
        model=spec.model,
        axes=spec.axes,
        wavelength=spec.wavelength,
        spectra={
            name: arrays[name] * output.units
            for name, output in spec.outputs.items()
            if output.spectra
        },
        log_on_read=spec.log_on_read,
        alt_axes=spec.alt_axes,
        descriptions=dict(spec.descriptions),
        weight=spec.weight,
    )
    for name, output in spec.outputs.items():
        if not output.spectra:
            out_grid.write_dataset(
                name,
                arrays[name] * output.units,
                output.description,
                log_on_read=False,
            )
    if spec.ionising_luminosity:
        out_grid.add_specific_ionising_lum()
    out_grid.write_source_data()
    return out_grid.filepath


def _fill(spec, arrays, results, n_points, verbose):
    """Put each point's blocks into the full-grid arrays."""
    for done, (index, blocks) in enumerate(results, 1):
        missing = set(spec.outputs) - set(blocks)
        if missing:
            raise ValueError(f"generate_point didn't return {sorted(missing)}")
        location = _location(spec, index)
        for name in spec.outputs:
            arrays[name][location] = blocks[name]
        if verbose and n_points > 1:
            print(f"  point {done}/{n_points}: {index}", flush=True)


def table_point(spec, index):
    """generate_point for models read whole from a table.

    Table models (e.g. BC03, BPASS, Maraston) read every spectrum at once in
    ``setup``, keep the arrays in ``spec.data["outputs"]`` and have no point
    axes, so their single point is the whole table.
    """
    return spec.data["outputs"]
