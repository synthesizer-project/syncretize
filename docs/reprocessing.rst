Reprocessing an incident grid
=============================

Syncretize uses Cloudy to add gas reprocessing to an incident grid. The core
workflow has three stages:

1. Expand incident and photoionisation axes into Cloudy input files.
2. Run Cloudy for every point in that Cartesian product.
3. Collect outputs into a new Synthesizer grid.

Cloudy 23 is the recommended path. The regular collector is not currently
complete for Cloudy 17 or 25.

Prerequisites
-------------

The incident grid must contain:

* ``spectra/incident`` and ``spectra/wavelength``;
* ordered axes and axis metadata;
* ``log10_specific_ionising_luminosity/HI``.

Included incident-model conversions normally calculate the ionising
luminosity. For a custom grid, call ``grid.add_specific_ionising_lum()`` after
writing the incident spectrum.

You also need a compiled Cloudy installation. Commands below should run from
``src/syncretize/cloudy`` because the current scripts resolve parameter
files, line lists, and helper scripts relative to that directory.

``--cloudy-executable-path`` is the directory containing version directories,
not the executable itself. For ``cloudy_version: c23.01`` the runner expects:

.. code-block:: text

   <cloudy-executable-path>/c23.01/source/cloudy.exe

Check that exact path before generating jobs.

Parameter files and added axes
------------------------------

The base YAML file defines Cloudy version, abundances, geometry, density,
ionisation parameter treatment, stopping conditions, and outputs. A second
file can override or extend it through ``--cloudy-paramfile-extra``.

A numeric list in the YAML becomes a grid axis. Nested lists become dotted
axis names, such as ``abundance_scalings.nitrogen_to_oxygen``. Scalars remain
fixed metadata under ``CloudyParams``. Syncretize takes the Cartesian product
of every incident and photoionisation axis.

This means reprocessing remains N-dimensional. For an incident grid with
shape ``(n_age, n_metallicity, n_imf)`` and Cloudy axes containing density and
ionisation parameter, the output shape is:

.. code-block:: text

   (n_age, n_metallicity, n_imf, n_density, n_ionisation_parameter)

The file format supports arbitrary added axes. The current collector also
requires each new photoionisation axis to have a plural name and unit mapping
in ``create_synthesizer_grid.py``. Add those mappings when introducing a new
parameter name.

The standard ``c23.01-sps`` file uses reference-ionisation mode and assumes
incident ``ages`` and ``metallicities`` axes. Every additional incident axis
needs a fixed ``reference_<axis-name>`` parameter; ages and metallicities use
the existing ``reference_age`` and ``reference_metallicity`` names. The regular
input-building script also currently expects a metallicity axis. Non-SPS
incident grids need a compatible parameter file and may require adapting this
mapping.

``--cloudy-paramfile`` and ``--cloudy-paramfile-extra`` take a path to a
parameter file or the name of one in ``params/`` (e.g. ``c23.01-sps``); the
``.yaml`` extension is optional. The grid is named after the file name without
its extension.

1. Generate Cloudy inputs
-------------------------

The Cloudy scripts can be run from any directory:

.. code-block:: console

   python create_cloudy_input_grid.py \
       --incident-grid my-incident-grid \
       --grid-dir /path/to/grids \
       --cloudy-output-dir /path/to/cloudy-output \
       --cloudy-paramfile c23.01-sps \
       --cloudy-executable-path /path/to/cloudy

The output grid name is based on the incident grid and parameter files. The
Cloudy work directory contains copied configuration, global grid parameters,
and one directory per incident-grid point. Each incident directory contains
the source spectrum and one Cloudy input per photoionisation-grid point.

Pass ``--machine artemis`` to generate a Slurm array script. By default each
task handles one incident-grid point and all its photoionisation points. This
usually balances repeated spectrum setup against scheduler overhead. The
script is written to ``<cloudy-output-dir>/<grid-name>/<grid-name>.slurm`` and
its logs to the ``logs`` directory beside it.

2. Run Cloudy
-------------

Run one incident point and all associated photoionisation models:

.. code-block:: console

   python run_cloudy.py \
       --grid-name my-incident-grid_cloudy-c23.01-sps \
       --cloudy-output-dir /path/to/cloudy-output \
       --cloudy-executable-path /path/to/cloudy \
       --incident-index 0

Selection behavior is:

.. list-table::
   :header-rows: 1

   * - Arguments
     - Work performed
   * - Both indices
     - One incident/photoionisation pair.
   * - Incident index only
     - Every photoionisation point for one incident point; recommended.
   * - Photoionisation index only
     - Every incident point for one photoionisation point.
   * - No indices
     - Entire grid serially; suitable only for small tests.
   * - ``--list-file``
     - Explicit pairs, normally generated when retrying failures.

``run_cloudy.py`` runs every selected model, then lists the ones that failed
(a non-zero exit status, or a ``.out`` file without ``Cloudy exited OK``) and
exits with status 1, so a scheduler marks the job as failed.

3. Assemble the final grid
--------------------------

After all jobs finish:

.. code-block:: console

   python create_synthesizer_grid.py \
       --incident-grid my-incident-grid \
       --grid-dir /path/to/grids \
       --cloudy-output-dir /path/to/cloudy-output \
       --cloudy-paramfile c23.01-sps \
       --include-spectra

Use ``--include-spectra`` for a complete current-format grid. Without it, the
collector omits continuum datasets required by current Synthesizer line
loading.

The final HDF5 file contains:

* original incident axes followed by photoionisation axes;
* original ``Model`` metadata and fixed ``CloudyParams``;
* the Cloudy parameter file in ``ParameterFile`` (and any extra one in
  ``ExtraParameterFile``), and the incident grid's own parameter file in
  ``IncidentParameterFile``, so the grid can be remade;
* incident, transmitted, nebular, and line spectra;
* line IDs, wavelengths, luminosities, and continua;
* expanded ionising-luminosity arrays;
* an axis-shaped ``failures`` mask.

Cloudy spectra and lines are normalised to the incident grid's HI ionising
photon luminosity.

Failures
--------

A model succeeded if its ``.out`` file ends with ``Cloudy exited OK`` and it
wrote a sufficiently large ``.emergent_elin`` file. Failed models are marked
``1`` in the ``failures`` dataset and their cells remain zero. The collector
prints why each failed (not run, aborted, did not converge, did not finish)
and writes the failed index pairs to
``<cloudy-output-dir>/<grid-name>/<grid-name>.failures`` for resubmission.

Do not publish a grid without checking this mask. A completed HDF5 write does
not mean every Cloudy model succeeded.

4. Validate the reprocessed grid
---------------------------------

Check the failure mask, load spectra and lines through Synthesizer, then run
the catalogue checker:

.. code-block:: python

   import h5py
   from synthesizer.grid import Grid

   path = "/path/to/grids/my-incident-grid_cloudy-c23.01-sps.hdf5"
   with h5py.File(path) as hdf:
       assert not hdf["failures"][...].any()

   grid = Grid(
       "my-incident-grid_cloudy-c23.01-sps",
       grid_dir="/path/to/grids",
   )
   print(grid.axes)
   print(grid.available_spectra)

.. code-block:: console

   syndex-check \
       /path/to/grids/my-incident-grid_cloudy-c23.01-sps.hdf5

Specialised paths
-----------------

``create_cloudy_input_sobol.py`` samples incident and photoionisation
parameters jointly for emulator or training-data generation. Its intermediate
and collected HDF5 files are not standard Synthesizer grids.
