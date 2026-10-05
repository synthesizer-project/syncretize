Making your own grid
====================

Use ``GridFile`` to convert model output into the :doc:`grid_format` expected
by Synthesizer. It creates the file, records package versions, and writes the
standard axes and spectra structure. The example uses age and metallicity, but
``GridFile`` supports any number of axes.

Before you start
----------------

Install Syncretize from a checkout:

.. code-block:: console

   pip install .

Numeric datasets must be ``unyt`` arrays, including dimensionless quantities.
Choose axis names and their order before constructing the spectra array:

* Use as many physical axes as the model needs.
* Use plural axis names, such as ``ages`` and ``metallicities``.
* Axis insertion order defines the leading dimensions of every spectrum.
* The final spectrum dimension is wavelength.
* Use ``log_on_read=True`` only when Synthesizer should interpolate that axis
  in logarithmic space. Axis values must then be positive.

Minimal incident grid
---------------------

This complete example writes a three-axis SPS grid. The third axis could be
replaced by any other model parameter:

.. code-block:: python

   import numpy as np
   from unyt import Angstrom, Hz, dimensionless, erg, s, yr

   from syncretize.grid_io import GridFile

   ages = np.array([1e6, 1e7, 1e8]) * yr
   metallicities = np.array([0.001, 0.01, 0.02]) * dimensionless
   imf_high_mass_slopes = np.array([1.8, 2.3, 2.8]) * dimensionless
   wavelength = np.linspace(100, 10_000, 1000) * Angstrom

   axes = {
       "ages": ages,
       "metallicities": metallicities,
       "imf_high_mass_slopes": imf_high_mass_slopes,
   }
   spectra = {
       "incident": np.ones(
           (
               len(ages),
               len(metallicities),
               len(imf_high_mass_slopes),
               len(wavelength),
           )
       )
       * erg / s / Hz,
   }

   expected_shape = tuple(len(axis) for axis in axes.values()) + (
       len(wavelength),
   )
   assert spectra["incident"].shape == expected_shape
   for axis in axes.values():
       assert axis.ndim == 1
       assert np.isfinite(axis).all()
       assert (np.diff(axis) > 0).all()

   grid = GridFile("example-grid.hdf5")
   grid.write_grid_common(
       axes=axes,
       wavelength=wavelength,
       spectra=spectra,
       log_on_read={
           "ages": True,
           "metallicities": False,
           "imf_high_mass_slopes": False,
       },
       descriptions={
           "imf_high_mass_slopes": "High-mass IMF power-law slope",
       },
       model={
           "sps_name": "example",
           "sps_version": "1.0",
           "imf_type": "bpl",
       },
       weight="initial_masses",
   )

.. warning::

   Constructing ``GridFile(path)`` always creates a new file and overwrites an
   existing file at that path.

What ``write_grid_common`` writes
---------------------------------

``write_grid_common`` creates:

.. list-table::
   :header-rows: 1

   * - Location
     - Contents
   * - Root attributes
     - Ordered ``axes``, ``WeightVariable``, creation date, and package versions.
   * - ``Model``
     - Attributes supplied in the ``model`` dictionary.
   * - ``axes/<name>``
     - One unit-aware dataset per axis, including its description and
       ``log_on_read`` setting.
   * - ``spectra/wavelength``
     - Shared wavelength sampling.
   * - ``spectra/<name>``
     - One array per spectral component.

Common axes have built-in descriptions. Every custom axis needs an entry in
``descriptions``, as shown for ``imf_high_mass_slopes`` above.

Model metadata
--------------

Metadata makes grids understandable outside the script that generated them.
For an SPS grid, include at least ``sps_name`` and ``sps_version``. Record the
IMF and model variant where applicable. For an AGN grid, include
``type="agn"``, ``name``, and ``family``. These values become attributes of the
``Model`` group.

Additional datasets
-------------------

Write model-specific numeric data with ``write_dataset``. Data must carry
units; use ``dimensionless`` rather than a bare NumPy array:

.. code-block:: python

   grid.write_dataset(
       "star_fraction",
       stellar_fraction * dimensionless,
       "Remaining stellar mass fraction",
       log_on_read=False,
   )

Use ``write_string_dataset`` for string arrays and ``write_attribute`` for
small metadata values. The target group for ``write_attribute`` must already
exist.

Lines and ionising luminosities
-------------------------------

``write_lines`` expects ``wavelength``, ``id``, and ``luminosity`` entries.
Any additional entries are written as continuum-luminosity datasets. A
reprocessed grid loadable by current Synthesizer also needs
``nebular_continuum`` and ``transmitted``:

.. code-block:: python

   grid.write_lines(
       {
           "wavelength": line_wavelengths,
           "id": ["H  1  4861.33A", "H  1  6562.81A"],
           "luminosity": line_luminosities,
           "nebular_continuum": continuum_luminosities,
           "transmitted": transmitted_continuum_luminosities,
       }
   )

After writing an ``incident`` spectrum, calculate ionising photon luminosities
with ``grid.add_specific_ionising_lum()``. This reads every grid point and can
be expensive for large grids.

Validate the result
-------------------

First load the result with Synthesizer:

.. code-block:: python

   from synthesizer.grid import Grid

   result = Grid("example-grid", grid_dir=".", ignore_lines=True)
   print(result.axes)
   print(result.available_spectra)

Before loading or sharing a grid, check these invariants:

* Every axis is one-dimensional, finite, unique, and increasing.
* Spectrum leading dimensions match axis lengths in root ``axes`` order.
* Every spectrum's final dimension matches ``wavelength``.
* Logged axes contain only positive values.
* Line arrays share one final line dimension and ordering.
* Units and ``WeightVariable`` describe the physical normalisation.

Then install Syndex and run its structural checker. Syndex requires Python
3.10 or later; use a separate validation environment if grid generation runs
on Python 3.8 or 3.9:

.. code-block:: console

   pip install cosmos-syndex
   syndex-check example-grid.hdf5

Fix errors before submission. Warnings do not block publication, but usually
identify missing metadata or naming conventions that make a grid harder to
find and use.

See :doc:`grid_file` for every writer method.
