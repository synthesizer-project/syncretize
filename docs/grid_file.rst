Using ``GridFile``
==================

``GridFile`` writes arrays and metadata into the :doc:`grid_format` expected by
Synthesizer. It does not prescribe which physical parameters form the grid or
which spectral components are stored. Axis and spectrum names are supplied by
the generation script.

Creating an arbitrary grid
--------------------------

The example below creates an AGN grid with three axes and two spectral
components. ``disc`` and ``corona`` are arbitrary names; any suitable spectrum
names can be used.

.. code-block:: python

   import numpy as np
   from unyt import Angstrom, Hz, Msun, dimensionless, erg, s

   from syncretize.grid_io import GridFile

   # Axis order defines the leading dimensions of every grid-shaped array.
   axes = {
       "masses": np.array([1e7, 1e8]) * Msun,
       "accretion_rates_eddington": (
           np.array([0.01, 0.1, 1.0]) * dimensionless
       ),
       "spins": np.array([-0.9, 0.0, 0.9]) * dimensionless,
   }
   wavelength = np.linspace(100.0, 30_000.0, 1000) * Angstrom

   grid_shape = tuple(len(values) for values in axes.values())
   spectra_shape = grid_shape + (len(wavelength),)

   # Replace these arrays with spectra calculated by the source model.
   disc = np.ones(spectra_shape) * erg / s / Hz
   corona = np.full(spectra_shape, 0.1) * erg / s / Hz

   grid = GridFile("custom-agn.hdf5")
   grid.write_grid_common(
       axes=axes,
       wavelength=wavelength,
       spectra={
           "disc": disc,
           "corona": corona,
       },
       log_on_read={
           "masses": True,
           "accretion_rates_eddington": True,
           "spins": False,
       },
       descriptions={
           "masses": "Black hole mass",
           "accretion_rates_eddington": (
               "Accretion rate as a fraction of the Eddington rate"
           ),
           "spins": "Dimensionless black hole spin",
       },
       model={
           "name": "custom-agn",
           "type": "agn",
           "family": "example",
       },
       weight="bolometric_luminosities",
   )

This produces:

.. code-block:: text

   custom-agn.hdf5
   |-- attributes: axes, WeightVariable, versions, date_created
   |-- Model/
   |-- axes/
   |   |-- masses
   |   |-- accretion_rates_eddington
   |   `-- spins
   `-- spectra/
       |-- wavelength
       |-- disc
       `-- corona

The spectrum arrays have shape
``(n_mass, n_accretion_rate, n_spin, n_wavelength)`` because that is the order
of the ``axes`` dictionary. Changing the axes only requires changing that
dictionary, the corresponding ``log_on_read`` and ``descriptions`` entries,
and the leading dimensions of the spectra.

.. warning::

   ``GridFile(path)`` immediately creates a new file. An existing file at the
   same path is overwritten.

Adding other data
-----------------

Model-specific arrays can be added with ``write_dataset``. Their shape is not
fixed by ``GridFile``; it should follow the axes relevant to that quantity. For
example, an axis-shaped coronal fraction can be added with:

.. code-block:: python

   coronal_fraction = np.full(grid_shape, 0.1) * dimensionless
   grid.write_dataset(
       "coronal_fraction",
       coronal_fraction,
       "Fraction of the bolometric luminosity emitted by the corona",
       log_on_read=False,
   )

``write_string_dataset`` writes arrays of strings, while ``write_attribute``
writes small metadata values to an existing group.

Preparing an incident grid for reprocessing
-------------------------------------------

The general grid format allows arbitrary spectrum names, but the
:doc:`reprocessing` workflow expects a component named ``incident``. After
writing that component, its ionising luminosities can be calculated with:

.. code-block:: python

   grid.add_specific_ionising_lum()

This calculation loops over every grid point and may be expensive for large
grids.

Requirements
------------

``GridFile`` requires units on every numeric array, including quantities that
are ``dimensionless``. It writes the units, descriptions, and interpolation
metadata alongside each dataset. It does not verify that axis values are
ordered or that array dimensions agree, so these should be checked before the
file is written.

See :doc:`making_your_own_grid` for the complete workflow from model arrays to
validation and publication.

API reference
-------------

.. autoclass:: syncretize.grid_io.GridFile
   :members:
   :exclude-members: descriptions
   :show-inheritance:
