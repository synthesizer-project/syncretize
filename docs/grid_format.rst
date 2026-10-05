Grid file format
================

A Synthesizer grid is an HDF5 file containing coordinates, emission arrays,
units, and metadata. Its defining feature is dimensional flexibility: the
format does not prescribe age and metallicity, or even two axes. A published
grid has one or more physical axes and may have as many as its model needs.

The root ``axes`` attribute provides the contract. It lists axis names in
array-dimension order. Every emission array follows that order.

.. code-block:: text

   axes = [axis_0, axis_1, ..., axis_N]

   spectra shape = (n_0, n_1, ..., n_N, n_wavelength)
   line shape    = (n_0, n_1, ..., n_N, n_lines)
   scalar-grid shape = (n_0, n_1, ..., n_N)

For example, a grid with axes ``ages``, ``metallicities``, and
``imf_high_mass_slopes`` stores spectra with shape
``(n_age, n_metallicity, n_slope, n_wavelength)``. Adding another axis only
adds another leading dimension and its metadata.

Core layout
-----------

.. code-block:: text

   /
   |-- attributes: axes, WeightVariable, versions, date_created
   |-- axes/
   |   |-- <axis_0>
   |   `-- <axis_N>
   |-- spectra/
   |   |-- wavelength
   |   `-- <spectrum_name>
   |-- Model/
   |-- lines/
   |-- CloudyParams/
   `-- log10_specific_ionising_luminosity/

Only pieces needed by a particular grid must be present. ``GridFile`` writes
the standard provenance and weighting metadata automatically.

New files record ``syncretize_version``. They also record the legacy
``synthesizer_grids_version`` key with the same value so existing grid
catalogues and readers remain compatible.

Required for an incident spectral grid
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 26 46

   * - Location
     - Requirement
     - Meaning
   * - Root ``axes`` attribute
     - Required
     - Ordered names of all physical axes.
   * - ``axes/<name>``
     - Required for each axis
     - One-dimensional coordinate array.
   * - Numeric dataset ``Units`` attribute
     - Required by ``GridFile``
     - A unit string understood by ``unyt``.
   * - Axis ``log_on_read`` attribute
     - Recommended
     - Whether Synthesizer interpolates in log space.
   * - ``spectra/wavelength``
     - Required for spectra
     - Shared one-dimensional wavelength sampling.
   * - ``spectra/<name>``
     - At least one
     - Usually ``incident`` for an incident grid.
   * - Root ``WeightVariable`` attribute
     - Standard
     - Emitter property represented by one unit of emission.
   * - ``Model`` attributes
     - Required for discoverability
     - Model name, version, family, IMF, and other provenance.

Units
-----

Every numeric dataset stores its units in a ``Units`` attribute. The format
therefore does not prescribe particular storage units for axes, wavelengths,
spectra, lines, or auxiliary data. ``GridFile`` takes the units directly from
the supplied ``unyt`` arrays; quantities without physical units must be marked
as ``dimensionless``.

Axes
----

Axis names are not limited to a fixed vocabulary. They connect stored
coordinates to properties supplied when Synthesizer extracts emission. Follow
four rules:

* Use plural property names such as ``ages``, ``metallicities``, ``masses``,
  and ``spins``.
* Store linear values. Never put ``log10`` in an axis name; use
  ``log_on_read=True`` instead.
* Keep values finite, unique, and strictly increasing.
* Give each custom axis a unit and description. Use ``dimensionless`` when no
  physical unit applies.

If ``log_on_read`` is true, every coordinate must be positive. Neither HDF5 nor
``GridFile`` validates these interpolation conditions.

Spectra
-------

``spectra/wavelength`` stores the shared wavelength axis, while each other
dataset stores a spectral component.

Spectral component names are not prescribed by the format. Apart from the
reserved ``wavelength`` and ``normalisation`` keys, any dataset name under
``spectra/`` is exposed by Synthesizer as a spectrum with that name. The names
below are conventions used by many existing grids and by the photoionisation
workflow; they are not required for a general grid:

``incident``
   Input model emission before photoionisation processing.

``transmitted``
   Incident emission after passage through the gas.

``nebular``
   Total nebular emission.

``linecont``
   Nebular line emission represented on the wavelength grid.

Component names are discovered directly from the datasets in ``spectra/``. A
``spec_names`` attribute is not required.

Weighting
---------

``WeightVariable`` tells Synthesizer how to scale one grid unit. Common values
are:

* ``initial_masses`` for SPS emission per unit initial stellar mass.
* ``bolometric_luminosities`` for AGN emission normalised by bolometric
  luminosity.
* The string ``"None"`` when no emitter weighting is needed.

Pass ``weight="None"`` to ``write_grid_common`` for the unweighted case;
Python ``None`` cannot be stored as an HDF5 attribute. Other values must match
a property available to the emitting object. Syncretize does not validate
custom values.

Model metadata
--------------

The ``Model`` group stores attributes rather than datasets. Recommended SPS
metadata includes ``sps_name``, ``sps_version``, ``sps_variant``, ``imf_type``,
``imf_masses``, and ``imf_slopes`` where applicable. Recommended AGN metadata
includes ``name``, ``type="agn"``, and ``family``.

Synthesizer can load a grid without these fields, but they are needed to
identify, reproduce, and catalogue it.

Ionising luminosities and auxiliary arrays
------------------------------------------

``log10_specific_ionising_luminosity/<ion>`` stores an axis-shaped array for
each ion, normally ``HI`` and ``HeII``. ``GridFile.add_specific_ionising_lum``
calculates these from ``spectra/incident``.

Other model-specific arrays, such as ``star_fraction``, may be stored at the
root. Their shape should match the ordered physical axes.

Reprocessed grids
-----------------

A reprocessed grid has a ``CloudyParams`` group. Its attributes hold fixed
photoionisation parameters; varying parameters become additional grid axes.
The format places no special limit on how many incident or photoionisation
axes exist.

Lines use this layout:

.. code-block:: text

   lines/id                    (n_lines,)
   lines/wavelength            (n_lines,)
   lines/luminosity            (..., n_lines)
   lines/nebular_continuum     (..., n_lines)
   lines/transmitted           (..., n_lines)

The leading dimensions follow root ``axes``. Current Synthesizer line loading
expects all five datasets above. Additional line-continuum datasets are
allowed.

What is not validated
---------------------

``GridFile`` checks units are attached and prevents duplicate writes. It does
not validate array shapes, coordinate ordering, finite values, wavelength
ordering, or consistency between axes, spectra, and lines. Check those before
writing and load the finished file with Synthesizer.

Use ``syndex-check`` before publishing. It catches catalogue-level metadata
and naming problems, but it is not a replacement for scientific validation.
See :doc:`uploading_to_syndex` and the full `Syndex documentation
<https://synthesizer-project.org/syndex/docs/>`_ for validation and submission
instructions.
