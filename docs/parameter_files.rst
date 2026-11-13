Parameter files
===============

Every grid is described by a YAML parameter file. There are three kinds:

* **incident parameter files**, one per run of an incident-model script, which
  choose the model variant and its axes;
* **Cloudy parameter files**, which set the photoionisation model used to
  reprocess an incident grid;
* the **reduced grid** parameter file, which picks a subset of an existing
  grid.

Each script ships with example files beside it. Their comments list the
allowed values of every key, so the usual way to make a new grid is to copy
the closest example and edit it. This page describes the formats those files
share.

YAML rules
----------

Parameter files are read with PyYAML (``yaml.safe_load``), which follows YAML
1.1. A few of its rules catch people out:

* **Exponents need a decimal point and a sign.** ``1.0e+6`` is a float, but
  ``1e6`` and ``1.0e6`` are read as the *strings* ``"1e6"`` and ``"1.0e6"``.
* **A key with no value is null.** ``grains:`` is the same as
  ``grains: None`` in Python, and turns that option off.
* **Quote strings that look like numbers or booleans**, e.g.
  ``sps_version: "4.0"``. Unquoted ``yes``, ``no``, ``on`` and ``off`` are
  booleans.
* Lists can be written inline (``[1.0, 2.0]``) or one item per line
  (``- 1.0``).

Keys common to all parameter files
----------------------------------

``grid_name`` (optional)
   The name of the output grid. The grid is written to
   ``<grid-dir>/<grid_name>.hdf5`` instead of the name the script would
   otherwise build from the model and its parameters. Use it to remake a
   published grid under its published name. The reduced grid script refuses a
   ``grid_name`` equal to the grid being reduced.

``contents`` (reserved)
   Don't use ``contents`` as a mapping: the verbatim parameter file is stored
   in the grid as ``ParameterFile/contents`` (see below).

How a parameter file is stored
------------------------------

The parameter file used to make a grid is stored in the grid's
``ParameterFile`` group:

* every key is written as an attribute, and nested mappings become subgroups;
* the file itself is stored verbatim in the ``contents`` dataset, with its
  original path in the dataset's ``path`` attribute.

So a grid can be remade from the file alone (see :doc:`grid_format` for how
to extract it). Keys that HDF5 can't store directly, such as numbers used as
keys or lists of mappings, are stored as strings.

Incident parameter files
------------------------

Incident scripts take the parameter file as their first argument:

.. code-block:: console

   python <script>.py <params>.yaml --grid-dir <grid-dir> \
       [--input-dir <input-dir>] [--download]

The file describes the grid; the command line only says where to read and
write files. One run makes one grid, so to make several variants (e.g. several
IMFs) use one parameter file per variant.

There are two styles of key.

**Choices** select a model variant from a fixed list, e.g. ``imf_type`` or
``variant``. A value that isn't in the list stops the script with an error
naming the allowed values. Most SPS models only use choices; their axes
(ages, metallicities) come from the source data.

**Axes** give the values of a grid axis. A list of numbers becomes an axis and
must be strictly increasing. For the scripts that allow it (``cloudy_agn``,
and the inclination of ``qsosed``, ``relagn`` and ``relqso``), a single
number instead fixes the parameter and no axis is made.

SPS models
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Model (example file)
     - Keys
     - Values
   * - FSPS (``fsps/fsps.yaml``)
     - ``imf_type``, ``imf_masses``, ``imf_slopes``, ``libraries``
     - ``bpl`` needs four ``imf_masses`` ``[m_low, 0.5, 1.0, m_high]`` and
       three ``imf_slopes``; ``chabrier03`` needs ``[m_low, m_high]`` with
       ``0 < m_low < m_high``. Optional ``libraries`` lists the
       ``[isochrones, spectra, dust]`` FSPS must have been compiled with.
   * - FSPS variable IMF (``fsps/configs/*.yaml``)
     - ``model``, ``imf``
     - See `FSPS variable IMF`_ below.
   * - BPASS 2.2.1 (``bpass/bpass-2.2.1.yaml``)
     - ``model``, ``variant``
     - ``model`` is the BPASS IMF directory, e.g.
       ``bpass_v2.2.1_imf_chab100``; ``variant`` is ``bin`` or ``sin``.
   * - BPASS 2.3 (``bpass/bpass-2.3.yaml``)
     - ``model``, ``variant``, ``alpha_enhancement``
     - ``alpha_enhancement`` is one of -0.2, 0.0, 0.2, 0.4, 0.6, or ``axis``
       to make alpha enhancement a grid axis.
   * - BC03 (``bc03/bc03.yaml``)
     - ``isochrones``, ``imf_type``
     - ``Padova2000`` and ``chabrier03`` only.
   * - BC03 2016 (``bc03/bc03-2016.yaml``)
     - ``variant``, ``imf_type``
     - ``variant``: ``BaSeL``, ``Miles`` or ``Stelib``; ``imf_type``:
       ``kroupa``, ``salpeter`` or ``chabrier``.
   * - Maraston 2005 (``maraston/maraston05.yaml``)
     - ``imf_type``, ``hr_morphology``
     - ``salpeter`` and ``rhb`` only.
   * - Maraston 2011 (``maraston/maraston11.yaml``)
     - ``variant``, ``imf_type``
     - ``variant``: ``standard``, ``nearIRextended``, ``UVtheoretical`` or
       ``UVtheoretical_nearIRextended`` (the last two Salpeter only).
   * - Maraston 2013 (``maraston/maraston13.yaml``)
     - ``imf_type``
     - ``salpeter`` or ``kroupa``.
   * - Maraston 2024 (``maraston/maraston24.yaml``)
     - ``imf_type``, ``rotation``, ``model_type``
     - ``rotation``: 0.0 or 0.4; ``model_type``: ``Te`` or ``Tenc``.
   * - Yggdrasil (``yggdrasil/yggdrasil.yaml``)
     - ``population``, ``covering_fraction``
     - ``population``: ``PopIII.1``, ``PopIII.2`` or ``PopIII_kroupa``;
       ``covering_fraction``: 0, 0.5 or 1. Use 0 for grids to reprocess
       with Cloudy.

FSPS variable IMF
^^^^^^^^^^^^^^^^^

The variable-IMF script describes a broken power-law IMF entry by entry, so
one of its parameters can become a grid axis:

.. code-block:: yaml

   model:
     sps_version: "4.0"
     imf_type: bpl

   imf:
     mass_boundaries:
       - name: lower_mass
         fixed_value: 0.08
         vary_axis: false
       # ... one entry per boundary
     slopes:
       - name: high_mass_slope
         fixed_value: 2.3
         vary_axis: true
         axis:
           name: imf_high_mass_slope
           log_on_read: false
           values: [1.3, 1.5, 1.7, 2.0, 2.3, 2.6, 3.0]

Each entry under ``mass_boundaries`` and ``slopes`` has a ``name``, a
``fixed_value`` and ``vary_axis``. At most one entry may set
``vary_axis: true``; it then needs an ``axis`` with a ``name`` and either
explicit ``values`` or a generated ``range``:

.. code-block:: yaml

   axis:
     name: imf_high_mass_slope
     log_on_read: false
     range:
       minimum: 1.3
       maximum: 3.0
       number: 18
       spacing: linear   # or log

``values`` and ``range`` can't be used together. ``log_on_read`` sets whether
Synthesizer interpolates the axis in log space; it doesn't change how the
values are sampled. The ``fixed_value`` of a varied parameter is used in the
grid name and must lie within the axis. Slopes are dimensionless and mass
boundaries are in Msun.

Black hole models
^^^^^^^^^^^^^^^^^

Every black hole parameter file has a ``model`` key, which names the grid
(``<grid-dir>/<model>.hdf5``).

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Model (example file)
     - Keys
   * - Blackbody (``blackbody/blackbody.yaml``)
     - ``log10blackbody_temperature``: list of log10(T/K).
   * - Broken power law (``broken_power_law/bpl-feltre16.yaml``)
     - ``edges``: the four wavelengths (Angstrom) bounding three power laws;
       ``indices``: three lists of slopes, one per power law, each becoming
       an axis. Only the first list may currently hold more than one value.
   * - Cloudy AGN (``cloudy_agn/cloudy_agn.yaml``)
     - ``log10blue_bump_temperature``, ``alpha_ox``, and optional
       ``alpha_uv`` (default -0.5) and ``alpha_x`` (default -1.0). Each is an
       axis if a list or fixed if a single number.
   * - QSOSED (``qsosed/qsosed.yaml``)
     - ``log10masses`` (Msun), ``log10accretion_rates_eddington``,
       ``cosine_inclinations`` (a single number for an isotropic grid).
   * - RELQSO (``relqso/relqso.yaml``)
     - As QSOSED, plus ``spins`` in [0, 0.998].
   * - RELAGN (``relagn/relagn.yaml``)
     - As QSOSED, plus exactly one of ``spins`` or
       ``radiative_efficiencies`` (converted to spins).

Reduced grids
^^^^^^^^^^^^^

``incident/create_reduced_grid.py`` picks the nearest points of an existing
grid in ``--grid-dir``:

.. code-block:: yaml

   original_grid: bpass-2.2.1-bin_chabrier03-0.1,300.0
   ages: [1.0e+6, 1.0e+7, 1.0e+8]   # yr
   # max_age: 1.0e+8                # yr
   metallicities: [0.001, 0.01, 0.02]

``original_grid`` is the grid name without ``.hdf5``. Give any of ``ages``,
``max_age`` and ``metallicities``; axes that aren't given are kept whole.

Cloudy parameter files
----------------------

Cloudy parameter files set the photoionisation model used by
``create_cloudy_input_grid.py`` and ``create_synthesizer_grid.py``. They are
passed by name or path:

.. code-block:: console

   --cloudy-paramfile c23.01-sps [--cloudy-paramfile-extra my-overrides]

An existing path is used as given. Otherwise ``.yaml`` is added if missing
and the file is looked up in ``src/syncretize/cloudy/params``. The file's
name (without ``.yaml``) becomes part of the reprocessed grid's name, e.g.
``<incident-grid>_cloudy-c23.01-sps``.

How values are read
^^^^^^^^^^^^^^^^^^^

* **A list of numbers becomes a photoionisation axis.** Each Cloudy run uses
  one value, and the reprocessed grid gains the axis after the incident axes.
* **A mapping groups related keys.** Its entries are named with a dot, e.g.
  ``abundance_scalings.nitrogen``, and a list inside it is also an axis.
* **Anything else is fixed** and stored in the grid's ``CloudyParams``
  group.

The extra file (``--cloudy-paramfile-extra``) is read the same way and its
keys replace those of the main file. The two files must have different names,
and the extra file's name is appended to the grid name. Give a key the same
form in both files: a key that is a list in one file and a single value in the
other ends up as both an axis and a fixed value.

Keys
^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``cloudy_version``
     - The Cloudy version directory, e.g. ``c23.01``. Selects both the
       executable and the Syncretize module that writes the inputs.
   * - ``reference_abundance``
     - A Synthesizer abundance pattern, e.g. ``GalacticConcordance``.
   * - ``alpha_enhancement``
     - [alpha/Fe].
   * - ``abundance_scalings``
     - Mapping of element to a scaling (a number, or the name of a
       Synthesizer scaling function).
   * - ``depletion_model``, ``depletion_scale``
     - A Synthesizer depletion model (blank for none) and its scale
       (default 1.0; use 0.5, i.e. fstar, for
       ``Jenkins2009_Gunasekera2021``).
   * - ``grains``, ``no_grain_scaling``
     - ``Orion``, ``ISM`` or blank for no grains. Grains are scaled to the
       depletion unless ``no_grain_scaling: true``, so grains need a
       depletion model.
   * - ``geometry``
     - ``spherical`` or ``planeparallel``.
   * - ``ionisation_parameter_model``
     - ``fixed`` uses ``ionisation_parameter`` directly. ``ref`` sets
       ``reference_ionisation_parameter`` at a reference incident model and
       scales it with each model's ionising luminosity.
   * - ``ionisation_parameter``
     - U, for ``fixed``.
   * - ``reference_ionisation_parameter``, ``reference_<axis>``
     - U at the reference model, for ``ref``. Every incident axis needs a
       reference value: ``reference_age`` (yr, not log),
       ``reference_metallicity``, and ``reference_<axis>`` for any other
       axis.
   * - ``hydrogen_density``
     - n_H in cm\ :sup:`-3` (not log).
   * - ``constant_density``, ``constant_pressure``
     - Density law (choose one).
   * - ``radius``
     - Inner radius in pc (not log), spherical geometry only.
   * - ``stop_T``, ``stop_efrac``, ``stop_column_density``
     - Stopping criteria: temperature (K), electron fraction (log), and
       log10(N_H/cm\ :sup:`-2`).
   * - ``T_floor``, ``turbulence``, ``covering_factor``, ``nend``
     - Temperature floor (K), turbulence (km/s), covering factor and maximum
       number of zones.
   * - ``CMB``, ``z``, ``cosmic_rays``
     - CMB heating at redshift ``z``, and background cosmic rays.
   * - ``resolution``
     - Continuum resolution relative to Cloudy's default.
   * - ``output_cont``, ``output_linelist``, ``iterate_to_convergence``
     - Save the continuum; the line list to save (a bundled name such as
       ``linelist-standard.dat`` or a path); iterate until converged.

**On/off options are on whenever the key has a value.** ``cosmic_rays``,
``CMB``, ``constant_density`` and ``constant_pressure`` add their Cloudy
command if the key is set to anything, including ``false``. To turn one off,
leave the value blank or remove the key.

Sobol parameter files
^^^^^^^^^^^^^^^^^^^^^

``create_cloudy_input_sobol.py`` samples parameters instead of building a
grid, so a varied parameter is a range rather than a list of values:

.. code-block:: yaml

   ionisation_parameter: [0.0001, 0.1, 'log']   # [min, max, spacing]
   hydrogen_density: [1.0e+1, 1.0e+4, 'log']

The spacing is ``log`` or ``linear``. Every incident axis is sampled over its
full range by default (ages and metallicities in log); give ``ages`` or
``metallicities`` as a range to narrow it. Other keys are as for a Cloudy
parameter file.
