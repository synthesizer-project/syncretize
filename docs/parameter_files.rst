Parameter files
===============

Every grid is described by a YAML parameter file. There are two kinds:

* **incident parameter files**, one per run of an incident-model script, which
  choose the model variant and its axes;
* **Cloudy parameter files**, which set the photoionisation model used to
  reprocess an incident grid.

Parameter files live in ``parameter_files/`` at the repository root, which is
laid out like ``src/syncretize``. For each incident script there is a template
named after it (e.g. ``incident/sps/fsps/generate_incident_grid_template.yaml``)
whose inline comments describe every key and its allowed values, so the usual
way to make a new grid is to copy the template and edit it. This page
describes the formats.

YAML rules
----------

Parameter files are YAML. A few of its rules catch people out:

* **A key with no value is null.** ``grains:`` is the same as
  ``grains: None`` in Python, and turns that option off. An empty section is
  written the same way, e.g. ``Constants:``.
* **Quote strings that look like numbers or booleans**, e.g.
  ``sps_version: "4.0"``. Unquoted ``yes``, ``no``, ``on`` and ``off`` are
  booleans.
* **Cloudy parameter files need exponents with a decimal point and a sign.**
  They are read with PyYAML's YAML 1.1 rules, so ``1.0e+6`` is a float but
  ``1e6`` and ``1.0e6`` are *strings*. Incident parameter files read all
  three as floats.
* Lists can be written inline (``[1.0, 2.0]``) or one item per line
  (``- 1.0``).

How a parameter file is stored
------------------------------

The parameter file used to make an incident grid is stored in the grid's
``ParameterFile`` group:

* every key is written as an attribute, and nested mappings (such as the
  sections below) become subgroups;
* the file itself is stored verbatim in the ``contents`` dataset, with its
  original path in the dataset's ``path`` attribute.

So a grid can be remade from the file alone (see :doc:`grid_format` for how
to extract it). Values HDF5 can't store directly, such as the list of axis
entries, are stored as strings.

Incident parameter files
------------------------

Incident scripts take the parameter file with ``--config-file``:

.. code-block:: console

   python <script>.py --config-file <params>.yaml --grid-dir <grid-dir> \
       [--input-dir <input-dir>] [--download]

The file describes the grid; the rest of the command line only says where to
read and write files. One run makes one grid, so to make several variants
(e.g. several IMFs) use one parameter file per variant.

Sections
^^^^^^^^

Every incident parameter file has the same four sections, in this order. Each
must be present but may be left blank to use the script's defaults:

.. code-block:: yaml

   Model:        # which model: choices and versions
     imf_type: bpl
   Constants:    # fixed numbers
     imf_masses: [0.08, 0.5, 1, 120]
     imf_slopes: [1.3, 2.3, 2.3]
   Axes:         # one entry per grid axis
   Sampling:     # settings for a global sampler (Sobol axes only)

``Model``
   Choices that select a model variant from a fixed list, e.g. ``imf_type``,
   plus versions. A value that isn't allowed stops the script with an error
   naming the allowed values. Every script also accepts ``grid_name``: the
   grid is then written to ``<grid-dir>/<grid_name>.hdf5`` instead of the
   name the script builds from the model, e.g. to remake a published grid
   under its published name.

``Constants``
   Numbers that are fixed for the grid, e.g. IMF masses or a single
   inclination. A parameter that a script can either fix or vary goes here
   when fixed and in ``Axes`` when varied.

``Axes``
   A list of axes, each a mapping with a ``name``, a ``type`` and the values
   for that type:

   .. list-table::
      :header-rows: 1
      :widths: 20 30 50

      * - ``type``
        - Keys
        - Values
      * - ``values``
        - ``values``
        - The listed values, which must be strictly increasing.
      * - ``linear-range``
        - ``min``, ``max``, ``n``
        - ``n`` linearly spaced values, including both limits.
      * - ``log-range``
        - ``min``, ``max``, ``n``
        - ``n`` logarithmically spaced values, including both (positive)
          limits.
      * - ``sobol``
        - ``min``, ``max``, ``scale``
        - Points sampled together from one Sobol sequence (see
          ``Sampling``). ``scale`` is ``linear`` (default) or ``log``.

   Prefer explicit ``values`` for released grids, so every value is visible
   in review. Some scripts read extra keys on an axis, such as ``target`` and
   ``log_on_read`` for the FSPS variable IMF. Sobol axes can only be used by
   scripts that support them, and either every axis or none is Sobol-sampled.

``Sampling``
   Settings for the Sobol sampler: ``n_samples`` (required), ``seed`` and
   ``scramble`` (default ``true``). The points are drawn from the next power
   of two and the first ``n_samples`` kept.

A section, key or axis that the script doesn't know stops it with an error,
so a typo can't silently fall back to a default.

SPS models
^^^^^^^^^^

Most SPS models only take ``Model`` choices; their axes (ages and
metallicities) come from the source data.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Model (template)
     - Keys
   * - FSPS (``fsps/generate_incident_grid_template.yaml``)
     - ``Model.sps_version``: ``"3.2"`` or ``"4.0"``.
       ``Model.libraries``: the ``[isochrones, spectra, dust]`` FSPS was
       compiled with. ``Model.imf_type``: ``bpl`` or ``chabrier03``.
       ``Constants.imf_masses``: ``[m_low, 0.5, 1.0, m_high]`` for ``bpl``,
       ``[m_low, m_high]`` with ``0 < m_low < m_high`` for ``chabrier03``.
       ``Constants.imf_slopes``: the three ``bpl`` slopes.
   * - FSPS variable IMF (``fsps/generate_incident_grid_variable_imf_template.yaml``)
     - See `FSPS variable IMF`_ below.
   * - BPASS 2.2.1 (``bpass/generate_incident_grid_2_2_1_template.yaml``)
     - ``Model.bpass_model``: the BPASS IMF directory, e.g.
       ``bpass_v2.2.1_imf_chab100``. ``Model.variant``: ``bin`` or ``sin``.
   * - BPASS 2.3 (``bpass/generate_incident_grid_2_3_template.yaml``)
     - ``alpha_enhancement``: in ``Constants`` (-0.2, 0.0, 0.2, 0.4 or 0.6)
       for a grid at that value, or in ``Axes`` with all five values for
       alpha enhancement as an axis.
   * - BC03 (``bc03/generate_incident_grid_template.yaml``)
     - None (Padova 2000 isochrones with a Chabrier 2003 IMF only).
   * - BC03 2016 (``bc03/generate_incident_grid_2016_template.yaml``)
     - ``Model.variant``: ``BaSeL``, ``Miles`` or ``Stelib``.
       ``Model.imf_type``: ``kroupa``, ``salpeter`` or ``chabrier``.
   * - Maraston 2005 (``maraston/generate_incident_grid_2005_template.yaml``)
     - None (Salpeter IMF with a red horizontal branch only).
   * - Maraston 2011 (``maraston/generate_incident_grid_2011_template.yaml``)
     - ``Model.variant``: ``standard``, ``nearIRextended``,
       ``UVtheoretical`` or ``UVtheoretical_nearIRextended`` (the last two
       Salpeter only). ``Model.imf_type``: ``salpeter``, ``kroupa`` or
       ``chabrier``.
   * - Maraston 2013 (``maraston/generate_incident_grid_2013_template.yaml``)
     - ``Model.imf_type``: ``salpeter`` or ``kroupa``.
   * - Maraston 2024 (``maraston/generate_incident_grid_2024_template.yaml``)
     - ``Model.imf_type``: ``kroupa`` or ``salpeter``. ``Model.rotation``:
       0.0 or 0.4. ``Model.model_type``: ``Te`` or ``Tenc``.
   * - Yggdrasil (``yggdrasil/generate_incident_grid_template.yaml``)
     - ``Model.population``: ``PopIII.1``, ``PopIII.2`` or
       ``PopIII_kroupa``. ``Model.covering_fraction``: 0, 0.5 or 1; use 0
       for grids to reprocess with Cloudy.

FSPS variable IMF
^^^^^^^^^^^^^^^^^

The variable-IMF script makes one IMF parameter a grid axis. The IMF is given
in ``Constants`` and the axis names the entry it varies with ``target``:

.. code-block:: yaml

   Model:
     sps_version: "4.0"
     imf_type: bpl            # only bpl is supported
     libraries: [mist, c3k_lr]
   Constants:
     imf_masses: [0.08, 0.5, 1.0, 120.0]
     imf_slopes: [1.3, 2.3, 2.3]
   Axes:
     - name: imf_high_mass_slope
       target: imf_slopes[2]
       type: values
       values: [1.3, 1.5, 1.7, 2.0, 2.3, 2.6, 3.0]
       log_on_read: false
   Sampling:

The IMF has either two masses and one slope or four masses (with internal
breaks at 0.5 and 1.0 Msun, which can't be varied) and three slopes. At most
one axis may be given. The value in ``Constants`` at the target is used in the
grid name and must lie within the axis. ``log_on_read`` sets whether
Synthesizer interpolates the axis in log space; it doesn't change how the
values are sampled. Slopes are dimensionless and masses are in Msun.
``Model.sps_version`` and ``Model.libraries`` are as for the FSPS script.

Both FSPS scripts check ``Model.sps_version`` (``"3.2"`` needs python-fsps
0.4.7, ``"4.0"`` needs 0.5.0) and ``Model.libraries`` against the installed
python-fsps, and stop if they differ, as FSPS is compiled into python-fsps.
``src/syncretize/incident/sps/fsps/README.md`` describes how to build each
version with chosen libraries.

Black hole models
^^^^^^^^^^^^^^^^^

Every black hole parameter file has a ``Model.grid_name``, which names the
grid (``<grid-dir>/<grid_name>.hdf5``). Its parameters are axes, given as
``values`` entries in ``Axes``.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Model (template)
     - Keys
   * - Blackbody (``blackbody/generate_incident_grid_template.yaml``)
     - Axis ``log10blackbody_temperature``: log10(T/K).
   * - Broken power law (``broken_power_law/generate_incident_grid_template.yaml``)
     - ``Constants.edges``: the four wavelengths (Angstrom) bounding three
       power laws. Axes ``alpha1``, ``alpha2`` and ``alpha3``: the slope of
       each. All three are axes, and only ``alpha1`` may currently hold more
       than one value.
   * - Cloudy AGN (``cloudy_agn/generate_incident_grid_template.yaml``)
     - ``log10blue_bump_temperature``, ``alpha_ox``, and optional
       ``alpha_uv`` (default -0.5) and ``alpha_x`` (default -1.0). Each is an
       axis or, in ``Constants``, a fixed value.
   * - QSOSED (``qsosed/generate_incident_grid_template.yaml``)
     - Axes ``log10masses`` (Msun) and ``log10accretion_rates_eddington``.
       ``cosine_inclinations``: an axis, or a single value in ``Constants``
       for an isotropic grid.
   * - RELQSO (``relqso/generate_incident_grid_template.yaml``)
     - As QSOSED, plus axis ``spins`` in [0, 0.998].
   * - RELAGN (``relagn/generate_incident_grid_template.yaml``)
     - As QSOSED, plus exactly one of the axes ``spins`` or
       ``radiative_efficiencies`` (converted to spins).

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
   * - ``ionising_luminosity_threshold``
     - The lowest log10 specific ionising luminosity of HI (as in the
       incident grid's ``log10_specific_ionising_luminosity/HI``) run through
       Cloudy. Points below it get no Cloudy runs: their incident light is
       transmitted unchanged, with no nebular emission or lines, and the
       grid's ``below_threshold`` mask marks them. Blank or absent for none.

``cosmic_rays``, ``CMB``, ``constant_density`` and ``constant_pressure`` are
on/off options: set them to ``true`` to add their Cloudy command, and to
``false``, leave the value blank or remove the key to leave it out.

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
