Included model conversions
==========================

These scripts convert model-specific data into the common :doc:`grid_format`.
Each model has its own directory, with its generator and any model-specific
configuration files together. All write into ``--grid-dir``. File-backed
conversions also need ``--input-dir`` even though the shared command-line
parser marks it as optional.

SPS models
----------

.. list-table::
   :header-rows: 1
   :widths: 16 18 21 18 27

   * - Model
     - Script
     - Axes
     - Source requirement
     - Status and special information
   * - FSPS
     - ``fsps/generate_incident_grid.py``
     - Ages, metallicities
     - Working python-FSPS/FSPS installation
     - Ready. Default BPL IMF; optional Chabrier and IMF variants.
   * - FSPS variable IMF
     - ``fsps/generate_incident_grid_variable_imf.py``
     - Ages, metallicities, optional IMF parameter
     - Working python-FSPS/FSPS installation
     - Ready. YAML defines one optional IMF axis.
   * - BPASS 2.2.1
     - ``bpass/generate_incident_grid_2_2_1.py``
     - Ages, metallicities
     - Manual BPASS download
     - Ready with data. Binary/single-star variants and multiple IMFs.
   * - BPASS 2.3
     - ``bpass/generate_incident_grid_2_3.py``
     - Ages, metallicities, alpha enhancement
     - Manual BPASS download
     - Constrained. Writes both separate and combined grids regardless of flags.
   * - BC03
     - ``bc03/generate_incident_grid.py``
     - Ages, metallicities
     - Download supported
     - Ready. Padova 2000, Chabrier IMF.
   * - BC03 2016
     - ``bc03/generate_incident_grid_2016.py``
     - Ages, metallicities
     - Download plus Fortran compiler
     - Setup required. Nine atmosphere/IMF combinations.
   * - Maraston 2005
     - ``maraston/generate_incident_grid_2005.py``
     - Ages, metallicities
     - Manual source data
     - Manual setup. Original download URL is unavailable.
   * - Maraston 2011
     - ``maraston/generate_incident_grid_2011.py``
     - Ages, metallicities
     - Manual source data
     - Manual setup. Several libraries/IMFs; solar metallicity only.
   * - Maraston 2013
     - ``maraston/generate_incident_grid_2013.py``
     - Ages, metallicities
     - Manual source data
     - Blocked by known age-unit bug; do not use until fixed.
   * - Maraston 2024
     - ``maraston/generate_incident_grid_2024.py``
     - Ages, metallicities
     - Data obtained from model authors
     - Manual setup. Rotation, temperature-correction, and IMF variants.
   * - Yggdrasil Pop III
     - ``yggdrasil/generate_incident_grid.py``
     - Ages, metallicities
     - Download supported
     - Caveats. Includes model-provided nebular products; incident products are best
       for consistent reprocessing.

FSPS
^^^^

Generate the default FSPS grid and optional variants:

.. code-block:: console

   python src/syncretize/incident/sps/fsps/generate_incident_grid.py \
       --grid-dir grids \
       --include-chabrier \
       --include-imf-variants

The result depends on the libraries compiled into the local FSPS installation.
Record that environment with the generated file.

The variable-IMF conversion adds one slope or outer mass boundary as a genuine
grid axis:

.. code-block:: console

   python src/syncretize/incident/sps/fsps/generate_incident_grid_variable_imf.py \
       --grid-dir grids \
       --config-file \
       src/syncretize/incident/sps/fsps/configs/variable_high_mass_slope.yaml

The provided configurations demonstrate explicit axis values and generated
ranges. The current FSPS implementation supports one variable IMF parameter at
a time and keeps internal mass breaks fixed at 0.5 and 1.0 solar masses.

BPASS
^^^^^

Place downloaded BPASS data beneath ``<input-dir>/bpass/``. Generate all
supported BPASS 2.2.1 IMF variants with:

.. code-block:: console

   python src/syncretize/incident/sps/bpass/generate_incident_grid_2_2_1.py \
       --input-dir model-data \
       --grid-dir grids \
       --models all

BPASS 2.3 adds alpha enhancement as a third axis in its combined grid. The
current script writes both individual and combined outputs regardless of the
``--individual`` and ``--full`` flags.

BC03 and Maraston
^^^^^^^^^^^^^^^^^

``bc03/generate_incident_grid.py --download`` retrieves the original BC03 data. The 2016
conversion also compiles a Fortran binary reader and therefore needs ``make``
and a Fortran compiler.

Maraston source URLs used by older scripts are no longer available. Supply
source files through ``--input-dir``. Treat the Maraston 2013 conversion as
unsupported until its age-unit bug is corrected.

Other file-backed SPS conversions follow this command shape:

.. code-block:: console

   python src/syncretize/incident/sps/<model>/generate_incident_grid*.py \
       --input-dir model-data \
       --grid-dir grids

Add ``--download`` only for BC03, BC03 2016, or Yggdrasil. Other scripts need
manually obtained files in the model-specific directory beneath
``--input-dir``.

Black-hole and AGN models
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 20 21 19 22

   * - Model
     - Script
     - Axes
     - Requirement
     - Status and special information
   * - Broken power law
     - ``broken_power_law/generate_incident_grid.py``
     - Three spectral slopes
     - Included YAML configuration
     - Limited. Supplied Feltre configuration works; multiple values on its
       second or third slope do not.
   * - RELAGN
     - ``relagn/generate_incident_grid.py``
     - Mass, accretion rate, spin/efficiency, optional inclination
     - RELAGN and XSPEC
     - External setup. Parallel generation through ``--num-procs``.
   * - RELQSO
     - ``relqso/generate_incident_grid.py``
     - Mass, accretion rate, spin, optional inclination
     - RELAGN and XSPEC
     - External setup. Simplified relativistic model.
   * - QSOSED
     - ``qsosed/generate_incident_grid.py``
     - Mass, accretion rate, optional inclination
     - RELAGN and XSPEC
     - External setup. Isotropic configs require NumPy 2.

Run the self-contained broken-power-law model with:

.. code-block:: console

   python src/syncretize/incident/blackholes/broken_power_law/generate_incident_grid.py \
       --grid-dir grids \
       --config-file \
       src/syncretize/incident/blackholes/broken_power_law/bpl-feltre16.yaml

The supplied configuration varies only its first slope. The current
implementation does not correctly support multiple values on the second or
third slope axes.

RELAGN, RELQSO, and QSOSED require an external RELAGN checkout and a working
XSPEC Python environment. Their supplied YAML files are under
model directories. RELQSO and QSOSED currently resolve RELAGN from
a working-directory-relative path, so run those scripts from their own
directory or make RELAGN importable through ``PYTHONPATH``.

Reduced grids
-------------

``incident/create_reduced_grid.py`` samples an existing incident grid at a
smaller set of ages and metallicities. It is useful for tests and exploratory
Cloudy runs, not a separate physical model.

.. code-block:: console

   python src/syncretize/incident/create_reduced_grid.py \
       --grid-dir grids \
       --original-grid my-full-grid \
       --ages 1e6 1e7 1e8 \
       --metallicities 0.001 0.01 0.02
