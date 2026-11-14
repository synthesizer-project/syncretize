Included model conversions
==========================

These scripts convert model-specific data into the common :doc:`grid_format`.
Each model has its own directory, with its generator and any model-specific
YAML parameter files together. Each run makes one grid. The parameter file
describes that grid (the model variant, IMF, axes and so on), and command line
arguments say where to read and write files::

   python <script>.py <params>.yaml --grid-dir <grid-dir> \
       [--input-dir <input-dir>] [--download]

``--grid-dir`` is where the grid is written. File-backed conversions also need
``--input-dir``, and those that can fetch their source data accept
``--download``. Every script has an example parameter file whose comments
list the allowed values of each parameter; copy it to make other variants.
Scripts reject values that aren't allowed. :doc:`parameter_files` describes
the formats. The parameter file is stored in the
grid's ``ParameterFile`` group (see :doc:`grid_format`), so every grid records
how it was made.

Each script names its grid from the model and its parameters. Setting
``grid_name`` in the parameter file overrides this, and the grid is written
to ``<grid-dir>/<grid_name>.hdf5``. Use it to remake a published grid under
its published name.

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
     - Ready. Broken power-law or Chabrier IMF.
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
     - Ready with data. One alpha enhancement, or alpha enhancement as an axis.
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
     - Manual setup.
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

Generate an FSPS grid for the IMF set in the parameter file (a broken power
law or Chabrier IMF):

.. code-block:: console

   python src/syncretize/incident/sps/fsps/generate_incident_grid.py \
       src/syncretize/incident/sps/fsps/fsps.yaml --grid-dir grids

The result depends on the libraries compiled into the local FSPS installation.
The grid records them in its ``fsps_libraries`` model attribute. To insist on
particular ones, list them in the parameter file (``libraries``, or
``model.libraries`` for the variable-IMF script), e.g. ``[mist, miles]``; the
script stops if FSPS was compiled with others.

The variable-IMF conversion adds one slope or outer mass boundary as a genuine
grid axis:

.. code-block:: console

   python src/syncretize/incident/sps/fsps/generate_incident_grid_variable_imf.py \
       src/syncretize/incident/sps/fsps/configs/variable_high_mass_slope.yaml \
       --grid-dir grids

The provided configurations demonstrate explicit axis values and generated
ranges. The current FSPS implementation supports one variable IMF parameter at
a time and keeps internal mass breaks fixed at 0.5 and 1.0 solar masses.

BPASS
^^^^^

Place downloaded BPASS data beneath ``<input-dir>/bpass/``. The parameter
file chooses the BPASS model (IMF) and whether to use binary or single star
populations:

.. code-block:: console

   python src/syncretize/incident/sps/bpass/generate_incident_grid_2_2_1.py \
       src/syncretize/incident/sps/bpass/bpass-2.2.1.yaml \
       --grid-dir grids --input-dir model-data

BPASS 2.3 has five alpha enhancements. Set ``alpha_enhancement`` to one of
them for a grid at that value, or to ``axis`` for a grid with alpha
enhancement as a third axis.

BC03 and Maraston
^^^^^^^^^^^^^^^^^

``bc03/generate_incident_grid.py --download`` retrieves the original BC03
data. The 2016 conversion's ``--download`` also compiles a Fortran binary
reader and therefore needs ``make`` and a Fortran compiler.

Maraston source URLs used by older scripts are no longer available. Supply
source files through ``--input-dir``.

Other file-backed SPS conversions follow this command shape, with an
example parameter file next to each script:

.. code-block:: console

   python src/syncretize/incident/sps/<model>/generate_incident_grid*.py \
       src/syncretize/incident/sps/<model>/<model>.yaml \
       --grid-dir grids --input-dir model-data

``--download`` only works for BC03, BC03 2016, and Yggdrasil. Other scripts
need manually obtained files in the model-specific directory beneath
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
   * - cloudy AGN
     - ``cloudy_agn/generate_incident_grid.py``
     - Blue bump temperature, optional ``alpha_ox``, ``alpha_uv``, ``alpha_x``
     - Included YAML configuration
     - Self-contained. Reproduces cloudy's ``agn`` command; uses the
       black holes' ``blue_bump_temperature``.
   * - cloudy blackbody
     - ``blackbody/generate_incident_grid.py``
     - Blackbody temperature
     - Included YAML configuration
     - Self-contained. Reproduces cloudy's ``blackbody`` command;
       ``blackbody_temperature`` must be set by the user.

Run the self-contained broken-power-law model with:

.. code-block:: console

   python src/syncretize/incident/blackholes/broken_power_law/generate_incident_grid.py \
       src/syncretize/incident/blackholes/broken_power_law/bpl-feltre16.yaml \
       --grid-dir grids

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
       src/syncretize/incident/reduced_grid.yaml --grid-dir grids

The parameter file names the ``original_grid`` in ``--grid-dir`` and the
``ages`` (yr), ``max_age`` (yr), and/or ``metallicities`` to keep. Write
exponents with a decimal point and a sign (``1.0e+6``, not ``1e6`` or
``1.0e6``) or YAML reads them as strings.
