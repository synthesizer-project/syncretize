Getting started
===============

Syncretize turns emission-model data into standardised HDF5 grids that
Synthesizer can read. The source may be tables downloaded from a model
provider, output from a package such as FSPS, or arrays produced by custom
code. The resulting grid stores the spectra alongside their parameter axes,
units, and model metadata.

Most workflows begin with an **incident grid**: the emission produced directly
by the source model, before any photoionisation processing. Incident grids can
be used in Synthesizer as they are, or reprocessed to add transmitted and
nebular emission using the :doc:`reprocessing` machinery.

.. important::

   Many ready-to-use grids are already available through the `Syndex catalogue
   <https://synthesizer-project.org/syndex>`_. Check there first: a new grid is
   only needed when the required model, parameter coverage, or physical
   treatment is not already available. New grids can be
   :doc:`uploaded to Syndex <uploading_to_syndex>` so that others can reuse
   them.

Install Syncretize
------------------

Clone the repository and install it in the environment used to process the
model data:

.. code-block:: console

   git clone https://github.com/synthesizer-project/syncretize.git
   cd syncretize
   python -m pip install .

Some models require separately licensed data, compiled software, or additional
Python packages. See :doc:`incident_models` for model-specific requirements.

Create an incident grid
-----------------------

Start from an included model conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Emission libraries use different file layouts, units, and conventions.
Syncretize therefore includes model-specific conversion scripts that read the
native model output, arrange it into ordered arrays, attach metadata, and write
the standard grid format.

For example, an existing FSPS installation can be converted with:

.. code-block:: console

   mkdir -p grids
   python src/syncretize/incident/sps/install_fsps.py \
       --grid-dir grids

The script writes one or more ``.hdf5`` files to ``grids``. Other models may
also need ``--input-dir`` to locate downloaded source data. See
:doc:`incident_models` for available conversions, required inputs, and known
limitations.

Start from your own model
^^^^^^^^^^^^^^^^^^^^^^^^^

If no conversion exists for your use case, you can use ``GridFile`` directly
to create a grid and write a model-specific generation script. Arrays
containing the model parameters, wavelength sampling, and spectra are passed to
``GridFile``, which orchestrates the writing of the standard HDF5 structure and metadata.

Read :doc:`grid_format` first, then follow the complete
:doc:`making_your_own_grid` example. The format does not restrict grids to age
and metallicity: a model may define any number of ordered axes.

Contributions of new model-conversion scripts are welcome in the `Syncretize
repository <https://github.com/synthesizer-project/syncretize>`_. Adding the
conversion alongside the finished grid gives future users a reproducible way
to build on the method and explore different parameter choices.

Check and use the grid
----------------------

Load the finished file with Synthesizer and inspect its axes and spectra:

.. code-block:: python

   from synthesizer.grid import Grid

   grid = Grid("my-grid", grid_dir="grids", ignore_lines=True)
   print(grid.axes)
   print(grid.available_spectra)

Before sharing the file, run the Syndex structural checker. Syndex requires
Python 3.10 or later and may be installed in a separate environment:

.. code-block:: console

   python -m pip install cosmos-syndex
   syndex-check grids/my-grid.hdf5

Add photoionisation processing
------------------------------

If the application needs gas reprocessing, use the incident grid as input to
the :doc:`reprocessing` workflow. Syncretize creates Cloudy inputs for every
combination of incident and photoionisation parameters, runs those models, and
collects their spectra and lines into a new grid.
