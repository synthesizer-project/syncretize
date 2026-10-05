Syncretize
==========

Syncretize provides the machinery to produce HDF5 grid files using the standardised `Synthesizer <https://github.com/synthesizer-project/synthesizer>`_ format. 

As well as providing scripts to generate AGN and stellar grids from a varied and ever growing list of models, Syncretize also provides a framework for users to reprocess any of these grids through photoionisation codes such as CLOUDY (with plans to include interfaces to more in the future).

Syncretize provides a common interface between emission libraries and forward
or inverse modelling pipelines without forcing every model into the same
parameter space. Grids may therefore define any number of ordered axes, with
metadata describing how Synthesizer should interpret them.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   getting_started
   grid_format
   making_your_own_grid

.. toctree::
   :maxdepth: 2
   :caption: Workflows

   incident_models
   reprocessing
   uploading_to_syndex

.. toctree::
   :maxdepth: 2
   :caption: Reference

   grid_file
