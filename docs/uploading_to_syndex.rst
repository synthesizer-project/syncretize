Uploading to Syndex
===================

`Syndex <https://synthesizer-project.org/syndex>`_ is the catalogue and
download service for Synthesizer data. Submit a grid there once it is complete,
documented, and usable with Synthesizer.

Before submission, complete the checks in :doc:`making_your_own_grid` and
confirm the file follows :doc:`grid_format`.

Check the grid
--------------

Install the contributor tooling under Python 3.10 or later and run the same
checker used by the submission service. A separate environment is fine:

.. code-block:: console

   pip install cosmos-syndex
   syndex-check my-grid.hdf5

The checker reports extracted metadata, errors that prevent publication, and
non-blocking warnings. Common grid requirements are:

* An ``axes`` root attribute and corresponding ``axes/`` datasets.
* ``Model.sps_name`` for an SPS grid, or ``Model.type = "agn"`` for an AGN grid.
* Enough spectral or photoionisation metadata to identify the emission type.
* Plural axis names with appropriate units.
* Model name, wavelength coverage, and Synthesizer version metadata.

Submit through the portal
-------------------------

Open the `Syndex submission portal
<https://synthesizer-project.org/syndex/submit>`_. Sign in with GitHub and ask
a Syndex maintainer for access if needed. Describe the dataset, including its
model, version, provenance, licence, citations, and whether it is intended for
scientific use or testing.

Files up to 10 GB can be uploaded in the browser.

Upload a large file
-------------------

For a file larger than 10 GB, or when working on a remote machine without a
browser, copy the submission token shown by the portal and run:

.. code-block:: console

   syndex-submit SUBMISSION_TOKEN my-grid.hdf5

The command authenticates through GitHub's device flow. Multipart uploads can
resume after interruption, so rerun the same command if the transfer stops.

Publication is reviewed rather than immediate. See the `Syndex documentation
<https://synthesizer-project.org/syndex/docs/>`_ for current catalogue and
submission requirements.
