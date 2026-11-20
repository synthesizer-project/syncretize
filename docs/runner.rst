Running a whole grid with the Runner
====================================

The Runner makes a grid from one config file: it makes the incident grid,
runs Cloudy on every model and assembles the final grid, either in one
process (``--local``) or as a chain of Slurm jobs. It does the same work as
the scripts of :doc:`incident_models` and :doc:`reprocessing`, which still
work on their own.

Quick start
-----------

1. Download the incident model's source data once, on a machine with
   internet access, by running the model's script with ``--download`` (see
   :doc:`incident_models`). A run never downloads.
2. Copy ``parameter_files/runs/run_template.yaml`` and edit it (see
   `The run config`_).
3. Submit the run:

   .. code-block:: console

      python -m syncretize.runner run run.yaml

   or run every stage here, which suits small grids and tests:

   .. code-block:: console

      python -m syncretize.runner run run.yaml --local

The final grid is written to ``Run.grid_dir``. Leave out the ``Cloudy``
section to make only the incident grid.

The run config
--------------

A run config has three sections:

``Run``
   Where things go (``work_dir`` for the run's state, Cloudy files, job
   scripts and logs; ``grid_dir`` for the grids; ``input_dir`` for the
   incident model's data), shell lines starting every job
   (``environment``), and the size of each kind of job.
``Incident``
   The incident model and its parameter file (``model`` and ``config``), or
   an existing incident grid (``grid``).
``Cloudy``
   The Cloudy config (a shipped name such as ``c23.01-sps`` or a path), an
   optional ``extra_config`` whose parameters replace the first's, the Cloudy
   build, and what the final grid stores.

``Cloudy.executable`` is the Cloudy build directory itself, holding
``source/cloudy.exe`` and ``data/`` (e.g. ``/cosma/local/cloudy/c25``). This
differs from the scripts' ``--cloudy-executable-path``, which is the
directory holding one directory per version. When the build directory is
named like a version (``c23.01``, ``c25``), its major version must match the
config's ``cloudy_version``.

Unknown keys, missing required keys and impossible values (e.g. a margin
longer than the workers' time limit) stop the run before anything is
submitted. Relative paths are relative to the run config's directory.

Every key, with its default:

.. literalinclude:: ../parameter_files/runs/run_template.yaml
   :language: yaml

How a run proceeds
------------------

Every stage is ``python -m syncretize.runner <stage> run.yaml``. On Slurm the
run is a chain of jobs, each submitting the next:

.. code-block:: text

   run (here)
    └─ start        incident grid, Cloudy inputs, first round     Run.incident
        └─ round 1  worker array (work) + finish job
            └─ round 2 ...  while models are left
                └─ assemble   the final grid                       Run.assemble

``start``
   Makes the incident grid (unless ``Incident.grid`` is given), writes the
   Cloudy inputs and puts every Cloudy model in the queue, then submits the
   first round.
``work``
   One worker of a round's job array. It runs ``Run.cloudy.cores_per_worker``
   Cloudy models at once, taking them from the queue until it is empty or the
   worker is within ``Run.cloudy.margin`` of its time limit.
``finish``
   Runs once every worker of the round has stopped, however they stopped. It
   puts failed and unfinished models back in the queue (see below) and
   submits the next round, or the assembly job when no models are left.
``assemble``
   Makes the final grid from the Cloudy outputs.

A round has ``Run.cloudy.workers`` workers, or fewer when fewer models are
left. Workers share the models through the queue, so a slow model holds up
one worker rather than a whole job, and the round ends when the models do.

With ``--local`` the same stages run in this process, with one worker and no
time limit.

Failures and time limits
------------------------

A Cloudy model counts as done when its ``.out`` file ends with ``Cloudy
exited OK`` and its ``.emergent_elin`` file holds at least 100 bytes.
Otherwise it has failed, and the finish job puts it back in the queue until it
has failed ``1 + Run.cloudy.max_retries`` times.

A model still running when its worker hits the time limit hasn't failed: the
finish job puts it back without counting an attempt, and the next round's
time limit is ``Run.cloudy.walltime_growth`` times longer, up to
``Run.cloudy.max_walltime``. Once the limit can't grow, a model stopped by it
counts as a failed attempt.

Models that failed every attempt are marked ``1`` in the final grid's
``failures`` dataset, with zeros in their spectra and lines, and listed in
``<work_dir>/cloudy/<grid name>.failures``. Check this mask before using or
publishing a grid.

The work directory
------------------

.. code-block:: text

   <work_dir>/
     state.yaml          the incident grid, Cloudy directory, round, time
                         limit and final grid
     queue/
       todo/<i>_<j>      waiting to run
       claimed/<i>_<j>@<worker>.<slot>
       done/<i>_<j>
       failed/<i>_<j>    holds how many times it failed
     cloudy/<grid name>/<i>/<j>.in, .out, ...
     jobs/<stage>[-<round>].sh
     logs/<run name>-<stage>-<job ID>.out

``<i>`` and ``<j>`` are a model's incident and photoionisation indices. To see
how a run is doing:

.. code-block:: console

   for state in todo claimed done failed; do
       echo $state $(ls <work_dir>/queue/$state | wc -l)
   done

Each worker's log has a line per model, and each finish job's log a summary of
its round.

Picking up a stopped run
------------------------

Each stage can be run by hand to restart the chain where it stopped, e.g. when
a finish job failed:

.. code-block:: console

   python -m syncretize.runner finish run.yaml

This tidies the queue and submits the next job. Setting the Cloudy inputs up
again keeps every model's state, so ``start`` can be repeated after a crash;
set ``Incident.grid`` to the incident grid first to avoid making it again.

The grid's run config
---------------------

The incident and final grids a run makes keep its config in a ``RunConfig``
group: the config with every default filled in as attributes, and the run file
itself in ``RunConfig/contents``, so

.. code-block:: python

   yaml.safe_load(hdf["RunConfig/contents"][()])

recovers it.
