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
exited OK`` and it wrote ``.emergent_elin`` (and ``.intrinsic_elin`` when the
grid stores intrinsic lines, and ``.refl`` for open geometries with spectra),
each at least 100 bytes. The workers and the assembly use this same check.
A model that fails goes straight back in the queue, to run again in the same
round, until it has failed ``1 + Run.cloudy.max_retries`` times; then it is
given up.

A worker stops claiming models ``Run.cloudy.margin`` before its time limit,
or earlier once the time left is under 1.5 times its longest Cloudy run so
far (so it doesn't start models that likely won't finish), and stops the
Cloudy runs still going 2 minutes before its time limit. A model stopped
like this hasn't failed: it goes back in the queue without counting an
attempt, and the next round's time limit is ``Run.cloudy.walltime_growth``
times longer, up to ``Run.cloudy.max_walltime``. Once the limit can't grow, a
model stopped by it counts as a failed attempt. A model whose worker died
without stopping it (lost with its job, e.g. a node failure) is treated the
same way by the finish job.

Models given up on are marked ``1`` in the final grid's ``failures`` dataset,
with zeros in their spectra and lines, listed with why they last failed under
``failures`` in ``state.yaml``, and listed in
``<work_dir>/cloudy/<grid name>.failures``. Check this mask before using or
publishing a grid.

The work directory
------------------

.. code-block:: text

   <work_dir>/
     run.log             the run's story (see below)
     state.yaml          the run's state: the next stage and its job, the
                         incident grid, Cloudy directory and final grid,
                         every job (submitted, started, ended, outcome),
                         every round's totals, and the models given up on
     models.json         every Cloudy model's state and attempts (worker,
                         result, why it failed, start time, run time)
     queue/
       todo/<i>_<j>      waiting to run
       claimed/<i>_<j>@<worker>.<slot>
       done/<i>_<j>
       failed/<i>_<j>    holds how many times it failed
     state/<worker>.jsonl  each worker's records, until a finish job
                         gathers them into models.json
     cloudy/<grid name>/<i>/<j>.in, .out, ...
     jobs/<job name>.sh
     logs/<job name>-<job ID>.out       each job's own log
     logs/<job name>-<job ID>.details   what the incident model and Cloudy
                         scripts printed

``<i>`` and ``<j>`` are a model's incident and photoionisation indices.
Jobs are named ``R<round>-<run name>-<kind>``, so they sort by round in
``squeue``, ``jobs/`` and ``logs/``: ``R0-<name>-Incident`` (the start job),
``R<n>-<name>-Cloudy`` (round n's workers, whose logs add ``_<task>``),
``R<n>-<name>-Runner`` (round n's finish job) and ``R<n>-<name>-Assemble``.
``state.yaml`` and ``models.json`` are written only by the start, finish and
assembly jobs, which never run at once, and each is replaced whole; the
workers, which do run at once, each write their own file in ``state/``.

Logs
----

``<work_dir>/run.log`` tells the whole run's story, one dated line at a time,
each saying which stage and job wrote it. It starts with a banner, the
syncretize and synthesizer versions, the run config with every default, the
incident model's parameters and the Cloudy parameters (fixed, and varied with
the number of photoionisation models). Each stage then adds what it did: the
incident grid and Cloudy inputs made, each round's workers and time limit, a
table of the rounds so far after each round (done, failed, retried, stopped,
lost, given up, left), Cloudy run times, the models given up on with why, and
finally the grid and the run's totals. A job that fails says so, with the
``resume`` command to carry on.

Only the start, finish and assembly jobs (and ``run`` and ``resume``) write to
``run.log``; they never run at once. Each job also has its own log in
``logs/``, with each line prefixed by the time since the job started: the
workers' logs have a line per model, with why it failed. What the incident
model and Cloudy scripts print goes to a ``.details`` file beside it instead,
and any warnings in it are repeated in the logs.

To see how a round is doing while its workers run:

.. code-block:: console

   for state in todo claimed done failed; do
       echo $state $(ls <work_dir>/queue/$state | wc -l)
   done

Picking up a stopped run
------------------------

When a run stops part way (a job failed, the chain was cancelled, the machine
went down), carry it on with

.. code-block:: console

   python -m syncretize.runner resume run.yaml

which submits the stage ``next`` in ``state.yaml`` names: the start job, the
finish job of the current round (after its workers, if they are still
running), or the assembly job. It does nothing while that stage's job is still
queued or running, or when the run is done. With ``--local`` it runs the
remaining stages here. Every stage keeps what earlier attempts did: the
incident grid isn't made again, setting up the Cloudy inputs again keeps each
model's state, and the finish job puts back the models of workers that died.

``run`` refuses a work directory that already holds a run; use ``resume``.

The grid's run config
---------------------

The incident and final grids a run makes keep its config in a ``RunConfig``
group: the config with every default filled in as attributes, and the run file
itself in ``RunConfig/contents``, so

.. code-block:: python

   yaml.safe_load(hdf["RunConfig/contents"][()])

recovers it.
