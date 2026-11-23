Running a whole grid with the Runner
====================================

The Runner makes grids from one config file: it makes the incident grid, runs
Cloudy on every model, once for each Cloudy set-up the config gives, and
assembles a final grid for each, either in one process (``--local``) or as a
chain of Slurm jobs. It does the same work as
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

The final grids are written to ``Run.grid_dir``. Leave out the ``Cloudy``
section to make only the incident grid. ``syncretize`` takes the same
commands, e.g. ``syncretize run run.yaml``.

The run config
--------------

A run config has these sections:

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
``Cloudy2``, ``Cloudy3``, ...
   More Cloudy set-ups for the same incident grid, each like ``Cloudy`` (which
   can also be written ``Cloudy1``). Each section makes its own grid, named
   after its configs (``<incident grid>_cloudy-<config>[-<extra_config>]``),
   so no two sections may have the same configs.

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
    └─ start        incident grid; each section's Cloudy inputs   Run.incident
        │           and first round
        ├─ section C1
        │   └─ round 1  worker array (work) + finish job
        │       └─ round 2 ...  while models are left
        │           └─ assemble   C1's final grid                  Run.assemble
        └─ section C2 ...        at the same time

``start``
   Makes the incident grid (unless ``Incident.grid`` is given), then, for
   each Cloudy section, writes its Cloudy inputs, puts every Cloudy model in
   its queue and submits its first round.
``work``
   One worker of a round's job array. It runs ``Run.cloudy.cores_per_worker``
   Cloudy models at once, taking them from the queue until it is empty or the
   worker is within ``Run.cloudy.margin`` of its time limit (given with
   ``--walltime``, which the round's jobs pass it).
``finish``
   Runs once every worker of the round has stopped, however they stopped. It
   puts failed and unfinished models back in the queue (see below) and
   submits the next round, or the assembly job when no models are left. Its
   time limit is a fixed hour, as it only tidies the queue.
``assemble``
   Makes the section's final grid from its Cloudy outputs. The run is done
   once every section's is.
``incident``
   Only makes the incident grid, here (no Slurm, and not into ``run.log``),
   e.g. to try an incident model before a run.

The ``work``, ``finish`` and ``assemble`` stages each belong to one Cloudy
section, named by ``--section <n>`` (1 for ``Cloudy``); the sections go their
own ways at once, so a slow or failing section holds up no other.

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
round, until it has failed ``1 + Run.cloudy.max_retries`` times; then it has
failed. A model Cloudy couldn't even be started for (say, its directory is
missing) fails the same way; a worker whose Cloudy build is missing stops
before claiming any.

A worker stops claiming models ``Run.cloudy.margin`` before its time limit,
or earlier once the time left is under 1.5 times its longest Cloudy run so
far (so it doesn't start models that likely won't finish; models claimed
together with ``Run.cloudy.claim`` go back unrun then too), and stops the
Cloudy runs still going 2 minutes before its time limit. A model stopped
like this hasn't failed: it goes back in the queue without counting an
attempt, and the next round's time limit is ``Run.cloudy.walltime_growth``
times longer, up to ``Run.cloudy.max_walltime`` (lowering
``max_walltime`` and resuming lowers a limit already grown past it). A model
whose worker died without stopping it (lost with its job, e.g. a node
failure) is treated the same way by the finish job. Running out of time is
never a failure: it means the run's time limits are too short. So once the
limit can't grow and models still run out of time, the section stops
(``next: walltime`` in its ``state.yaml``, shown in red by watch) with those
models back in the queue; raise ``Run.cloudy.max_walltime`` and resume the
run to carry on with a longer limit. The finish job starts as
the round's workers end, but Slurm's accounting can lag behind, so it asks
``sacct`` again for up to about 7 minutes before taking a worker as still
running.

Models that failed every attempt (and couldn't be interpolated, see below)
are marked ``1`` in the final grid's ``failures`` dataset, with NaN in
their spectra and lines (they're most likely where the physics doesn't hold,
so there's no honest value to give them), listed with why they last failed under
``failures`` in ``state.yaml``, and listed in
``<work_dir>/<section>/cloudy/<grid name>.failures``. Check this mask before using or
publishing a grid: interpolating over a grid with NaN points gives NaN for
anything that touches them.

Interpolating failed models
---------------------------

A model that fails every attempt may fail only because of its exact inputs.
With ``Cloudy.perturb`` given (``perturb: true`` takes its defaults; it is off
unless given), once every model has run the finish job runs each such model again slightly
either side of its parameters, along every Cloudy axis of the grid (or along
its ionisation parameter when the grid has no Cloudy axes), and assembly
interpolates each pair of successful runs back to the model's own parameters
(linearly, in log for log-spaced axes such as U and n_H), averaging over the
axes. The two sides are ``Cloudy.perturb.step`` (default 0.1) of the gap to
the neighbouring grid value on that side, or ``log_step`` (default 0.05) dex
for the ionisation parameter; a side that fails is tried again twice as far
out, up to ``max_steps`` times in all (default 3). Only pairs with
both sides run are used: a model with none stays failed. A model whose
every neighbour on the grid of Cloudy axes failed too isn't perturbed at
all (it's most likely where Cloudy can't run), and stays failed, unless a
round of perturbed runs makes one of its neighbours interpolable, when it is
perturbed after all. Interpolated models
are marked ``1`` in the final grid's ``interpolated`` dataset, and the plan
and every run are kept in ``perturb.json`` beside the Cloudy inputs. These
rounds are the section's perturbation phase: their workers are named
``R<n>-<name>-C<k>-PerturbCloudy``, and run.log, the rounds table and
``syncretize watch`` mark them "perturb".

Incident points too faint to ionise
-----------------------------------

Setting ``ionising_luminosity_threshold`` in the Cloudy config (the lowest
log10 specific ionising luminosity of HI to run, e.g. 37) skips the incident
points below it: Cloudy has nothing to do there and can fail. Their incident
light is transmitted unchanged, with no nebular emission or lines, and the
final grid's ``below_threshold`` dataset marks them.

The work directory
------------------

.. code-block:: text

   <work_dir>/
     run.log             the run's story (see below)
     state.yaml          the run's state: the next stage (start, cloudy
                         while its sections run, or done), the start job,
                         the incident grid, its sections and final grids
     jobs/<job name>.sh
     logs/<job name>-<job ID>.out       each job's own log
     logs/<job name>-<job ID>.details   what the incident model and Cloudy
                         scripts printed
     <section>/          each Cloudy section's own, named after its configs
       state.yaml        its next stage and job, Cloudy directory and final
                         grid, every job (submitted, started, ended,
                         outcome), the round and its time limit, every
                         round's totals, whether it is perturbing, and the
                         models that failed (and, once assembled, those
                         interpolated)
       models.json       every Cloudy model's state and attempts (worker,
                         result, why it failed, start time, run time)
       queue/            (perturbed runs are <i>_<j>p<a>±<k>, see above)
         todo/<i>_<j>    waiting to run
         claimed/<i>_<j>@<worker>.<slot>
         done/<i>_<j>
         failed/<i>_<j>  holds how many times it failed
       state/<worker>.jsonl  each worker's records, until a finish job
                         gathers them into models.json
       cloudy/<grid name>/<i>/<j>.in, .out, ...
       cloudy/<grid name>/grid_points.json   each model's grid point
       cloudy/<grid name>/perturb.json       the perturbation plan, if any

``<i>`` and ``<j>`` are a model's incident and photoionisation indices.
Jobs are named ``R<round>-<run name>[-C<k>]-<kind>``, so they sort by round
in ``squeue``, ``jobs/`` and ``logs/``: ``R0-<name>-Incident`` (the start
job), then for Cloudy section k ``R<n>-<name>-C<k>-Cloudy`` (round n's
workers, whose logs add ``_<task>``), ``R<n>-<name>-C<k>-PerturbCloudy`` (the
workers running failed models' perturbed runs, see above),
``R<n>-<name>-C<k>-Runner`` (round n's finish job) and
``R<n>-<name>-C<k>-Assemble``. A section's ``state.yaml`` and
``models.json`` are written only by its start, finish and assembly jobs,
which follow each other, and each is replaced whole; its workers, which run
at once, each write their own file in ``state/``.

Logs
----

``<work_dir>/run.log`` tells the whole run's story, one dated line at a time,
each saying which stage and job wrote it. It starts with a banner, the
syncretize and synthesizer versions, the run config with every default, the
incident model's parameters and the Cloudy parameters (fixed, and varied with
the number of photoionisation models) of each Cloudy section. Each stage
then adds what it did: the
incident grid and Cloudy inputs made, each round's workers and time limit, a
table of the rounds so far after each round (done; errored, the runs that
failed, hit the time limit or were lost with their job; timed out, those that
hit the time limit; retried, those put back to run again; failed, those given
up on; and left), Cloudy run times, the models that failed with why, and
finally the grid and the run's totals. A job that fails says so, with the
``resume`` command to carry on.

Only the start, finish and assembly jobs (and ``run`` and ``resume``) write to
``run.log``. A line about one Cloudy section starts its tag with the
section's label, e.g. ``[C2 finish 12379722]``; the sections' jobs may write
at once, a line at a time. Each job also has its own log in
``logs/``, with each line prefixed by the time since the job started: the
workers' logs have a line per model, with why it failed. What the incident
model and Cloudy scripts print goes to a ``.details`` file beside it instead,
and any warnings in it are repeated in the logs.

Watching a run
--------------

.. code-block:: console

   syncretize watch [run.yaml | work_dir | directory]

shows a live view of a run that fills the terminal, refreshed every 2 seconds
(``-n`` to change). It shows one grid (one Cloudy section) at a time: a card
with the grid's run, name, stage and progress (with the rate and an ETA),
then its queue, its current round (workers running and queued, models
finished, how many errored, timed out, were retried and failed, run
times), its jobs and how they ended,
every round so far, the ``run.log`` lines about it and the whole run, the
models that failed and the models running longest (with their grid points
and Cloudy iterations; those slow for the round in yellow, to spot one that
hangs). The title bar says which grid is shown, e.g. ``grid 2 of 4``.

``Tab`` moves between the panels with more to show (or click one), and
``Enter`` shows more: a job's details (its entry in the state file, its
``run.log`` lines, the end of its Slurm output, its tasks from ``sacct`` and
its script), a failed or running model's (its grid point, every attempt with
why it ended, and the end of its Cloudy output), or ``run.log`` full screen.
The details open over the page; ``Esc`` or ``Enter`` closes them.

Press ``g`` for the all grids page: progress across every grid, then a row
per grid (its stage, progress, models done, running and failed, its run and
its name); ``Enter`` shows the grid picked (``g`` or ``Esc`` goes back).
``o`` sorts the rows by the next column (counts most first, names A to Z),
``O`` reverses the sort, and clicking a column's title sorts by it. Number
keys ``1``-``9`` and ``0`` show the grid with that number (``#``). Every
panel scrolls with the mouse, and ``run.log`` follows new lines unless
scrolled up. Press ``l``, ``r`` or ``f`` to show ``run.log``, the running models
or the failed models full screen (again to go back), ``s`` for your Slurm
queue, refreshed every 10 seconds (``Enter`` shows a job's details, ``s`` or
``Esc`` goes back), and ``q`` to quit; ``--once`` prints a
summary once. With no argument it watches the run in the current directory.
It only reads the run's files (and asks Slurm about its jobs every 30
seconds), so it can be left running anywhere.

Given a directory holding runs instead (any directory with runs' ``run.yaml``
files in it or below, e.g. a campaign's ``runs/``), it watches every grid of
every run under way there, and the all grids page counts the runs under way,
done and not yet submitted. Runs appear once submitted and drop off once done
(the directory is searched every 30 seconds); only the shown grid's run is
read in full, the others' state files and queues every 10 seconds.
``--once`` prints a summary of each run under way.

Without ``syncretize watch``, to see how a round is doing while its workers
run:

.. code-block:: console

   for state in todo claimed done failed; do
       echo $state $(ls <work_dir>/<section>/queue/$state | wc -l)
   done

Picking up a stopped run
------------------------

When a run stops part way (a job failed, the chain was cancelled, the machine
went down), carry it on with

.. code-block:: console

   python -m syncretize.runner resume run.yaml

which submits the start job again if the run hadn't got past it, or else
each unfinished section's next stage (``next`` in its ``state.yaml``): the
finish job of its current round (after its workers, if they are still
running), or its assembly job; a section whose models ran out of time at
``max_walltime`` gets a round with a longer limit, once it's been raised; a
round whose job died before submitting its
workers has them submitted now, and a run whose sections are all done is
marked done. It does nothing for a job still queued or running, or when the
run is done. With ``--local`` it runs the
remaining stages here. Every stage keeps what earlier attempts did: the
incident grid isn't made again, setting up the Cloudy inputs again keeps each
model's state, and the finish job puts back the models of workers that died.
A finish job keeps the workers' records until it has saved the round, so one
that dies part way counts the round in full when run again.

``run`` refuses a work directory that already holds a run; use ``resume``.

The grid's run config
---------------------

The incident and final grids a run makes keep its config in a ``RunConfig``
group: the config with every default filled in as attributes (a final grid's
``Cloudy`` is the section that made it), and the run file itself in
``RunConfig/contents``, so

.. code-block:: python

   yaml.safe_load(hdf["RunConfig/contents"][()])

recovers it.
