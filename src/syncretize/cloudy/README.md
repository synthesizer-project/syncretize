## Cloudy photoionisation modelling pipeline

This set of scripts facilitates the creation of new sets of grids with photoionisation modelling. This adds additional spectra (transmission, nebular, nebular_continuum) and adds line quantities (luminosities and continuum values).

There are three steps to the process:

- First, we need to create the input files for `cloudy` from an incident grid.
- Next, we use `run_cloudy.py` to run `cloudy`.
- Finally, `create_synthesizer_grid.py` gathers the `cloudy` outputs and produces a new grid.

The details of each of these steps are described below.

### Creating the cloudy input grid

The first step in the process is the creation of a grid of `cloudy` input files, including the incident spectra, the configuration file, and the list of lines to save. This uses an **incident grid** (for example from an SPS model as above). `cloudy`'s built-in `agn` and `blackbody` shapes are also available as incident grids, made by `incident/blackholes/cloudy_agn/generate_incident_grid.py` and `incident/blackholes/blackbody/generate_incident_grid.py`.

#### Using an incident grid

To use an incident grid we can run `create_cloudy_input_grid.py` providing the incident grid name, grid directory, output directory, `cloudy` parameter file(s), a machine, and the path to the `cloudy` executable as follows:

```
create_cloudy_input_grid.py \
    --incident-grid=test \
    --grid-dir=/path/to/grids \
    --cloudy-output-dir=/path/to/cloudy/outputs \
    --cloudy-paramfile=c23.01-sps \
    --cloudy-paramfile-extra=test_suite/reference_ionisation_parameter \
    --machine=machine \
    --cloudy-executable-path=/path/to/cloudy/executable
```

##### Parameter files

An integral part of this process are the provision of a parameter file(s) which contains the photoionisation parameters. These can either be single values or lists (arrays). When a quantity is an array this adds an additional axis to the reprocessed grid. A range of ready-made parameter files are available for a range of scnearios in the `params/` directory.

It is possible to provide two parameter files, for example a default set of parameters and then a file containing a limited set of parameters to be changed. This approach is used to build a *photoionisation test suite*; this is a series of parameter files used to generate grids where we systematically vary a photoionisation parameter (e.g. the hydrogen density or ionisation parameter) or flag. 

##### Gas metallicity

By default the gas has the metallicity of the incident grid point. Setting `gas_metallicity` (a metal mass fraction) in the parameter file decouples the two; like any other parameter it can be a list, which adds a `gas_metallicities` axis to the reprocessed grid. A `gas_metallicity` of zero (or a metal free incident grid point with no `gas_metallicity`) gives primordial gas: every metal is turned off in `cloudy`, so `depletion_model` and `grains` must be empty and the line list must only hold hydrogen and helium lines (`linelist-primordial.dat`), since `cloudy` stops on lines of elements that are off.

##### Pop III stars

`c25.00-popIII-primordial` and `c25.00-popIII-enriched` reprocess Pop III incident grids (e.g. the pure stellar grids of the three Yggdrasil populations, made with `incident/sps/yggdrasil/generate_incident_grid.py`) through metal free gas and through marginally enriched gas (10^-4 to 10^-2 Zsun, alpha enhanced) respectively, over a range of ionisation parameters and densities. `popIII/carbon_enhanced` and `popIII/high_z` can be added with `--cloudy-paramfile-extra` to use a carbon rich abundance pattern or include the CMB at z = 10:

```
python create_cloudy_input_grid.py \
    --incident-grid=yggdrasil-1.3.3-PopIII_salpeter-50,500 \
    --grid-dir=/path/to/grids \
    --cloudy-output-dir=/path/to/cloudy/outputs \
    --cloudy-paramfile=c25.00-popIII-enriched \
    --cloudy-paramfile-extra=popIII/carbon_enhanced \
    --machine=cosma7 \
    --cloudy-executable-path=/cosma/local/cloudy/c25/bin
```

##### Machines

If `--machine` is specified, and it is one that is recognised (e.g. Sussex's artemis system), then a job submission script will be produced. The will be an array job with each job being a single incident grid point, i.e. each jobs runs all the photoionisation models. In most cases this will be a handful of models, but for comprehensive grids this could be hundreds or even thousands of cloudy models that will take sometime to run.

For `--machine=cosma7` the script is instead written to the grid's `cloudy` output directory (`cosma7.slurm`) and runs `cloudy` itself, so the next step is just `sbatch cosma7.slurm`. `--cloudy-executable-path` is then the directory holding `cloudy.exe`, with `cloudy`'s `data` directory alongside it. Each array task fills a node with the models of several incident grid points and skips any that have already finished, so resubmitting the script reruns only the models that failed or timed out.

### Run `cloudy`

Next, we can use the `run_cloudy.py` to automatically run either a single model, all models, or all models for a given photoionisation grid point (the suggested behaviour for coupling with a HPC array job). This behaviour depends on the choice of --incident-index and --photoionisation-index.
Setting both will run a single model, setting only `--incident-index` will run all models at a particularly incident grid point, while setting neither will result in all models being run in serial (not recommended except for tiny grids).

```
python run_cloudy.py \
    --grid-name=grid_name \
    --cloudy-output-dir=/path/to/cloudy/outputs \
    --cloudy-executable-path=/path/to/cloudy/executable \
    --incident-index=0
    --photoionisation-index=0
```

### Create synthesizer grid

Finally, we need to create a new `synthesizer` grid object containing the cloudy computed lines and spectra. This is acheived by running `create_synthesizer_grid.py`. By default spectra are not extracted; if this is desired it necessary to add the `include_spectra` argument as below.

```
python create_synthesizer_grid.py \
  --incident-grid=incident_grid_name \
  --grid-dir=/path/to/grids \
  --cloudy-output-dir=/path/to/cloudy/outputs \
  --cloudy-paramfile=c23.01-sps \
  --cloudy-paramfile-extra=test_suite/reference_ionisation_parameter
  --include_spectra \
```

