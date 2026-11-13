
# Linelist generation

In the synthesizer photoionisation modelling it is necessary to provide cloudy with a list of lines which we're interested in. This tool generates two lists from a set of reference `cloudy` runs, covering both stars and AGN and a range of metallicities:

- the **standard list** (`linelist-standard.dat`, and `linelist-vacuum.dat` for vacuum wavelengths), our default. It holds the lines with luminosities above some threshold relative to a reference line in at least one model. The default threshold is $10^{-1.5}$, with $H\beta$ (`H 1 4861.32A`) as the reference line. Custom linelists can be created manually or using this tool.
- the **full list** (`../linelist-full-{version}.dat`, e.g. `../linelist-full-c25.00.dat`), every line that Cloudy transfers, i.e. all the lines that make up the line contribution (`linecont`) of the saved continuum. It is built from Cloudy's own line labels by the tools in `full_linelist/` and depends on the Cloudy version. The list (about 1.3 MB) is not kept in the repository: create it with the scripts as described below.

## Setup

The scripts read the following environment variables:

- `CLOUDY_EXE`: the `cloudy.exe` to use (required).
- `CLOUDY_DATA_PATH`: Cloudy's data directory. If unset, the `data/` directory next to the build's `source/` is used.
- `CLOUDY_EXTRA` (optional): commands added to the start of every input. Use `CLOUDY_EXTRA="set continuum resolution 0.05"` for a build whose grain opacities were compiled at that resolution (the stellar models include grains).
- `SYNTHESIZER_DIR`: synthesizer's data directory. The incident spectra are taken from the test grids `test_grid.hdf5` (SPS) and `test_grid_agn-nlr.hdf5` (AGN) in `$SYNTHESIZER_DIR/grids`.

## Creating the full list

Neither the full list nor the exclusion list it is built with (`full_linelist/exclude-{version}.txt`) is kept in the repository: both are created by the scripts, by following the procedure below. This takes the labels model, the seven `*_standard` reference models and two `full_linelist.py` commands (or the "Full line list" section of `create_linelist.ipynb`). With c25.00, the AGN models take a few minutes each and the stellar models about half an hour each, at `set continuum resolution 0.05`. The models can be run in parallel.

The result, `../linelist-full-{version}.dat` (in the cloudy directory, next to the other line lists), can then be used as `output_linelist` in the grid parameter files. Once the exclusion list exists, the full list can be rebuilt from it without rerunning the reference models:

```
cd full_linelist
sh run.sh line_labels
python full_linelist.py build line_labels --add ../linelist-standard.dat --exclude exclude-c25.00.txt
```

The labels run must use the same Cloudy version as the exclusion list; otherwise the build is refused.

## Procedure

All commands are run from this directory.

- extract the stellar and AGN incident spectra for use by cloudy (`1.sed` in `stellar_*/` and `agn_*/`) using `extract_spectra.ipynb`.
- run the labels model and build the unfiltered list (`full_linelist/linelist-unfiltered.dat`). The reference models in `stellar_standard/` and `agn_standard/` save their line luminosities for this list, so it must exist before they are run:

  ```
  cd full_linelist
  sh run.sh line_labels
  python full_linelist.py build line_labels --unfiltered --add ../linelist-standard.dat
  cd ..
  ```

  (or run step 1 of the "Full line list" section of `create_linelist.ipynb`).
- run all the `cloudy` models. `cloudy` input files exist for both stellar and AGN sources, including both conventions for line labelling (naming), i.e. standard (air at >200nm, vacuum <200nm) or vacuum wavelengths. In each of `stellar_standard/`, `stellar_vacuum/`, `agn_standard/` and `agn_vacuum/`:

  ```
  sh run_all.sh
  ```

  (or `sh run.sh <n>` for a single model `<n>.in`).
- run `create_linelist.ipynb`:
  - the first sections generate the standard list (changing the parameters as necessary);
  - the "Full line list" section finds the labels to exclude (`full_linelist/exclude-{version}.txt`) and writes the full list (`../linelist-full-{version}.dat`).

  The full list can also be made from the command line:

  ```
  cd full_linelist
  python full_linelist.py check linelist-unfiltered.dat ../stellar_standard/{1..4} ../agn_standard/{1..3} --keep ../linelist-standard.dat
  python full_linelist.py build line_labels --add ../linelist-standard.dat --exclude exclude-c25.00.txt
  ```

The Cloudy version is read from the `.out` files of the runs; runs made with different versions, or an exclusion file for another version, are refused. Builds that do not print a release number need `--version`.

## Full line list

The tools are in `full_linelist/`: `full_linelist.py` and the labels model `line_labels.in` (with `run.sh`). The exclusion lists (`full_linelist/exclude-{version}.txt`) and the list itself (`../linelist-full-{version}.dat`, next to the other line lists, where the grid parameter files find it) are generated and not kept in the repository.

`full_linelist.py build` keeps every transferred line in Cloudy's `save line labels` output. It drops non-line entries, labels without a wavelength and labels with species qualifiers, and always includes `linelist-standard.dat`.

`full_linelist.py check` lists the labels to exclude, using the reference runs made with the unfiltered list:

- ambiguous labels: those Cloudy matches to several lines, or to no exact line. These are read from the warnings in the `.out` files.
- duplicates: labels that resolve to the same line as another label, i.e. with identical luminosities in every run.

The list uses the standard (air) wavelength convention of `line_labels.in`, so only the `*_standard` models are used. Its docstring has the details. The `save line labels` output (`line_labels.labels`) is not kept either, since `line_labels.in` regenerates it in under a minute.

## Adding extra lines

One can add extra lines that could get excluded by the above criteria to the standard list. One can put them in `extra_lines.dat`.
