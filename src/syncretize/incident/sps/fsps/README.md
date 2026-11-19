# Building FSPS for the FSPS grid scripts

`generate_incident_grid.py` and `generate_incident_grid_variable_imf.py` use
[python-fsps](https://github.com/dfm/python-fsps), which compiles FSPS into
itself. The FSPS version and the isochrone, spectral and dust libraries are
therefore fixed when python-fsps is built; they can't be chosen at run time.
Each config sets them in its `Model` section, and the scripts stop if the
installation differs:

```yaml
Model:
  sps_version: "4.0"         # which FSPS, via the python-fsps release
  libraries: [mist, c3k_lr]  # [isochrones, spectra, dust], as compiled
```

## Versions

Each python-fsps release bundles one FSPS commit, so `sps_version` picks the
python-fsps release to install:

| `sps_version` | python-fsps | Bundled FSPS                    | Notes |
|---------------|-------------|---------------------------------|-------|
| `"4.0"`       | 0.5.0       | v4.0 (`05b5e55`, Jul 2026)      | MIST with [Fe/H] and [alpha/Fe] (13 metallicities, Z☉ = 0.0185). Defaults to MIST + C3K_LR. |
| `"3.2"`       | 0.4.7       | `82a8735` (May 2024)            | Older MIST (12 metallicities, Z☉ = 0.0142). Defaults to MIST + MILES. The published `fsps-3.2-*` grids were made with this. |

`SPS_HOME` must point to a checkout of the bundled FSPS commit, which holds the
data files that code reads. A different checkout usually fails at start-up
with an FSPS `... cannot be opened` error, or gives different isochrones.

## Choosing the libraries

The libraries are C preprocessor flags passed through `FFLAGS` when
python-fsps is built. Set the ones you want to 1 and the defaults you don't
want to 0:

| Kind        | Flags (default in **bold** for 4.0) |
|-------------|--------------------------------------|
| Isochrones  | **`MIST`**, `PADOVA`, `PARSEC`, `BASTI`, `GENEVA` |
| Spectra     | **`C3K_LR`**, `C3K_HR`, `MILES` |
| Dust        | **`DL07`**, `THEMIS` |

For example, MIST + MILES: `FFLAGS="-DMIST=1 -DMILES=1 -DC3K_LR=0"`.

## Installing FSPS 4.0 (python-fsps 0.5.0)

```bash
git clone --depth 1 --branch v4.0 https://github.com/cconroy20/fsps.git
export SPS_HOME=$PWD/fsps

# Default libraries (MIST + C3K_LR)
pip install --no-binary fsps fsps==0.5.0

# Or other libraries, e.g. MIST + MILES
FFLAGS="-DMIST=1 -DMILES=1 -DC3K_LR=0" \
    pip install --force-reinstall --no-deps --no-binary fsps fsps==0.5.0
```

## Installing FSPS 3.2 (python-fsps 0.4.7)

python-fsps 0.4.7 needs `scikit-build-core` older than 0.8, which pip's
isolated build environment won't use, so build it in a separate venv without
build isolation, against the same numpy as the venv it will run in:

```bash
# The data files for the bundled FSPS commit
mkdir fsps-0.4.7-data && cd fsps-0.4.7-data
git init -q && git remote add origin https://github.com/cconroy20/fsps.git
git fetch --depth 1 origin 82a873508d500ca353bbb922459bf928498f7a72
git checkout FETCH_HEAD && cd ..
export SPS_HOME=$PWD/fsps-0.4.7-data

# The numpy version of the venv syncretize runs in
NP=$(python -c 'import numpy; print(numpy.__version__)')

# Build in a throwaway venv, installing into a separate directory
python -m venv venv-fsps-build && source venv-fsps-build/bin/activate
pip install "scikit-build-core[pyproject]<0.8" pyproject-metadata \
    setuptools_scm cmake ninja "numpy==$NP"
FFLAGS="-DMIST=1 -DMILES=1 -DC3K_LR=0" pip install --no-deps \
    --no-build-isolation --no-binary fsps --target $PWD/fsps-0.4.7 fsps==0.4.7
deactivate

# Use it by putting it ahead of any other python-fsps
export PYTHONPATH=$PWD/fsps-0.4.7:$PYTHONPATH
```

## Checking an installation

```bash
python -c "import fsps; sp = fsps.StellarPopulation(zcontinuous=0); \
    print(fsps.__version__, sp.libraries, len(sp.zlegend))"
```

This prints the python-fsps version (`0.5.0` for `sps_version: "4.0"`,
`0.4.7` for `"3.2"`), the compiled libraries and the number of metallicities.
