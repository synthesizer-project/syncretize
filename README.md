# Syncretize

Syncretize builds HDF5 model grids for the
[`synthesizer`](https://github.com/synthesizer-project/synthesizer) synthetic
observations package. It includes generators for stellar and black-hole
incident models, a flexible `GridFile` writer for new models, and a Cloudy
reprocessing pipeline.

## Installation

```sh
python -m pip install .
```

Model generators can require external software or source data. See the model
guide before running one.

## Documentation

- [Getting started](docs/getting_started.rst)
- [Grid file format](docs/grid_format.rst)
- [Making your own grid](docs/making_your_own_grid.rst)
- [Incident model generators](docs/incident_models.rst)
- [Cloudy reprocessing](docs/reprocessing.rst)
- [Uploading to Syndex](docs/uploading_to_syndex.rst)
- [GridFile reference](docs/grid_file.rst)

Build the HTML documentation with:

```sh
python -m sphinx -W docs docs/_build/html
```

## Contributing

Contributions are welcome through the
[`syncretize` repository](https://github.com/synthesizer-project/syncretize).
