"""Rebin a Synthesizer grid onto another grid's wavelength sampling."""

import argparse
import os
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
from spectres import spectres


def _copy_attrs(source, target):
    """Copy HDF5 attributes between objects."""
    for key, value in source.attrs.items():
        target.attrs[key] = value


def rebin_grid(input_grid, reference_grid, output_grid, chunk_size=1):
    """Rebin spectral datasets while preserving all other grid data."""
    input_grid = Path(input_grid)
    reference_grid = Path(reference_grid)
    output_grid = Path(output_grid)
    temporary_grid = output_grid.with_suffix(output_grid.suffix + ".tmp")

    if output_grid.exists() or temporary_grid.exists():
        raise FileExistsError(f"Output already exists: {output_grid}")

    with ExitStack() as stack:
        source = stack.enter_context(h5py.File(input_grid, "r"))
        reference = stack.enter_context(h5py.File(reference_grid, "r"))
        output = stack.enter_context(h5py.File(temporary_grid, "w"))
        old_wavelength = source["spectra/wavelength"][:]
        new_wavelength = reference["spectra/wavelength"][:]

        if np.any(np.diff(old_wavelength) <= 0) or np.any(
            np.diff(new_wavelength) <= 0
        ):
            raise ValueError("Wavelength grids must be strictly increasing")
        if (
            new_wavelength[0] < old_wavelength[0]
            or new_wavelength[-1] > old_wavelength[-1]
        ):
            raise ValueError("Reference wavelengths exceed input grid range")

        _copy_attrs(source, output)
        for name in source:
            if name != "spectra":
                source.copy(name, output)

        source_spectra = source["spectra"]
        output_spectra = output.create_group("spectra")
        _copy_attrs(source_spectra, output_spectra)

        wavelength = output_spectra.create_dataset(
            "wavelength", data=new_wavelength
        )
        _copy_attrs(source_spectra["wavelength"], wavelength)

        for name, dataset in source_spectra.items():
            if name == "wavelength":
                continue
            if dataset.ndim == 0 or dataset.shape[-1] != len(old_wavelength):
                source_spectra.copy(name, output_spectra)
                continue

            shape = (*dataset.shape[:-1], len(new_wavelength))
            chunks = (1, *dataset.shape[1:-1], len(new_wavelength))
            rebinned = output_spectra.create_dataset(
                name, shape=shape, dtype=dataset.dtype, chunks=chunks
            )
            _copy_attrs(dataset, rebinned)

            for start in range(0, dataset.shape[0], chunk_size):
                stop = min(start + chunk_size, dataset.shape[0])
                rebinned[start:stop] = spectres(
                    new_wavelength,
                    old_wavelength,
                    dataset[start:stop],
                    fill=0.0,
                    verbose=False,
                )

    os.replace(temporary_grid, output_grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebin a grid to a reference grid's wavelengths"
    )
    parser.add_argument("--input-grid", required=True)
    parser.add_argument("--reference-grid", required=True)
    parser.add_argument("--output-grid", required=True)
    parser.add_argument("--chunk-size", type=int, default=1)
    args = parser.parse_args()

    rebin_grid(
        args.input_grid,
        args.reference_grid,
        args.output_grid,
        args.chunk_size,
    )
