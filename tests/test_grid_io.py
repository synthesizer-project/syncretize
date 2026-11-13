import h5py
import numpy as np
import pytest
from unyt import Angstrom, Hz, Lsun, erg, s, unyt_array, yr

from syncretize.grid_io import GridFile, add_ionisation_edges


def _decode_attr(value):
    """Helper to decode HDF5 string attributes."""
    return value.decode() if isinstance(value, bytes) else value


@pytest.fixture
def simple_grid(tmp_path):
    out_path = tmp_path / "grid.hdf5"
    grid = GridFile(str(out_path))

    axes = {
        "ages": unyt_array([1.0, 10.0], yr),
        "metallicities": unyt_array([0.001, 0.02], "dimensionless"),
    }
    log_on_read = {"ages": True, "metallicities": False}
    wavelength = unyt_array([500.0, 1000.0, 2000.0], Angstrom)
    spectra_shape = (
        len(axes["ages"]),
        len(axes["metallicities"]),
        len(wavelength),
    )
    spectra = {
        "incident": unyt_array(
            np.ones(spectra_shape),
            erg / s / Hz,
        )
    }

    grid.write_grid_common(
        axes=axes,
        wavelength=wavelength,
        spectra=spectra,
        log_on_read=log_on_read,
        model={"model": "unit-test"},
    )

    # write_grid_common adds points either side of the ionisation edges
    written, _ = add_ionisation_edges(wavelength, spectra)

    return {
        "grid": grid,
        "path": out_path,
        "axes": axes,
        "wavelength": written,
        "spectra_shape": (*spectra_shape[:-1], len(written)),
    }


def test_gridfile_sets_metadata(tmp_path):
    out_path = tmp_path / "grid.hdf5"

    GridFile(str(out_path))

    with h5py.File(out_path, "r") as hdf:
        attrs = {key: _decode_attr(val) for key, val in hdf.attrs.items()}
        assert "syncretize_version" in attrs
        assert (
            attrs["synthesizer_grids_version"] == attrs["syncretize_version"]
        )
        assert "synthesizer_version" in attrs
        assert "date_created" in attrs


def test_write_dataset_requires_units(tmp_path):
    out_path = tmp_path / "grid.hdf5"
    grid = GridFile(str(out_path))

    with pytest.raises(ValueError):
        grid.write_dataset(
            key="axes/ages",
            data=np.array([1, 2]),
            description="unitless ages should fail",
            log_on_read=False,
        )


def test_write_grid_common_writes_expected_structure(simple_grid):
    with h5py.File(simple_grid["path"], "r") as hdf:
        axes_attr = list(map(_decode_attr, hdf.attrs["axes"]))
        assert axes_attr == ["ages", "metallicities"]
        assert hdf.attrs["WeightVariable"] == "initial_masses"

        ages_dataset = hdf["axes"]["ages"]
        assert ages_dataset.shape == (2,)
        assert ages_dataset.attrs["log_on_read"]

        metallicities_dataset = hdf["axes"]["metallicities"]
        assert not metallicities_dataset.attrs["log_on_read"]

        assert hdf["spectra"]["wavelength"].shape == (
            len(simple_grid["wavelength"]),
        )
        assert hdf["spectra"]["incident"].shape == simple_grid["spectra_shape"]

        model_group = hdf["Model"]
        assert model_group.attrs["model"] == "unit-test"


def test_write_lines_creates_expected_datasets(simple_grid):
    grid = simple_grid["grid"]
    axes = simple_grid["axes"]
    n_lines = 2
    base_shape = (
        len(axes["ages"]),
        len(axes["metallicities"]),
        n_lines,
    )

    lines = {
        "wavelength": unyt_array([4861.0, 6563.0], Angstrom),
        "id": ["Hbeta", "Halpha"],
        "luminosity": unyt_array(np.ones(base_shape), erg / s),
        "continuum": unyt_array(np.full(base_shape, 2.0), erg / s),
    }

    grid.write_lines(lines)

    with h5py.File(simple_grid["path"], "r") as hdf:
        assert hdf["lines"]["wavelength"].shape == (n_lines,)
        assert hdf["lines"]["luminosity"].shape == base_shape
        assert hdf["lines"]["luminosity"].attrs["Units"] == "Lsun"
        np.testing.assert_allclose(
            hdf["lines"]["luminosity"][...], (1.0 * erg / s).to_value(Lsun)
        )
        assert hdf["lines"]["continuum"].shape == base_shape
        assert list(map(_decode_attr, hdf["lines"]["id"][...])) == [
            "Hbeta",
            "Halpha",
        ]


def test_add_specific_ionising_luminosity(simple_grid):
    grid = simple_grid["grid"]
    axes = simple_grid["axes"]

    grid.add_specific_ionising_lum(ions=("HI",))

    with h5py.File(simple_grid["path"], "r") as hdf:
        data = hdf["log10_specific_ionising_luminosity"]["HI"][...]

    expected_shape = (
        len(axes["ages"]),
        len(axes["metallicities"]),
    )
    assert data.shape == expected_shape
    assert np.isfinite(data).all()


def test_grid_readable_by_synthesizer(simple_grid):
    """A grid written here can be loaded back into a Synthesizer Grid.

    This exercises the grid-generation <-> Synthesizer link end to end,
    confirming the on-disk layout matches what Synthesizer expects.
    """
    # Skip gracefully if Synthesizer is not installed in the environment.
    grid_module = pytest.importorskip("synthesizer.grid")
    Grid = grid_module.Grid

    path = simple_grid["path"]
    axes = simple_grid["axes"]
    wavelength = simple_grid["wavelength"]

    # Grid takes the file name (without extension) and its directory.
    read_grid = Grid(
        path.stem,
        grid_dir=str(path.parent),
        ignore_lines=True,
    )

    # Axes round-trip with the names and values we wrote.
    assert read_grid.axes == ["ages", "metallicities"]
    np.testing.assert_allclose(
        np.asarray(read_grid.ages.to("yr")),
        np.asarray(axes["ages"].to("yr")),
    )
    np.testing.assert_allclose(
        np.asarray(read_grid.metallicities),
        np.asarray(axes["metallicities"]),
    )

    # Wavelength and spectra round-trip with the expected shape and values.
    np.testing.assert_allclose(
        np.asarray(read_grid.lam.to("Angstrom")),
        np.asarray(wavelength.to("Angstrom")),
    )
    assert read_grid.available_spectra == ["incident"]
    assert read_grid.spectra["incident"].shape == simple_grid["spectra_shape"]
    np.testing.assert_allclose(read_grid.spectra["incident"], 1.0)


def test_gridfile_stores_param_file(tmp_path):
    param_file = tmp_path / "params.yaml"
    param_file.write_text(
        "grid_dir: grids\n"
        "models: [a, b]\n"
        "nested:\n"
        "  value: 1.5\n"
        "  missing: null\n"
        "imf: [{name: lower, value: 0.1}]\n"
        "0.02: solar\n"
    )
    out_path = tmp_path / "grid.hdf5"

    GridFile(str(out_path), param_file=str(param_file))

    with h5py.File(out_path, "r") as hdf:
        grp = hdf["ParameterFile"]
        assert grp.attrs["grid_dir"] == "grids"
        assert grp.attrs["0.02"] == "solar"
        assert [_decode_attr(m) for m in grp.attrs["models"]] == ["a", "b"]
        assert grp["nested"].attrs["value"] == 1.5
        assert grp["nested"].attrs["missing"] == "None"
        assert "{'name': 'lower', 'value': 0.1}" in grp.attrs["imf"]
        assert grp["contents"][()].decode() == param_file.read_text()
        assert grp["contents"].attrs["path"] == str(param_file.resolve())


def test_gridfile_grid_name_sets_file_name(tmp_path):
    param_file = tmp_path / "params.yaml"
    param_file.write_text("grid_name: published-name_0.1,100\n")

    grid = GridFile(str(tmp_path / "script-name.hdf5"), param_file=param_file)

    assert grid.filepath == str(tmp_path / "published-name_0.1,100.hdf5")
    assert (tmp_path / "published-name_0.1,100.hdf5").exists()
    assert not (tmp_path / "script-name.hdf5").exists()


def test_gridfile_rejects_contents_mapping_before_truncating(tmp_path):
    param_file = tmp_path / "params.yaml"
    param_file.write_text("contents:\n  a: 1\n")
    out_path = tmp_path / "grid.hdf5"
    out_path.write_text("existing")

    with pytest.raises(ValueError, match="contents"):
        GridFile(str(out_path), param_file=str(param_file))
    assert out_path.read_text() == "existing"
