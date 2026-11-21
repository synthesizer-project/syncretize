"""What grids record about how they were made (syncretize.provenance)."""

import h5py

from syncretize import provenance
from syncretize.grid_io import GridFile


def test_source_data_are_the_files_read(tmp_path):
    # A data file read (and a config, which isn't source data)
    data = tmp_path / "spectra.dat"
    data.write_text("1 2 3\n")
    config = tmp_path / "config.yaml"
    config.write_text("Model: {}\n")
    provenance.record_reads()
    with open(data) as file:
        file.read()
    with open(config) as file:
        file.read()
    # Written, not read, so not source data
    with open(tmp_path / "written.dat", "w") as file:
        file.write("x")

    grid = GridFile(str(tmp_path / "grid.hdf5"), param_file=str(config))
    grid.write_source_data()
    with h5py.File(grid.filepath) as hdf:
        entries = [dict(e.attrs) for e in hdf["SourceData"].values()]
    found = {entry["path"]: entry for entry in entries}
    assert str(data.resolve()) in found
    assert found[str(data.resolve())]["sha256"] == provenance.sha256(data)
    assert found[str(data.resolve())]["size"] == 6
    assert str(config.resolve()) not in found
    assert str((tmp_path / "written.dat").resolve()) not in found


def test_software_names_the_commit(tmp_path):
    # Run from a git checkout, the commit and its uncommitted changes
    software = provenance.software()
    assert len(software["syncretize_commit"]) == 40
    assert "syncretize_diff" in software
    with h5py.File(GridFile(str(tmp_path / "grid.hdf5")).filepath) as hdf:
        attrs = hdf["Software"].attrs
        assert attrs["syncretize_commit"] == software["syncretize_commit"]
        assert attrs["numpy"] == software["numpy"]
        assert hdf.attrs["syncretize_version"] == software["syncretize"]


def test_cloudy_version_and_directory_checksum(tmp_path):
    out = tmp_path / "0.out"
    out.write_text(
        " " * 50 + "Cloudy 23.01\n" + " " * 50 + "www.nublado.org\n\n"
        "   PRNG seed: 0x1\n\n"
        + " " * 20
        + "*" * 30
        + "23Jul25"
        + "*" * 30
        + "\n"
    )
    assert provenance.cloudy_version(out) == {
        "version": "23.01",
        "date": "23Jul25",
    }
    assert provenance.find_out_file(tmp_path) == str(out)

    # A directory's checksum changes with any of its files
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.dat").write_text("a")
    before = provenance.directory_sha256(data)
    (data / "a.dat").write_text("b")
    assert provenance.directory_sha256(data) != before
