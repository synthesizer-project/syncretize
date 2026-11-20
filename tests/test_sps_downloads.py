"""Tests for the helpers that unpack downloaded SPS model archives."""

import gzip
import io
import tarfile

from syncretize.incident.sps.utils import extract_gzipped_tables


def test_extract_gzipped_tables(tmp_path):
    archive = tmp_path / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in (
            ("spectra-bin.z001.dat.gz", gzip.compress(b"1 2\n")),
            ("colours-bin.z001.dat.gz", gzip.compress(b"3 4\n")),
            ("starmass-bin.z001.dat", b"5 6\n"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))

    written = extract_gzipped_tables(
        archive, tmp_path / "model", ("spectra-", "starmass-")
    )

    # Only the chosen tables, decompressed
    assert sorted(p.split("/")[-1] for p in map(str, written)) == [
        "spectra-bin.z001.dat",
        "starmass-bin.z001.dat",
    ]
    assert (
        tmp_path / "model" / "spectra-bin.z001.dat"
    ).read_bytes() == b"1 2\n"
