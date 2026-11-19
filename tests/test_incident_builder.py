import h5py
import numpy as np
import pytest
from unyt import Angstrom, Hz, dimensionless, erg, s, yr

from syncretize.incident.builder import GridSpec, Output, build_grid, points

AGES = np.array([1e6, 1e7, 1e8])
METALLICITIES = np.array([0.001, 0.01])
SLOPES = np.array([1.3, 2.3])
LAM = np.geomspace(100.0, 1e5, 400)


def fake_spec(point_axes=("metallicities", "slope")):
    """A model with ages as a block and metallicity and slope as points."""
    return GridSpec(
        grid_name="fake",
        model={"sps_name": "fake"},
        axes={
            "ages": AGES * yr,
            "metallicities": METALLICITIES * dimensionless,
            "slope": SLOPES * dimensionless,
        },
        wavelength=LAM * Angstrom,
        outputs={
            "incident": Output(units=erg / s / Hz),
            "star_fraction": Output(spectra=False, description="Fake"),
        },
        point_axes=point_axes,
        log_on_read={"ages": True, "metallicities": False, "slope": False},
        descriptions={"slope": "Fake IMF slope"},
    )


def expected(age, metallicity, slope):
    """The fake spectrum at one grid point."""
    return 1e20 * (LAM / 1e3) ** -slope * (1 + metallicity) / np.sqrt(age)


def fake_generate_point(spec, index):
    """Return the block over the axes that aren't point axes."""
    ages = spec.axes["ages"].value
    z = spec.axes["metallicities"].value
    slopes = spec.axes["slope"].value
    iz = (
        [index["metallicities"]] if "metallicities" in index else range(len(z))
    )
    js = [index["slope"]] if "slope" in index else range(len(slopes))
    block = np.array(
        [
            [[expected(a, z[i], slopes[j]) for j in js] for i in iz]
            for a in ages
        ]
    )
    # Drop the point axes, which generate_point returns no dimension for
    if "metallicities" in index:
        block = block[:, 0]
    if "slope" in index:
        block = block[..., 0, :]
    return {"incident": block, "star_fraction": block[..., 0] * 0 + 0.5}


def test_points_cover_the_point_axes():
    assert list(points(fake_spec())) == [
        {"metallicities": i, "slope": j} for i in range(2) for j in range(2)
    ]
    # No point axes is a single point covering the whole grid
    assert list(points(fake_spec(point_axes=()))) == [{}]


@pytest.mark.parametrize("point_axes", [("metallicities", "slope"), ()])
def test_build_grid_writes_every_point(tmp_path, point_axes):
    path = build_grid(
        fake_spec(point_axes),
        fake_generate_point,
        str(tmp_path),
        verbose=False,
    )

    with h5py.File(path, "r") as hdf:
        assert list(hdf.attrs["axes"]) == ["ages", "metallicities", "slope"]
        assert hdf["Model"].attrs["sps_name"] == "fake"
        spectra = hdf["spectra/incident"][()]
        lam = hdf["spectra/wavelength"][()]
        # GridFile adds points either side of ionisation edges; compare at
        # the model's own wavelengths
        keep = np.isin(lam, LAM)
        for ia, age in enumerate(AGES):
            for iz, z in enumerate(METALLICITIES):
                for js, slope in enumerate(SLOPES):
                    np.testing.assert_allclose(
                        spectra[ia, iz, js][keep], expected(age, z, slope)
                    )
        assert hdf["star_fraction"].shape == (3, 2, 2)
        assert (hdf["star_fraction"][()] == 0.5).all()
        assert "log10_specific_ionising_luminosity/HI" in hdf


def test_build_grid_cores_match_serial(tmp_path):
    serial = build_grid(
        fake_spec(), fake_generate_point, str(tmp_path), verbose=False
    )
    with h5py.File(serial, "r") as hdf:
        one_core = hdf["spectra/incident"][()]
    parallel = build_grid(
        fake_spec(), fake_generate_point, str(tmp_path), cores=3, verbose=False
    )
    with h5py.File(parallel, "r") as hdf:
        np.testing.assert_array_equal(hdf["spectra/incident"][()], one_core)


def test_build_grid_stores_the_param_file(tmp_path):
    param_file = tmp_path / "params.yaml"
    param_file.write_text("Model:\nConstants:\nAxes:\nSampling:\n")
    path = build_grid(
        fake_spec(),
        fake_generate_point,
        str(tmp_path),
        param_file=str(param_file),
        verbose=False,
    )
    with h5py.File(path, "r") as hdf:
        assert hdf["ParameterFile/contents"][()].decode() == (
            param_file.read_text()
        )


def test_build_grid_needs_every_output(tmp_path):
    def incomplete(spec, index):
        return {"incident": fake_generate_point(spec, index)["incident"]}

    with pytest.raises(ValueError, match="star_fraction"):
        build_grid(fake_spec(), incomplete, str(tmp_path), verbose=False)


def test_point_axes_must_be_grid_axes():
    with pytest.raises(ValueError, match="aren't grid axes"):
        fake_spec(point_axes=("imf",))
