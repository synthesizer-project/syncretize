import h5py
import numpy as np
from spectres import spectres

from synthesizer_grids.extras.rebin_grid import rebin_grid


def test_rebin_grid_preserves_data_and_uses_reference_wavelengths(tmp_path):
    source_path = tmp_path / "source.hdf5"
    reference_path = tmp_path / "reference.hdf5"
    output_path = tmp_path / "output.hdf5"
    old_wavelength = np.array([1.0, 2.0, 3.0, 4.0])
    new_wavelength = np.array([1.5, 2.5, 3.5])
    flux = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])

    with h5py.File(source_path, "w") as source:
        source.attrs["axes"] = ["ages"]
        source.create_group("axes").create_dataset("ages", data=[1.0, 2.0])
        source.create_dataset("failures", data=[0, 1])
        source.create_group("lines").create_dataset("id", data=[b"line"])
        spectra = source.create_group("spectra")
        spectra.attrs["spec_names"] = ["incident"]
        wavelength = spectra.create_dataset("wavelength", data=old_wavelength)
        wavelength.attrs["Units"] = "Angstrom"
        incident = spectra.create_dataset("incident", data=flux)
        incident.attrs["Units"] = "erg/(Hz*s)"
        spectra.create_dataset("normalisation", data=[2.0, 3.0])

    with h5py.File(reference_path, "w") as reference:
        reference.create_group("spectra").create_dataset(
            "wavelength", data=new_wavelength
        )

    rebin_grid(source_path, reference_path, output_path)

    with h5py.File(output_path) as output:
        np.testing.assert_array_equal(
            output["spectra/wavelength"][:], new_wavelength
        )
        np.testing.assert_allclose(
            output["spectra/incident"][:],
            spectres(
                new_wavelength,
                old_wavelength,
                flux,
                fill=0.0,
                verbose=False,
            ),
        )
        np.testing.assert_array_equal(output["failures"][:], [0, 1])
        np.testing.assert_array_equal(
            output["spectra/normalisation"][:], [2.0, 3.0]
        )
        assert output["spectra/incident"].attrs["Units"] == "erg/(Hz*s)"
