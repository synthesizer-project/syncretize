"""A module containing I/O helper functions.

This module contains a helper class for writing Synthesizer grids to HDF5
files. It should be used in grid-generation scripts to produce the standard
Synthesizer format.

When initialised, a new file will always be created. This will overwrite any
existing file with the same name. However, for the use case in these scripts
that is the desirable behaviour. If this gets in your way raise an issue on the
GitHub repository.

Any subsequent interactions with the file will be done in append mode.

Common parts of the grid can be written out using the write_grid_common method.

Any non-standard datasets or attributes can be written using the write_dataset
and write_attribute methods. These can be used to write out model specific
datasets and attributes.

Example usage:

    # Create a grid file
    grid = GridFile("grid.hdf5")

    # Write out the common parts of the grid
    grid.write_grid_common(
        model={"model": "BPASS v2.2.1"},
        wavelength=wavelength,
        axes=axes,
        spectra=spectra,
        log_on_read={"ages": True, "metallicities": False},
    )

"""

from datetime import date

import h5py
import numpy as np
from synthesizer._version import __version__ as synthesizer_version
from synthesizer.emissions import Sed
from synthesizer.photoionisation import Ions
from synthesizer.units import has_units
from tqdm import tqdm
from unyt import dimensionless, unyt_array

from syncretize._version import __version__ as syncretize_version


class GridFile:
    """
    A helper object for writing Synthesizer grids to HDF5 files.

    A new file will always be created when a GridFile object is created. This
    will overwrite any existing file with the same name. However, for the use
    case in these scripts that is the desirable behaviour. If this gets in
    your way raise an issue on the GitHub repository.

    Grids may contain any number of axes. Axis dictionary order defines the
    leading dimensions of spectra, lines, and other grid-shaped arrays.

    Attributes:
        filepath (string)
            The file path to where the file should be stored.
        hdf (h5py.File or None)
            The open HDF5 handle, or ``None`` between operations.
        mode (string)
            The mode used to open the file. Initialisation uses ``"w"`` and
            subsequent operations use ``"r+"``.
    """

    # Define common descriptions
    descriptions = {
        "log10age": "Logged stellar population ages (dex(yr))",
        "log10ages": "Logged stellar population ages (dex(yr))",
        "age": "Stellar population ages (yr)",
        "ages": "Stellar population ages (yr)",
        "log10metallicities": "Logged stellar population metallicity",
        "log10metallicity": "Logged stellar population metallicity",
        "metallicities": "Stellar population metallicity",
        "metallicity": "Stellar population metallicity",
        "log_on_read": "Boolean, True for interpolated axes",
    }

    def __init__(self, filepath):
        """
        Initialise the helper.

        This will open the HDF5 file.

        Args:
            filepath (str)
                The file path to where the file should be stored. This should
                include the file name itself.
        """
        # Store the filepath for posterity
        self.filepath = filepath

        # Setup the HDF5 file attribute
        self.hdf = None

        # Set the mode to write by default
        self.mode = "w"

        # Create the file if it doesn't exist
        self._create_file()

    def _create_file(self):
        """
        Create a new file, replacing an existing file at the same path.

        The mode is changed to ``"r+"`` after creation.
        """
        self.hdf = h5py.File(self.filepath, self.mode)
        self.hdf.attrs["syncretize_version"] = syncretize_version
        # Existing grid consumers still use the pre-rename metadata key.
        self.hdf.attrs["synthesizer_grids_version"] = syncretize_version
        self.hdf.attrs["synthesizer_version"] = synthesizer_version
        self.hdf.attrs["date_created"] = str(date.today())
        self.hdf.close()
        self.hdf = None
        self.mode = "r+"

    def _open_file(self):
        """Open the file if it isn't already open."""
        if self.hdf is None:
            self.hdf = h5py.File(self.filepath, self.mode)

    def _close_file(self):
        """Close the file if it is open."""
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None

    def _dataset_exists(self, key):
        """
        Check whether an attribute already exists.

        Only appicable when self.mode = "r+" (i.e. appending)

        Args:
            key (str)
                The key to find. Must be the full path,
                i.e. "Group/SubGroup/Dataset".
        """
        if key not in self.hdf:
            return False
        return True

    def _attr_exists(self, group, attr_key):
        """
        Check to see if a dataset exists already.

        Only appicable when self.mode = "r+" (i.e. appending)

        Args:
            group (str)
                The key of the group where key is stored. Must be the full
                path, i.e. "Group/SubGroup".
            attr_key (str)
                The attribute key.
        """
        if attr_key not in self.hdf[group].attrs:
            return False
        return True

    def write_attribute(self, group, attr_key, data):
        """
        Write data into an attribute.

        Args:
            group (str)
                The key of the group where attr_key will be stored. Must be the
                full path, i.e. "Group/SubGroup".
            attr_key (str)
                The attribute name.
            data (array-like or scalar)
                HDF5-compatible attribute data.
        """
        # Open the file
        self._open_file()

        # If the dataset exists already we need to throw an error (safe if
        # we are appending and overwriting as this was handled above)
        if self._attr_exists(group, attr_key):
            raise ValueError(f"{attr_key} already exists in {group}")

        # Finally, Write it!
        self.hdf[group].attrs[attr_key] = data

        self._close_file()

    def write_dataset(
        self,
        key,
        data,
        description,
        log_on_read,
        verbose=True,
        **extra_attrs,
    ):
        """
        Write data into a dataset.

        Args:
            key (str)
                The key to write data at. Must be the full path,
                i.e. "Group/SubGroup/Dataset".
            data (unyt_array)
                Unit-aware data to write. Shape and dtype are inferred from
                this input.
            description (str)
                A short description of the dataset to be stored alongside
                the data.
            log_on_read (bool)
                Whether Synthesizer should transform the dataset to logarithmic
                space when reading it for interpolation.
            verbose (bool)
                Retained for compatibility; currently has no effect.
            extra_attrs (dict)
                Any attributes of the dataset can be passed in the form:
                attr_key=attr_value.
        """
        # Open the file
        self._open_file()

        # If the dataset exists already we need to throw an error (safe if
        # we are appending and overwriting as this was handled above)
        if self._dataset_exists(key):
            raise ValueError(f"{key} already exists")

        # Ensure we have units on the data
        if not has_units(data):
            raise ValueError(
                f"Data for {key} has no units. Please provide units."
            )

        # Finally, Write it!
        dset = self.hdf.create_dataset(
            key,
            data=data.value,
            shape=data.shape,
            dtype=data.dtype,
        )

        # Set the units attribute
        dset.attrs["Units"] = str(data.units)

        # Include a brief description
        dset.attrs["Description"] = description

        # Write out whether we should log this dataset before using it to
        # interpolate spectra
        dset.attrs["log_on_read"] = log_on_read

        # Handle any other attributes passed as kwargs
        for dset_attr_key, val in extra_attrs.items():
            dset.attrs[dset_attr_key] = val

        self._close_file()

    def write_string_dataset(
        self,
        key,
        data,
        description,
        encoding="utf-8",
        verbose=True,
        **extra_attrs,
    ):
        """
        Write string data into a dataset.

        Args:
            key (str)
                The key to write data at. Must be the full path,
                i.e. "Group/SubGroup/Dataset".
            data (numpy.ndarray)
                String array to write. Shape is inferred from this input.
            description (str)
                A short description of the dataset to be stored alongside
                the data.
            encoding (str)
                The string encoding to use.
            verbose (bool)
                Retained for compatibility; currently has no effect.
            extra_attrs (dict)
                Any attributes of the dataset can be passed in the form:
                attr_key=attr_value.
        """
        # Open the file
        self._open_file()

        # If the dataset exists already we need to throw an error (safe if
        # we are appending and overwriting as this was handled above)
        if self._dataset_exists(key):
            raise ValueError(f"{key} already exists")

        dtype = h5py.string_dtype(encoding=encoding)

        # Finally, Write it!
        dset = self.hdf.create_dataset(
            key,
            data=data,
            shape=data.shape,
            dtype=dtype,
        )

        # Include a brief description
        dset.attrs["Description"] = description

        # log_on_read will always be False for a string dataset
        dset.attrs["log_on_read"] = False

        # Handle any other attributes passed as kwargs
        for dset_attr_key, val in extra_attrs.items():
            dset.attrs[dset_attr_key] = val

        self._close_file()

    def read_attribute(self, attr_key, group="/"):
        """
        Read an attribute and return it.

        Args:
            attr_key (str)
                The key to read.
            group (str)
                The key of the group where attr_key is stored. Must be the
                full path, i.e. "Group/SubGroup".

        Returns:
            array-like/float/int/str
                The attribute stored at hdf[group].attrs[attr_key].
        """
        self._open_file()
        attr = self.hdf[group].attrs[attr_key]
        self._close_file()
        return attr

    def read_dataset(self, key, print_description=False, indices=None):
        """
        Read a dataset and return it.

        Args:
            key (str)
                The key to read. Must be the full path,
                i.e. "Group/SubGroup/Dataset".
            print_description (bool)
                Print the stored ``Description`` attribute when true.
            indices (tuple or slice, optional)
                HDF5 indices selecting a subset of the dataset. The complete
                dataset is returned when omitted.

        Returns:
            unyt_array
                The selected data with units restored from its ``Units``
                attribute.
        """
        # Open the file if necessary
        self._open_file()

        # Get the data, handling whether we get everything or a subset
        if indices is None:
            data = self.hdf[key][...]
        else:
            data = self.hdf[key][indices]

        # Get the units
        unit_str = self.hdf[key].attrs["Units"]

        # Print the description if asked
        if print_description:
            print(self.hdf[key].attrs["Description"])

        self._close_file()

        return unyt_array(data, unit_str)

    def copy_dataset(self, alt_key, key):
        """
        Copy an existing numeric dataset to an alternative key.

        This duplicates the data, description, units, and ``log_on_read``
        attribute. It should therefore only be used for small datasets.

        Args:
            alt_key (str)
                The alternative key to copy to.
            key (str)
                The existing key to copy.
        """
        # Open the file
        self._open_file()

        # Get the data to copy
        data = self.hdf[key][...]
        des = self.hdf[key].attrs["Description"]
        units = self.hdf[key].attrs["Units"]
        log_on_read = self.hdf[key].attrs["log_on_read"]

        # Write the alternative version
        self.write_dataset(
            alt_key,
            unyt_array(data, units),
            des,
            log_on_read=log_on_read,
        )

        self._close_file()

    def write_grid_common(
        self,
        axes,
        wavelength,
        spectra,
        log_on_read,
        alt_axes=(),
        descriptions={},
        model={},
        weight="initial_masses",
    ):
        """
        Write out the common parts of a Synthesizer grid.

        This writer method writes all datasets and attributes that Synthesizer
        expects to exist, regardless of model.

        Any model specific datasets and attributes can still be written using
        explicit calls to write_dataset and write_attribute.

        Args:
            axes (dict[str, unyt_array])
                Axis names and their grid points, in array-dimension order.
            wavelength (unyt_array)
                The wavelength array of the spectra grid.
            spectra (dict[str, unyt_array])
                A dictionary containing the spectra grids. Each key-value pair
                should be {"spectra_type": spectra_grid}. "spectra_type" will
                be the key used for the dataset. Leading dimensions must match
                ``axes`` in insertion order and the final dimension must match
                ``wavelength``.
            log_on_read (dict[str, bool])
                A dictionary with Boolean values for each axis, where True
                makes Synthesizer transform that axis to logarithmic space when
                reading it for interpolation.
            alt_axes (sequence[str])
                Alternative axis names stored in the root
                ``axes_alternative`` attribute. This may be deprecated.
            descriptions (dict[str, str])
                A dictionary containing a brief description of each dataset.
                Keys must match axes. Common descriptions are
                already included in the descriptions class attribute but can
                be overridden here.
            model (dict)
                A dictionary containing the metadata of the model used.
            weight (str)
                Synthesizer property represented by one unit of the spectra.
                SPS grids normally use ``"initial_masses"``.

        Raises:
            ValueError: If units or descriptions are missing, or ``alt_axes``
                does not contain one name per axis.
        """
        if len(model) > 0:
            self.write_model_metadata(model)

        # Write out the axis names to an attribute
        self.write_attribute("/", "axes", list(axes.keys()))

        # Store the weight variable as an attribute
        self.write_attribute("/", "WeightVariable", weight)

        # Parse descriptions and use defaults if not given
        for key in axes:
            if key in descriptions:
                continue
            if key not in descriptions and key in self.descriptions:
                descriptions[key] = self.descriptions[key]
            else:
                raise ValueError(
                    "No description was provided for non-standard "
                    f"axis ({key}). Pass a description to the descriptions "
                    "kwarg."
                )

        # Handle any alternative axis names that have been given
        if len(alt_axes) > 0 and len(alt_axes) == len(axes):
            self.write_attribute("/", "axes_alternative", alt_axes)
        elif len(alt_axes) > 0:
            raise ValueError(
                "alt_axes were passed but conflicted with axes dict."
                f"axes.keys()={axes.keys()}, alt_axes={alt_axes}"
            )

        # Write out each axis array
        for axis_key, axis_arr in axes.items():
            # Ensure we have units on this axis
            if not has_units(axis_arr):
                raise ValueError(
                    f"Axis {axis_key} has no units. Please provide units."
                )

            # Write the dataset
            self.write_dataset(
                "axes/" + axis_key,
                axis_arr,
                descriptions[axis_key],
                log_on_read=log_on_read[axis_key],
            )

        # Write out the spectra grids
        self.write_spectra(spectra, wavelength)

    def write_spectra(self, spectra, wavelength):
        """
        Write out the spectra grids.

        This writes the line datasets to the file.

        Args:
            spectra (dict)
                A dictionary containing the spectra grids. Each key value pair
                should be {"spectra_type": spectra_grid}. "spectra_type" will
                be the key used for the dataset. The final array dimension must
                match ``wavelength``.
            wavelength (unyt_array)
                The wavelength array of the spectra grid.
        """
        # Write out the wavelength array
        self.write_dataset(
            "spectra/wavelength",
            wavelength,
            "Wavelength of the spectra grid",
            log_on_read=False,
        )

        # Write out each spectra
        for key, val in spectra.items():
            # Make sure the spectra has units
            if not has_units(val):
                raise ValueError(
                    f"Spectra {key} has no units. Please provide units."
                )

            # Write the spectra
            self.write_dataset(
                "spectra/" + key,
                val,
                "Three-dimensional spectra grid,"
                " [age, metallicity, wavelength]",
                log_on_read=False,
            )

    def write_lines(self, lines, weight="initial_masses"):
        """
        Write out the lines grids.

        This will write out the spectra grids to the file.

        Args:
            lines (dict)
                A dictionary containing the lines grid, including wavelengths
                and ids. Grid-shaped arrays must follow root axis order and use
                line ID as their final dimension. Reprocessed grids should also
                include ``nebular_continuum`` and ``transmitted`` for current
                Synthesizer line loading.
            weight (str)
                Retained for compatibility; currently has no effect.
        """
        # Write out the wavelength array
        self.write_dataset(
            "lines/wavelength",
            lines["wavelength"],
            "Wavelength of each emission line",
            log_on_read=False,
        )

        # Write out the id array
        self.write_string_dataset(
            "lines/id",
            np.array(lines["id"]).astype("object"),
            "Cloudy ID of each emission line",
        )

        # Write out the lumminosity array
        self.write_dataset(
            "lines/luminosity",
            lines["luminosity"],
            "Line luminosity",
            log_on_read=False,
        )

        # If there are additional entries assume these are continuum
        # luminosities and save.
        if len(lines.keys()) > 3:
            for key, array in lines.items():
                if key not in ["luminosity", "wavelength", "id"]:
                    print(key)
                    self.write_dataset(
                        f"lines/{key}",
                        lines[key],
                        f"{' '.join(key.split('_'))} luminosity",
                        log_on_read=False,
                    )

    def add_specific_ionising_lum(self, ions=("HI", "HeII"), limit=100):
        """
        Calculate the specific ionising photon luminosity for different ions.

        This will also write them to the file.

        This can only be used after the spectra arrays have been written out!

        Args:
            ions (list)
                A list of ions to calculate Q for.
            limit (float/int)
                An upper bound on the number of subintervals
                used in the integration adaptive algorithm.

        """
        # Open the file
        self._open_file()

        # Get the properties of the grid including the dimensions etc.
        (
            _,
            shape,
            _,
            _,
            _,
            index_list,
        ) = self.get_grid_properties()

        self._close_file()

        # Set up output arrays in a dict
        out_arrs = {}
        for ion in ions:
            out_arrs[ion] = np.zeros(shape)

        # Get wavelength grid
        lam = self.read_dataset("spectra/wavelength")

        # Loop over grid points and calculate Q and store it
        for indices in tqdm(index_list):
            indices = tuple(indices)

            # Loop over ions
            for ion in ions:
                # Get the ionisation energy
                ionisation_energy = Ions.energy[ion]

                # Get incident spectrum
                lnu = self.read_dataset("spectra/incident", indices=indices)

                # Calculate Q
                sed = Sed(lam, lnu)
                ionising_lum = sed.calculate_ionising_photon_production_rate(
                    ionisation_energy=ionisation_energy,
                    limit=limit,
                )

                # Store the results at the correct indices
                out_arrs[ion][indices] = np.log10(ionising_lum)

        # Loop over the iopns and write out their arrays
        for ion in ions:
            self.write_dataset(
                f"log10_specific_ionising_luminosity/{ion}",
                out_arrs[ion] * dimensionless,
                "Two-dimensional {ion} ionising photon "
                "production rate grid, [age, Z]",
                log_on_read=False,
            )

        self._close_file()

    def write_model_metadata(self, model):
        """
        Write out the model metadata.

        Args:
            model (dict)
                A dictionary containing the metadata of the model used.
        """
        # Open the file
        self._open_file()

        # Create a group for the model metadata
        grp = self.hdf.create_group("Model")

        # Write out model parameters as attributes
        for key, value in model.items():
            grp.attrs[key] = value

        self._close_file()

    def write_cloudy_metadata(self, params):
        """
        Write out the Cloudy metadata.

        Args:
            params (dict)
                A dictionary containing the metadata of the Cloudy run.
        """
        # Open the file
        self._open_file()

        # Create the CloudyParams group if it doesn't exist
        if "CloudyParams" not in self.hdf:
            self.hdf.create_group("CloudyParams")
        cloudy_grp = self.hdf["CloudyParams"]

        # Add other parameters as attributes
        for k, v in params.items():
            # If v is None then convert to string None for saving in the
            # HDF5 file.
            if v is None:
                v = "None"

            # If the parameter is a dictionary (e.g. as used for abundances)
            if isinstance(v, dict):
                # Create group for this key
                nested_grp = cloudy_grp.create_group(k)

                for k2, v2 in v.items():
                    nested_grp.attrs[k2] = v2

            else:
                cloudy_grp.attrs[k] = v

        # Close the file
        self._close_file()

    def get_grid_properties(self, verbose=False):
        """Return dimensions, values, and indices for every grid point.

        Args:
            verbose (bool): Print each returned property when true.

        Returns:
            tuple: ``(n_axes, shape, n_models, mesh, model_list, index_list)``.
                ``shape`` follows the order in the root ``axes`` attribute;
                ``model_list`` contains the axis values for every model and
                ``index_list`` contains the corresponding array indices.
        """
        self._open_file()

        axes = self.hdf.attrs["axes"]  # list of axes

        # dictionary of axis grid points
        axes_values = {axis: self.hdf["axes"][axis][:] for axis in axes}

        self._close_file()

        # the grid axes
        if verbose:
            print(f"axes: {axes}")

        # number of axes
        n_axes = len(axes)
        if verbose:
            print(f"number of axes: {n_axes}")

        # the shape of the grid (useful for creating outputs)
        shape = list([len(axes_values[axis]) for axis in axes])
        if verbose:
            print(f"shape: {shape}")

        # determine number of models
        n_models = np.prod(shape)
        if verbose:
            print(f"number of models to run: {n_models}")

        # create the mesh of the grid
        mesh = np.array(
            np.meshgrid(*[np.array(axes_values[axis]) for axis in axes])
        )

        # create the list of the models
        model_list = mesh.T.reshape(n_models, n_axes)
        if verbose:
            print("model list:")
            print(model_list)

        # create a list of the indices

        index_mesh = np.array(np.meshgrid(*[range(n) for n in shape]))

        index_list = index_mesh.T.reshape(n_models, n_axes)
        if verbose:
            print("index list:")
            print(index_list)

        return n_axes, shape, n_models, mesh, model_list, index_list


def read_params(param_file):
    """
    Read a parameter file and return the imported parameters.

    Args:
    param_file (str) location of parameter file

    Returns:
    parameters (object)
    """
    return __import__(param_file)
