"""
Create an FSPS synthesizer grid for the IMF chosen in the parameter file.

To add an IMF parameter as a grid axis use
generate_incident_grid_variable_imf.py instead.

Example:
    python generate_incident_grid.py fsps.yaml --grid-dir path/to/grids
"""

import fsps
import numpy as np
from unyt import Hz, angstrom, dimensionless, erg, s, yr

from syncretize.grid_io import GridFile
from syncretize.incident.sps.utils import get_model_filename
from syncretize.parser import get_choice, incident_parser, load_param_file


def generate_grid(model, grid_dir, param_file=None):
    """Main function to create fsps grids used by synthesizer"""

    # set up StellarPopulation grid
    imf_type = model["imf_type"]
    imf_masses = model["imf_masses"]

    if imf_type == "chabrier03":
        sp = fsps.StellarPopulation(
            imf_type=1,
            imf_upper_limit=imf_masses[-1],
            imf_lower_limit=imf_masses[0],
        )
    elif imf_type == "bpl":
        imf_slopes = model["imf_slopes"]

        # FSPS fixes the internal boundaries at 0.5 and 1.0 Msun
        if (
            len(imf_masses) != 4
            or imf_masses[1] != 0.5
            or imf_masses[2] != 1.0
            or len(imf_slopes) != 3
        ):
            raise ValueError(
                "The bpl IMF needs imf_masses [m_low, 0.5, 1.0, m_high] "
                "and three imf_slopes"
            )

        sp = fsps.StellarPopulation(
            imf_type=2,
            imf_upper_limit=imf_masses[-1],
            imf_lower_limit=imf_masses[0],
            imf1=imf_slopes[0],
            imf2=imf_slopes[1],
            imf3=imf_slopes[2],
        )

    # include the isochrones/spectra as the sps_variant name
    # model['sps_variant'] = '-'.join(map(lambda x: str(x), sp.libraries[:2]))

    model["sps_variant"] = "-".join(
        map(lambda x: x.decode("utf-8"), sp.libraries[:2])
    )
    print(model["sps_variant"])

    # generate the synthesizer_model_name
    synthesizer_model_name = get_model_filename(model)

    # this is the full path to the ultimate HDF5 grid file
    out_filename = f"{grid_dir}/{synthesizer_model_name}.hdf5"

    lam = sp.wavelengths  # units: Angstroms
    log10ages = sp.log_age  # units: log10(years)
    ages = 10**log10ages
    metallicities = sp.zlegend  # units: log10(metal)

    na = len(log10ages)
    nmetal = len(metallicities)

    spec = np.zeros((na, nmetal, len(lam)))
    stellar_fraction = np.zeros((na, nmetal))

    for imetal in range(nmetal):
        spec_ = sp.get_spectrum(zmet=imetal + 1)[1]  # 2D array Lsol / AA
        _mfrac = sp.stellar_mass / sp.formed_mass
        for ia in range(na):
            lnu = spec_[ia]  # Lsol / Hz
            lnu *= 3.826e33  # erg s^-1 Hz^-1 Msol^-1
            spec[ia, imetal] = lnu
            stellar_fraction[ia, imetal] = _mfrac[ia]

    # Create the GridFile ready to take outputs
    out_grid = GridFile(out_filename, param_file=param_file)

    print("metallicities:", metallicities)

    # A dictionary with Boolean values for each axis, where True
    # indicates that the attribute should be interpolated in
    # logarithmic space.
    log_on_read = {"ages": True, "metallicities": False}

    # Write everything out thats common to all models
    out_grid.write_grid_common(
        model=model,
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=lam * angstrom,
        spectra={"incident": spec * erg / s / Hz},
        alt_axes=("log10ages", "metallicities"),
        log_on_read=log_on_read,
        weight="initial_masses",
    )

    # Write datasets specific to fsps
    out_grid.write_dataset(
        "star_fraction",
        stellar_fraction * dimensionless,
        "Two-dimensional remaining stellar fraction grid, [age, Z]",
        log_on_read=False,
    )

    # Include the specific ionising photon luminosity
    out_grid.add_specific_ionising_lum()


if __name__ == "__main__":
    parser = incident_parser("FSPS grid creation")
    args = parser.parse_args()
    params = load_param_file(args.param_file)

    imf_type = get_choice(params, "imf_type", ["bpl", "chabrier03"])
    imf_masses = [float(m) for m in params["imf_masses"]]
    if imf_type == "chabrier03" and len(imf_masses) != 2:
        raise ValueError("The chabrier03 IMF needs imf_masses [m_low, m_high]")

    model = {
        "sps_name": "fsps",
        # this is set later depending on the isochrones/spectra used
        "sps_variant": False,
        "imf_type": imf_type,
        "imf_masses": imf_masses,
        "imf_slopes": params.get("imf_slopes", [])
        if imf_type == "bpl"
        else [],
        "alpha": False,
        "pyfsps_version": str(fsps.__version__),
        "sps_version": "3.2",
    }

    generate_grid(model, args.grid_dir, param_file=args.param_file)
