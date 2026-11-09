"""
Create the Draine and Li 2007 dust emission grids.
Here we are using the updated the grids from 2014,
where the maximum of the interstellar radiation field
is fixed to 1e7. Similar to the SPS models, the spectra
are per mass: the DL07 emissivities are per hydrogen nucleon,
so here they are per Msun of hydrogen.
"""

import argparse

import numpy as np
from unyt import Angstrom, Hz, Msun, Unit, c, dimensionless, erg, mp, s, um

from syncretize.grid_io import GridFile


def process_dl07_to_hdf5(
    grid_name="MW3.1",
    grid_loc="./",
    data_name="MW3.1",
    data_loc="./synthesizer_grid_data/DL07",
):
    """
    Function to process the text files from dl07 to N-dimensional hdf5 grids
    for use in dust/emission.py
    These are the updated 2014 files.
    Args:
        grid_name (str)
            Name of the hdf5 grid to create from input data
        grid_loc (str)
            Location to store the grid (default is current directory)
        data_name (str)
            Dust type of the data file to process (currently only MW type)
        data_loc (str)
            Location of the input data file to process
    """

    # Define the models parameters
    suffix = np.array(
        [
            "000",
            "010",
            "020",
            "030",
            "040",
            "050",
            "060",
            "070",
            "080",
            "090",
            "100",
        ]
    )
    qpahs = 0.01 * np.array(
        [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58, 5.26, 5.95, 6.63, 7.32]
    )
    # np.array([0.0047, 0.0112, 0.0177, 0.0250, 0.0319, 0.0390,
    #                   0.0458])
    alphas = np.arange(1.0, 3.1, 0.1)
    umins = np.array(
        [
            "0.100",
            "0.120",
            "0.150",
            "0.170",
            "0.200",
            "0.250",
            "0.300",
            "0.350",
            "0.400",
            "0.500",
            "0.600",
            "0.700",
            "0.800",
            "1.000",
            "1.200",
            "1.500",
            "1.700",
            "2.000",
            "2.500",
            "3.000",
            "3.500",
            "4.000",
            "5.000",
            "6.000",
            "7.000",
            "8.000",
            "10.00",
            "12.00",
            "15.00",
            "17.00",
            "20.00",
            "25.00",
            "30.00",
            "35.00",
            "40.00",
            "50.00",
        ]
    )
    umaxs = np.array(["1e7"])

    axes_shape = list([len(qpahs), len(umins), len(alphas)])

    # Read from a file to get wavelength array and its shape
    test_file = f"{data_loc}/U0.100_0.100_{data_name}_{suffix[0]}/spec_1.0.dat"
    tmp = np.genfromtxt(test_file, skip_header=71)
    lam = tmp[:, 0][::-1]
    n_lam = len(lam)
    nu = c / (lam * um)
    nu = nu.to(Hz).value
    msun_by_mp = (Msun / mp).value

    # Allocate the spectra. The diffuse spectra do not depend on alpha.
    pdr = np.zeros((*axes_shape, n_lam))
    diffuse = np.zeros((len(qpahs), len(umins), n_lam))
    pdr_fsil = np.zeros((*axes_shape, n_lam))
    diffuse_fsil = np.zeros((len(qpahs), len(umins), n_lam))

    def read_spec(fname):
        """Read a DL07 spectrum file, returning lnu per Msun of H and fsil."""
        with open(fname) as f:
            # Skip the header, keeping the 1001 wavelength rows
            skip_header = len(f.readlines()) - 1001
        tmp = np.genfromtxt(fname, skip_header=skip_header)
        spec = (tmp[:, 1][::-1] * erg / s / (nu * Hz)).to(erg / s / Hz).value
        fsil = tmp[:, 4][::-1] / (tmp[:, 3][::-1] + tmp[:, 4][::-1])
        return spec * msun_by_mp, fsil

    for ii in range(len(qpahs)):
        for jj, umin in enumerate(umins):
            # Diffuse dust spectrum
            fname = (
                f"{data_loc}/U{umin}_{umin}_{data_name}_{suffix[ii]}"
                "/spec_1.0.dat"
            )
            diffuse[ii, jj], diffuse_fsil[ii, jj] = read_spec(fname)

            # PDR dust spectrum
            for kk, alpha in enumerate(alphas):
                fname = (
                    f"{data_loc}/U{umin}_{umaxs[0]}_{data_name}_"
                    + f"{suffix[ii]}/spec_{np.round(alpha, 2)}.dat"
                )
                pdr[ii, jj, kk], pdr_fsil[ii, jj, kk] = read_spec(fname)

    # Write the grid
    out_grid = GridFile(f"{grid_loc}/{grid_name}.hdf5")
    out_grid.write_attribute("/", "axes", ["qpah", "umin", "alpha"])
    out_grid.write_dataset(
        "axes/qpah",
        qpahs * dimensionless,
        "Fraction of dust mass in the form of PAHs",
        log_on_read=False,
    )
    out_grid.write_dataset(
        "axes/umin",
        umins.astype(float) * dimensionless,
        "Radiation field heating majority of the dust or diffuse dust",
        log_on_read=False,
    )
    out_grid.write_dataset(
        "axes/alpha",
        alphas * dimensionless,
        "Powerlaw slope dU/dM propto U^alpha",
        log_on_read=False,
    )
    out_grid.write_dataset(
        "spectra/wavelength",
        (lam * um).to(Angstrom),
        "Wavelength of the spectra grid",
        log_on_read=False,
    )
    # Built from a string: erg / s / Hz / Msun would simplify to erg/kg
    lnu_units = Unit("erg/(s*Hz*Msun)")
    for key, data, units, description in (
        ("pdr", pdr, lnu_units, "PDR dust emission per Msun of hydrogen"),
        (
            "diffuse",
            diffuse,
            lnu_units,
            "Diffuse dust emission per Msun of hydrogen",
        ),
        ("pdr_fsil", pdr_fsil, dimensionless, "PDR silicate fraction"),
        (
            "diffuse_fsil",
            diffuse_fsil,
            dimensionless,
            "Diffuse silicate fraction",
        ),
    ):
        out_grid.write_dataset(
            f"spectra/{key}",
            data * units,
            description,
            log_on_read=False,
        )
    out_grid.write_attribute(
        "spectra", "spec_names", ["pdr", "diffuse", "pdr_fsil", "diffuse_fsil"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the DL07 grid in the specified directory."
    )
    parser.add_argument("-dir", "--directory", type=str, required=True)
    args = parser.parse_args()

    synthesizer_data_dir = args.directory

    process_dl07_to_hdf5(grid_loc=synthesizer_data_dir)
