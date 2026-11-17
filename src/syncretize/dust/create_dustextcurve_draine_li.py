#!/usr/bin/env python3
"""
Create dust extinction curves from Draine optical properties.

Parse Draine Sil_81.gz, Gra_81.gz, PAHion_30.gz, PAHneu_30.gz files,
construct grain-size distributions (either MRN or the custom lognormal form)
and compute the mass opacity of each grain component,
kappa(lam) = integral pi a^2 Q(a, lam) n(a) da (units of cm^2 per g of dust),
with n(a) normalised to one gram of dust over the radii where Q is defined.

The extinction scales linearly with the dust mass, so no dust-to-gas grid
is needed: for a line of sight with dust mass column Sigma_dust,c of each
component c, the optical depth is tau(lam) = sum_c kappa_c(lam) Sigma_dust,c
and A(lam) = 1.086 tau(lam).

Two sets of curves are written:
  - extinction_curves: computed with Q_ext, i.e. every scattered photon is
    removed from the beam (point source behind a dust screen).
  - attenuation_curves: computed with Q_abs + (1 - g) Q_sca, i.e.
    forward-scattered light is kept in the beam (transport approximation),
    a simple stand-in for unresolved scattering geometry.

Currently this is done only for 2 binsizes (small and large) for silicate
and graphite, and 1 bin for PAHs (ionized and neutral).

The grid can be coupled with simulations having two bins of grain
size distribution to generate self-consistent extinction curves
using the dust mass column along each line-of-sight for each
dust grain component.
"""

import argparse
import gzip
import io
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from synthesizer.emission_models.attenuation import Calzetti2000, GrainModels
from synthesizer.utils.integrate import trapezoid
from unyt import (
    Angstrom,
    cm,
    g,
    um,
)

from syncretize.grid_io import GridFile


def download_bytes(url: str, timeout: float = 30.0) -> bytes:
    """
    Download bytes from URL with timeout.
    Returns content as bytes.

    Args:
        url (string)
            URL to download
        timeout (float)
            Timeout in seconds (default: 30.0)
    Returns:
        bytes
    """

    print(f"Downloading {url} ...")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def read_gz_lines(gzbytes: bytes, savename: str) -> List[str]:
    """
    Read lines from gzipped bytes, save to savename.
    Returns list of lines as strings.

    Args:
        gzbytes (bytes)
            Gzipped content
        savename (string)
            Filename to save uncompressed text
    Returns:
        string with line breaks
    """
    with gzip.GzipFile(fileobj=io.BytesIO(gzbytes)) as gf:
        txt = gf.read().decode("utf-8", errors="ignore")
    with open(savename, "w") as file:
        file.writelines(txt)
    return txt


def parse_draine_file_lines(
    filename: str,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Parse Draine optical property file (e.g., Sil_81.txt) and extract
    radius grid (micron), wavelength grid (micron), and Qext, Qabs,
    Qsca, gcos arrays.

    Args:
        filename (string)
            path to Draine optical property file
    Returns:
        a_grid (NDArray)
            radius grid (micron)
        wav_grid (NDArray)
            wavelength grid (micron)
        Qext (NDArray)
            extinction efficiency array, shape (len(wav_grid), len(a_grid))
        Qabs (NDArray)
            absorption efficiency array, shape (len(wav_grid), len(a_grid))
        Qsca (NDArray)
            scattering efficiency array, shape (len(wav_grid), len(a_grid))
        gcos (NDArray)
            asymmetry parameter (g) array, shape (len(wav_grid), len(a_grid))
    """

    NRAD = None
    NWAV = None
    a_min = None
    a_max = None
    line_num_NWAV = None

    with open(filename, "r") as file:
        lines = file.read().splitlines()

    # Find NRAD, NWAV, a_min, a_max in early header lines
    for ii, L in enumerate(lines[:50]):
        if "NRAD" in L:
            left = L.split("=")[0].strip()
            toks = left.split()
            if len(toks) >= 3:
                try:
                    NRAD = int(toks[0])
                    a_min = float(toks[1])
                    a_max = float(toks[2])
                except Exception:
                    pass
        if "NWAV" in L:
            left = L.split("=")[0].strip()
            toks = left.split()
            if len(toks) >= 1:
                try:
                    NWAV = int(toks[0])
                    line_num_NWAV = ii
                except Exception:
                    pass
    if NRAD is None or NWAV is None:
        raise RuntimeError(
            "Failed to find NRAD (number of radii) or"
            "NWAV (number of wavelength) in header."
            "Inspect file header manually, and retry."
        )

    print(f"Parsing {filename}:")
    print(f"NRAD = {NRAD}, NWAV = {NWAV}, amin ={a_min}, amax={a_max}")

    skip_ftr = 3  # lines to skip after NWAV line (usually 3)
    first_skip_hdr = (
        1 + line_num_NWAV + skip_ftr
    )  # start parsing after this line

    a_grid = 10 ** np.linspace(
        np.log10(a_min), np.log10(a_max), num=NRAD, endpoint=True
    )

    Qabs: NDArray = np.zeros((NRAD, NWAV))
    Qsca: NDArray = np.zeros((NRAD, NWAV))
    Qext: NDArray = np.zeros((NRAD, NWAV))
    gcos: NDArray = np.zeros((NRAD, NWAV))

    # Pre-read and clean the file to fix missing spaces before minus signs
    # e.g., converts '6.53E-23-1.00E-02' into '6.53E-23 -1.00E-02'
    with open(filename, "r") as f:
        cleaned_lines = [
            line.replace("-", " -").replace("E -", "E-").replace("e -", "e-")
            for line in f
        ]

    for ii in range(NRAD):
        if "PAH" in filename:
            add = 1
            usecols = (0, 1, 2, 3, 4)
        else:
            add = 0
            usecols = (0, 1, 2, 3)

        tmp = np.genfromtxt(
            cleaned_lines,
            skip_header=first_skip_hdr + ii * NWAV + ii * skip_ftr,
            max_rows=NWAV,
            dtype=float,
            usecols=usecols,
        )

        if ii == 0:
            wav_grid = tmp[:, 0]  # micron
        Qabs[ii] = tmp[:, 1 + add]
        Qsca[ii] = tmp[:, 2 + add]
        if "PAH" in filename:
            Qext[ii] = tmp[:, 1]
        else:
            Qext[ii] = tmp[:, 1 + add] + tmp[:, 2 + add]
        gcos[ii] = tmp[:, 3 + add]

    return a_grid, wav_grid, Qext.T, Qabs.T, Qsca.T, gcos.T


def build_mrn_component(
    a_grid_micron: NDArray,
    a_min: float,
    a_max: float,
    power: float,
    dust_mass: float,
    rho: float,
) -> NDArray:
    """
    Build a truncated MRN-like grain-size distribution component
    with dn/da propto a^power over [a_min, a_max] (micron).
    n(a) = C * a^power, where C is a normalisation constant set
    numerically so that the mass on a_grid_micron equals dust_mass;
    any part of [a_min, a_max] outside the grid gets no mass.

    Args:
        a_grid_micron (NDArray)
            radius grid (micron) where n(a) is evaluated
        a_min (float)
            minimum grain radius (micron)
        a_max (float)
            maximum grain radius (micron)
        power (float)
            power-law exponent (dn/da ∝ a^power)
        dust_mass (float)
            dust mass the distribution is normalised to (g). Use 1.0 for
            a distribution per gram of dust.
        rho (float)
            material density (g cm^-3)
    Returns:
        n(a) (NDArray)
            grain size distribution on a_grid_micron: number of grains
            with radii between a and a+da, per dust_mass, per cm.
    """
    micron_to_cm = (1.0 * um).to("cm").value  # 1e-4
    a_grid_cm = a_grid_micron * micron_to_cm

    shape = np.zeros_like(a_grid_micron)
    mask = (a_grid_micron >= a_min) & (a_grid_micron <= a_max)
    shape[mask] = a_grid_cm[mask] ** power

    return normalise_to_mass(a_grid_micron, shape, dust_mass, rho)


def build_lognormal_component(
    a_grid_micron: NDArray,
    a0_micron: float,
    sigma_ln: float,
    dust_mass: float,
    rho: float,
) -> NDArray:
    """
    Build the lognormal-type component with the form
    n(a) = C / a^4 * exp( - (ln(a/a0))^2 / (2 sigma^2) ).
    Reference: Hirashita 2015 (https://arxiv.org/abs/1412.3866), eq 31.
    C is chosen so that integral (4/3 pi a^3 rho n(a) da) = dust_mass,
    with the integral taken over a_grid_micron only (not 0 -> infinity).
    The distribution is therefore truncated to the grid and renormalised,
    so the full dust mass sits on radii where Q is defined. The integral
    is done numerically with the same quadrature as the kappa integral.

    Args:
        a_grid_micron (NDArray)
            radius grid (micron) where n(a) is evaluated
        a0_micron (float)
            center of lognormal (micron)
        sigma_ln (float)
            width of lognormal in ln-space
        dust_mass (float)
            dust mass the distribution is normalised to (g). Use 1.0 for
            a distribution per gram of dust.
        rho (float)
            material density (g cm^-3)
    Returns:
        n(a) (NDArray)
            grain size distribution on a_grid_micron: number of grains
            with radii between a and a+da, per dust_mass, per cm.
    """
    if sigma_ln <= 0:
        raise ValueError("sigma_ln must be > 0")

    exponent = -0.5 * ((np.log(a_grid_micron / a0_micron)) / sigma_ln) ** 2

    micron_to_cm = (1.0 * um).to("cm").value  # 1e-4
    shape = np.exp(exponent) / ((a_grid_micron * micron_to_cm) ** 4)

    return normalise_to_mass(a_grid_micron, shape, dust_mass, rho)


def lognormal_mass_fraction(
    a_lo_micron: float,
    a_hi_micron: float,
    a0_micron: float,
    sigma_ln: float,
) -> float:
    """
    Fraction of the (untruncated) lognormal component's mass lying in
    [a_lo, a_hi]. With n(a) propto a^-4 exp(-(ln(a/a0))^2 / (2 sigma^2)),
    the mass per ln(a) is a Gaussian in x = ln(a/a0) of width sigma.

    Args:
        a_lo_micron (float)
            lower radius (micron)
        a_hi_micron (float)
            upper radius (micron)
        a0_micron (float)
            center of lognormal (micron)
        sigma_ln (float)
            width of lognormal in ln-space
    Returns:
        float
            mass fraction in [a_lo, a_hi]
    """
    z_lo = math.log(a_lo_micron / a0_micron) / (sigma_ln * math.sqrt(2))
    z_hi = math.log(a_hi_micron / a0_micron) / (sigma_ln * math.sqrt(2))
    return 0.5 * (math.erf(z_hi) - math.erf(z_lo))


def normalise_to_mass(
    a_grid_micron: NDArray,
    shape: NDArray,
    dust_mass: float,
    rho: float,
) -> NDArray:
    """
    Scale a size-distribution shape so that its mass on a_grid_micron,
    integral (4/3 pi a^3 rho n(a) da), equals dust_mass. The integral
    uses the same trapezoid quadrature as calculate_kappa.

    Args:
        a_grid_micron (NDArray)
            radius grid (micron) where the shape is evaluated
        shape (NDArray)
            unnormalised n(a) on a_grid_micron (a in cm)
        dust_mass (float)
            dust mass to normalise to (g)
        rho (float)
            material density (g cm^-3)
    Returns:
        n(a) (NDArray)
            normalised size distribution (per dust_mass per cm)
    """
    micron_to_cm = (1.0 * um).to("cm").value  # 1e-4
    a_grid_cm = a_grid_micron * micron_to_cm
    mass = trapezoid(
        (4.0 / 3.0) * math.pi * rho * a_grid_cm**3 * shape, a_grid_cm
    )
    if not mass > 0:
        raise ValueError(
            "Size distribution has no mass on the integration grid; "
            "check the bin overlaps the radii covered by the optical table."
        )
    return shape * (dust_mass / mass)


def calculate_kappa(
    wav_micron: NDArray,
    radii_micron: NDArray,
    Q: NDArray,
    a_grid_micron: NDArray,
    n_a_per_gram: NDArray,
) -> NDArray:
    """
    Mass opacity of a grain component,
    kappa(lam) = integral pi a^2 Q(a, lam) n(a) da,
    with n(a) normalised to one gram of dust. kappa is the optical depth
    per unit dust mass column, so for a line of sight with dust mass
    column Sigma_dust (g cm^-2) the optical depth is
    tau(lam) = kappa(lam) * Sigma_dust, and A(lam) = 1.086 tau(lam).

    Args:
        wav_micron (NDArray)
            wavelength grid (micron)
        radii_micron (NDArray)
            radius grid (micron) corresponding to Q
        Q (NDArray)
            efficiency array (Q_ext or Q_abs + (1 - g) Q_sca), shape
            (len(wav_micron), len(radii_micron))
        a_grid_micron (NDArray)
            radius grid (micron) where n(a) is defined; must lie within
            radii_micron
        n_a_per_gram (NDArray)
            grain size distribution on a_grid_micron, normalised to one
            gram of dust (grains per g per cm)
    Returns:
        kappa (NDArray)
            mass opacity on wav_micron (units: cm^2 g^-1)
    """
    micron_to_cm = (1.0 * um).to("cm").value  # 1e-4
    a_grid_cm = a_grid_micron * micron_to_cm

    kappa = np.zeros(wav_micron.size, dtype=float)
    for iw in range(wav_micron.size):
        # The integration grid must lie within the radii where Q is
        # defined, so never extrapolate (raise instead)
        f_lin = interp1d(radii_micron, Q[iw, :], bounds_error=True)
        q_interp = f_lin(a_grid_micron)
        # Integrand: pi a^2 Q n(a) (a in cm, n in per g per cm)
        integrand = math.pi * (a_grid_cm**2) * q_interp * n_a_per_gram
        kappa[iw] = trapezoid(integrand, a_grid_cm)

    return kappa


def table_radius_grid(radii_micron: NDArray, n_points: int) -> NDArray:
    """
    Log-spaced grain-size integration grid spanning exactly the radii
    covered by an optical table. The endpoints are set to the table's
    first and last radii so the Q interpolation never falls outside
    the table through rounding.

    Args:
        radii_micron (NDArray)
            radius grid (micron) of the optical table, in increasing order
        n_points (int)
            number of points in the integration grid
    Returns:
        grid (NDArray)
            integration grid (micron)
    """
    grid = np.geomspace(radii_micron[0], radii_micron[-1], n_points)
    grid[0], grid[-1] = radii_micron[0], radii_micron[-1]
    return grid


def interp_Q_to_grid(
    wav_orig: NDArray,
    radii_orig: NDArray,
    Q_orig: NDArray,
    wav_target: NDArray,
) -> NDArray:
    """Interpolate Qext (or Qabs, Qsca) from original wavelength
    grid to target wavelength grid.
    Args:
        wav_orig (NDArray)
            original wavelength grid (micron)
        radii_orig (NDArray)
            original radius grid (micron)
        Q_orig (NDArray)
            original Q array, shape (len(wav_orig),
            len(radii_orig))
        wav_target (NDArray)
            target wavelength grid (micron)
    Returns:
        Qw (NDArray)
            interpolated Q array on target wavelength grid,
            shape (len(wav_target), len(radii_orig))
    """
    Nr = radii_orig.size
    Qw = np.zeros((wav_target.size, Nr))
    for j in range(Nr):
        f = interp1d(
            wav_orig,
            Q_orig[:, j],
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        Qw[:, j] = f(wav_target)

    return Qw


def plot_extinction_curve(
    wav_micron: NDArray,
    curves_over_Av: Dict[str, NDArray],
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot extinction/attenuation curves A(λ)/A(V) against reference curves.

    Args:
        wav_micron (NDArray)
            wavelength grid (micron)
        curves_over_Av (dict)
            mapping of legend label -> curve A(λ)/A(V) on wav_micron
        ax (Axes)
            axes to plot on (a new figure is created if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    wav_angstrom = wav_micron * 1e4
    for label, A_over_Av in curves_over_Av.items():
        ax.plot(wav_angstrom, A_over_Av, label=label)
    curve = Calzetti2000(ampl=10.0)
    ax.plot(
        wav_angstrom,
        curve.get_tau(wav_angstrom * Angstrom),
        ls="--",
        label="Calzetti+2000",
    )
    for c in ["MW", "LMC", "SMC"]:
        curve_wd01 = GrainModels(submodel=c)
        ax.plot(
            wav_angstrom,
            curve_wd01.get_tau(wav_angstrom * Angstrom),
            ls=":",
            label=f"{c}",
        )
    ax.set_xlabel("Wavelength (AA)", fontsize=12)
    ax.set_ylabel(r"A($\lambda$) / A(V)", fontsize=12)
    ax.set_xlim(800, 1e4)
    ax.set_xticks(np.arange(1000, 10001, 1000))
    # plt.ylim(1e-3, max(5.0, np.max(A_over_Av)*2.0))
    ax.grid(which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=11, ncol=2)
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Build extinction curves from Draine
        optical files with MRN or lognormal-size components."""
    )
    parser.add_argument(
        "--mode",
        choices=["mrn", "lognormal"],
        default="lognormal",
        help="size-distribution mode",
    )
    parser.add_argument(
        "--grid-loc",
        type=str,
        default=".",
        help="directory to save the extinction curve grid",
    )
    parser.add_argument(
        "--grid-name",
        type=str,
        default="dust_extcurve_draine_li",
        help="name of the extinction curve grid file (without extension)",
    )
    parser.add_argument(
        "--small-centre",
        type=float,
        default=0.01,
        help="centre of small bin (micron)",
    )
    parser.add_argument(
        "--large-centre",
        type=float,
        default=0.1,
        help="centre of large bin (micron)",
    )
    parser.add_argument(
        "--pah-centre",
        type=float,
        default=0.001,
        help="centre of PAH bin (micron)",
    )
    parser.add_argument(
        "--sigma-lognormal",
        type=float,
        default=0.75,
        help="lognormal sigma_ln (width in ln-space)",
    )
    # Needed if mode is MRN
    parser.add_argument(
        "--a-min-mrn",
        type=float,
        default=0.001,
        help="MRN a_min (micron) when using MRN",
    )
    parser.add_argument(
        "--a-max-mrn",
        type=float,
        default=0.25,
        help="MRN a_max (micron) when using MRN",
    )
    parser.add_argument(
        "--a-min-mrn_pah",
        type=float,
        default=0.0001,
        help="MRN a_min (micron) when using MRN",
    )
    parser.add_argument(
        "--a-max-mrn_pah",
        type=float,
        default=0.01,
        help="MRN a_max (micron) when using MRN",
    )
    parser.add_argument(
        "--power-mrn",
        type=float,
        default=-3.5,
        help="MRN power-law exponent (dn/da ∝ a^power)",
    )
    parser.add_argument(
        "--plot_example",
        action="store_true",
        help="plot example extinction curves",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="show the extinction curve plot",
    )

    args = parser.parse_args()

    # Download URLs for Draine optical property files
    # We use the high-resolution versions (81 radii for
    # silicate/graphite, 30 for PAH)
    SIL_URL = "https://www.astro.princeton.edu/~draine/dust/diel/Sil_81.gz"
    saveSIL = "./Sil_81.txt"  # save
    GRA_URL = "https://www.astro.princeton.edu/~draine/dust/diel/Gra_81.gz"
    saveGRA = "./Gra_81.txt"  # save
    PAH_ion = "https://www.astro.princeton.edu/~draine/dust/diel/PAHion_30.gz"
    savePAHion = "./PAHion_30.txt"  # save
    PAH_neu = "https://www.astro.princeton.edu/~draine/dust/diel/PAHneu_30.gz"
    savePAHneu = "./PAHneu_30.txt"  # save

    # Material densities (g cm^-3)
    RHO_SIL = 3.5
    RHO_GRA = 2.24
    RHO_PAH = 2.24

    # Number of points in each grain-size integration grid. Each material
    # gets its own log-spaced grid spanning exactly the radii covered by
    # its Draine optical table (built once the tables are read)
    N_A_INT = 500

    # Download and process files
    if Path(saveSIL).exists():
        radii_sil, wav_sil, Qext_sil, Qabs_sil, Qsca_sil, gcos_sil = (
            parse_draine_file_lines(saveSIL)
        )
    else:
        sil_bytes = download_bytes(SIL_URL)
        read_gz_lines(sil_bytes, savename=saveSIL)
        radii_sil, wav_sil, Qext_sil, Qabs_sil, Qsca_sil, gcos_sil = (
            parse_draine_file_lines(saveSIL)
        )

    if Path(saveGRA).exists():
        radii_gra, wav_gra, Qext_gra, Qabs_gra, Qsca_gra, gcos_gra = (
            parse_draine_file_lines(saveGRA)
        )
    else:
        gra_bytes = download_bytes(GRA_URL)
        read_gz_lines(gra_bytes, savename=saveGRA)
        radii_gra, wav_gra, Qext_gra, Qabs_gra, Qsca_gra, gcos_gra = (
            parse_draine_file_lines(saveGRA)
        )

    if Path(savePAHion).exists():
        (
            radii_pah_ion,
            wav_pah_ion,
            Qext_pah_ion,
            Qabs_pah_ion,
            Qsca_pah_ion,
            gcos_pah_ion,
        ) = parse_draine_file_lines(savePAHion)
    else:
        pah_ion_bytes = download_bytes(PAH_ion)
        read_gz_lines(pah_ion_bytes, savename=savePAHion)
        (
            radii_pah_ion,
            wav_pah_ion,
            Qext_pah_ion,
            Qabs_pah_ion,
            Qsca_pah_ion,
            gcos_pah_ion,
        ) = parse_draine_file_lines(savePAHion)

    if Path(savePAHneu).exists():
        (
            radii_pah_neu,
            wav_pah_neu,
            Qext_pah_neu,
            Qabs_pah_neu,
            Qsca_pah_neu,
            gcos_pah_neu,
        ) = parse_draine_file_lines(savePAHneu)
    else:
        pah_neu_bytes = download_bytes(PAH_neu)
        read_gz_lines(pah_neu_bytes, savename=savePAHneu)
        (
            radii_pah_neu,
            wav_pah_neu,
            Qext_pah_neu,
            Qabs_pah_neu,
            Qsca_pah_neu,
            gcos_pah_neu,
        ) = parse_draine_file_lines(savePAHneu)

    # unify wavelength grid
    all_wav = np.unique(
        np.concatenate([wav_sil, wav_gra, wav_pah_ion, wav_pah_neu])
    )
    all_wav.sort()
    # Interpolate Qext to unified wavelength grid
    Qs_sil = interp_Q_to_grid(wav_sil, radii_sil, Qext_sil, all_wav)
    Qs_gra = interp_Q_to_grid(wav_gra, radii_gra, Qext_gra, all_wav)
    Qs_pahion = interp_Q_to_grid(
        wav_pah_ion, radii_pah_ion, Qext_pah_ion, all_wav
    )
    Qs_pahneu = interp_Q_to_grid(
        wav_pah_neu, radii_pah_neu, Qext_pah_neu, all_wav
    )

    # Interpolate components to calculate the effective attenuation efficiency
    # Qeff = Qabs + (1 - g) * Qsca
    Qabs_sil_interp = interp_Q_to_grid(wav_sil, radii_sil, Qabs_sil, all_wav)
    Qsca_sil_interp = interp_Q_to_grid(wav_sil, radii_sil, Qsca_sil, all_wav)
    g_sil = interp_Q_to_grid(wav_sil, radii_sil, gcos_sil, all_wav)
    Qeff_sil = Qabs_sil_interp + (1.0 - g_sil) * Qsca_sil_interp

    Qabs_gra_interp = interp_Q_to_grid(wav_gra, radii_gra, Qabs_gra, all_wav)
    Qsca_gra_interp = interp_Q_to_grid(wav_gra, radii_gra, Qsca_gra, all_wav)
    g_gra = interp_Q_to_grid(wav_gra, radii_gra, gcos_gra, all_wav)
    Qeff_gra = Qabs_gra_interp + (1.0 - g_gra) * Qsca_gra_interp

    Qabs_pahion_interp = interp_Q_to_grid(
        wav_pah_ion, radii_pah_ion, Qabs_pah_ion, all_wav
    )
    Qsca_pahion_interp = interp_Q_to_grid(
        wav_pah_ion, radii_pah_ion, Qsca_pah_ion, all_wav
    )
    g_pahion = interp_Q_to_grid(
        wav_pah_ion, radii_pah_ion, gcos_pah_ion, all_wav
    )
    Qeff_pahion = Qabs_pahion_interp + (1.0 - g_pahion) * Qsca_pahion_interp

    Qabs_pahneu_interp = interp_Q_to_grid(
        wav_pah_neu, radii_pah_neu, Qabs_pah_neu, all_wav
    )
    Qsca_pahneu_interp = interp_Q_to_grid(
        wav_pah_neu, radii_pah_neu, Qsca_pah_neu, all_wav
    )
    g_pahneu = interp_Q_to_grid(
        wav_pah_neu, radii_pah_neu, gcos_pah_neu, all_wav
    )
    Qeff_pahneu = Qabs_pahneu_interp + (1.0 - g_pahneu) * Qsca_pahneu_interp

    # Grain-size integration grids: one per optical table, spanning exactly
    # the radii where Q is defined
    a_grid_sil = table_radius_grid(radii_sil, N_A_INT)
    a_grid_gra = table_radius_grid(radii_gra, N_A_INT)
    a_grid_pahion = table_radius_grid(radii_pah_ion, N_A_INT)
    a_grid_pahneu = table_radius_grid(radii_pah_neu, N_A_INT)

    # Size-distribution parameters for each component: (integration grid,
    # density, then bin centre for lognormal or (a_min, a_max) for MRN).
    # Each distribution is normalised to one gram of dust on its grid, i.e.
    # truncated to the optical table's radii and renormalised.
    if args.mode == "lognormal":
        bins = {
            "s_small": (a_grid_sil, RHO_SIL, args.small_centre),
            "s_large": (a_grid_sil, RHO_SIL, args.large_centre),
            "g_small": (a_grid_gra, RHO_GRA, args.small_centre),
            "g_large": (a_grid_gra, RHO_GRA, args.large_centre),
            "pahion": (a_grid_pahion, RHO_PAH, args.pah_centre),
            "pahneu": (a_grid_pahneu, RHO_PAH, args.pah_centre),
        }
        n_per_gram = {}
        for label, (a_grid, rho, centre) in bins.items():
            frac = lognormal_mass_fraction(
                a_grid[0], a_grid[-1], centre, args.sigma_lognormal
            )
            print(
                f"{label}: {100 * frac:.2f}% of the lognormal mass lies in "
                f"the table range [{a_grid[0]:.4g}, {a_grid[-1]:.4g}] um; "
                "renormalised to 100%"
            )
            n_per_gram[label] = build_lognormal_component(
                a_grid, centre, args.sigma_lognormal, 1.0, rho
            )
    else:
        # choose MRN bin bounds around the centers specified
        small_a_min = args.a_min_mrn
        small_a_max = args.small_centre + (args.small_centre - args.a_min_mrn)
        large_a_min = small_a_max
        large_a_max = args.a_max_mrn
        pah_a_min = args.a_min_mrn_pah
        pah_a_max = args.a_max_mrn_pah

        bins = {
            "s_small": (a_grid_sil, RHO_SIL, (small_a_min, small_a_max)),
            "s_large": (a_grid_sil, RHO_SIL, (large_a_min, large_a_max)),
            "g_small": (a_grid_gra, RHO_GRA, (small_a_min, small_a_max)),
            "g_large": (a_grid_gra, RHO_GRA, (large_a_min, large_a_max)),
            "pahion": (a_grid_pahion, RHO_PAH, (pah_a_min, pah_a_max)),
            "pahneu": (a_grid_pahneu, RHO_PAH, (pah_a_min, pah_a_max)),
        }
        n_per_gram = {}
        for label, (a_grid, rho, (a_min, a_max)) in bins.items():
            lo, hi = max(a_min, a_grid[0]), min(a_max, a_grid[-1])
            print(
                f"{label}: MRN bin [{a_min:.4g}, {a_max:.4g}] um, "
                f"integrated over [{lo:.4g}, {hi:.4g}] um (table range)"
            )
            n_per_gram[label] = build_mrn_component(
                a_grid, a_min, a_max, args.power_mrn, 1.0, rho
            )

    a_grids = {label: b[0] for label, b in bins.items()}

    # Two kinds of curve are written, differing only in the efficiency:
    #   extinction_curves:  Q_ext. Any scattered photon leaves the beam,
    #       appropriate for a point source behind a dust screen.
    #   attenuation_curves: Q_abs + (1 - g) Q_sca. Forward-scattered light
    #       is kept in the beam (transport approximation), a cheap stand-in
    #       for unresolved scattering geometry.
    CURVE_TYPES = {
        "extinction_curves": {
            "efficiency": "Q_ext",
            "description": "Extinction mass opacity computed with Q_ext",
        },
        "attenuation_curves": {
            "efficiency": "Q_abs + (1 - g) Q_sca",
            "description": (
                "Effective attenuation mass opacity computed with "
                "Q_abs + (1 - g) Q_sca"
            ),
        },
    }

    # Grain components: label -> dataset key, description, optical-table
    # radii and efficiency for each curve type
    components = {
        "g_small": {
            "key": f"graphite_a{args.small_centre}um",
            "desc": f"graphite small (bin centre {args.small_centre} um)",
            "radii": radii_gra,
            "Q": {"extinction_curves": Qs_gra, "attenuation_curves": Qeff_gra},
        },
        "g_large": {
            "key": f"graphite_a{args.large_centre}um",
            "desc": f"graphite large (bin centre {args.large_centre} um)",
            "radii": radii_gra,
            "Q": {"extinction_curves": Qs_gra, "attenuation_curves": Qeff_gra},
        },
        "s_small": {
            "key": f"silicate_a{args.small_centre}um",
            "desc": f"silicate small (bin centre {args.small_centre} um)",
            "radii": radii_sil,
            "Q": {"extinction_curves": Qs_sil, "attenuation_curves": Qeff_sil},
        },
        "s_large": {
            "key": f"silicate_a{args.large_centre}um",
            "desc": f"silicate large (bin centre {args.large_centre} um)",
            "radii": radii_sil,
            "Q": {"extinction_curves": Qs_sil, "attenuation_curves": Qeff_sil},
        },
        "pahion": {
            "key": f"pahionised_a{args.pah_centre}um",
            "desc": f"ionised PAH (bin centre {args.pah_centre} um)",
            "radii": radii_pah_ion,
            "Q": {
                "extinction_curves": Qs_pahion,
                "attenuation_curves": Qeff_pahion,
            },
        },
        "pahneu": {
            "key": f"pahneutral_a{args.pah_centre}um",
            "desc": f"neutral PAH (bin centre {args.pah_centre} um)",
            "radii": radii_pah_neu,
            "Q": {
                "extinction_curves": Qs_pahneu,
                "attenuation_curves": Qeff_pahneu,
            },
        },
    }

    # kappa[curve_type][component] is the mass opacity (cm^2 per g of dust)
    # on all_wav. For a dust mass column Sigma_dust of that component,
    # tau(lam) = kappa(lam) * Sigma_dust.
    kappa = {
        curve_type: {
            label: calculate_kappa(
                all_wav,
                comp["radii"],
                comp["Q"][curve_type],
                a_grids[label],
                n_per_gram[label],
            )
            for label, comp in components.items()
        }
        for curve_type in CURVE_TYPES
    }

    # Filter and write
    if args.mode == "lognormal":
        avoid = "mrn"
    else:
        avoid = "lognormal"
    args_dict = vars(args)
    filtered_args = {
        k: str(v)
        for k, v in args_dict.items()
        if v is not None and avoid not in k.lower() and "plot" not in k.lower()
    }

    header = {}
    for key, value in filtered_args.items():
        header[key] = value

    grid_name = (
        f"{args.grid_name}_{args.mode}"
        f"_asmall{str(args.small_centre).replace('.', 'p')}"
        f"_alarge{str(args.large_centre).replace('.', 'p')}"
        f"_apah{str(args.pah_centre).replace('.', 'p')}"
    )

    # Ensure output directory exists
    Path(args.grid_loc).mkdir(parents=True, exist_ok=True)

    # Create GridFile to store outputs
    out_grid = GridFile(f"{args.grid_loc}/{grid_name}.hdf5")

    model = {
        "model_name": "Draine & Li dust extinction curves",
        "grains": "Graphite, Silicates, PAH",
        "grain bins": 2,
        "url": "https://www.astro.princeton.edu/~draine/dust/dust.diel.html",
    }
    model.update(header)

    print(model)

    # Write the model metadata
    out_grid.write_model_metadata(model)

    # Write out the extinction and attenuation mass opacities and their
    # wavelengths, one group per curve type. There are no grid axes: the
    # optical depth is just tau(lam) = sum_c kappa_c(lam) * Sigma_dust,c.
    for curve_type, info in CURVE_TYPES.items():
        out_grid.write_dataset(
            key=f"{curve_type}/wavelength",
            data=all_wav * um,
            description=f"Wavelength of the {curve_type} grid",
            log_on_read=False,
        )
        for label, comp in components.items():
            out_grid.write_dataset(
                key=f"{curve_type}/{comp['key']}",
                data=kappa[curve_type][label] * cm**2 / g,
                description=(
                    f"{info['description']} for the {comp['desc']} grain "
                    "component: optical depth per unit dust mass column "
                    "(tau = kappa * Sigma_dust; A = 1.086 tau)"
                ),
                log_on_read=False,
            )
        # Record which efficiency was used (the group now exists)
        out_grid.write_attribute(curve_type, "efficiency", info["efficiency"])

    print(f"Saved extinction curve grid to:{args.grid_loc}/{grid_name}.hdf5")

    if args.plot_example:
        # V wavelength for A(V)
        V_WAVELENGTH = 0.55  # micron
        # Plot example extinction curves for specific DTG values
        iv = np.argmin(np.abs(all_wav - V_WAVELENGTH))

        # Same DTG for both small and large grains
        DTG_sil = 0.006  # example dust-to-gas ratio for plotting
        DTG_gra = 0.003  # example dust-to-gas ratio for plotting

        DTG_PAH = 0.0001  # example dust-to-gas ratio for plotting

        # Gas mass per H (g) with helium correction, used only to turn the
        # example DTGs into dust mass per H for the Av/NH comparison
        MU = 1.4
        M_H = 1.6738e-24  # in grams
        GAS_MASS_PER_H = MU * M_H  # g per H

        # Dust mass per H (g per H) of each component
        dust_mass_per_H = {
            "s_small": DTG_sil * GAS_MASS_PER_H,
            "s_large": DTG_sil * GAS_MASS_PER_H,
            "g_small": DTG_gra * GAS_MASS_PER_H,
            "g_large": DTG_gra * GAS_MASS_PER_H,
            "pahion": DTG_PAH * GAS_MASS_PER_H,
            "pahneu": DTG_PAH * GAS_MASS_PER_H,
        }
        MAG_PER_TAU = 2.5 * np.log10(np.e)  # A = 1.086 tau
        plot_labels = {
            "extinction_curves": r"Extinction ($Q_{\rm ext}$)",
            "attenuation_curves": (
                r"Attenuation ($Q_{\rm abs} + (1-g)\,Q_{\rm sca}$)"
            ),
        }
        curves_over_Av = {}
        for curve_type in CURVE_TYPES:
            # A(lam)/N_H in mag cm^2 per H
            A_total = MAG_PER_TAU * sum(
                kappa[curve_type][label] * m
                for label, m in dust_mass_per_H.items()
            )
            A_V = A_total[iv]
            curves_over_Av[plot_labels[curve_type]] = A_total / A_V
            print(f"{curve_type}: Av/NH = {A_V:.3e} mag cm^2")
        print("Milky Way sightline (extinction), Rv=3.1:")
        print("Av/NH = 5.3E-22 mag cm^2")
        out_prefix = "dust_extinction_curve"
        fig, axs = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
        axs = axs.ravel()
        plot_extinction_curve(all_wav, curves_over_Av, axs[0])

        micron_to_cm = (1.0 * um).to("cm").value  # 1e-4

        # (labels summed, integration grid, legend label, line style)
        size_dists = [
            (("g_small", "g_large"), a_grid_gra, "Graphite", {"c": "black"}),
            (("s_small", "s_large"), a_grid_sil, "Silicate", {"c": "blue"}),
            (("pahion",), a_grid_pahion, "PAH ionised", {"c": "olive"}),
            (("pahneu",), a_grid_pahneu, "PAH neutral", {"c": "red"}),
        ]
        for labels, a_grid, name, style in size_dists:
            # n(a) per H per cm
            n_tot = sum(
                n_per_gram[label] * dust_mass_per_H[label] for label in labels
            )
            ok = n_tot != 0
            ls = "dotted" if name.startswith("PAH") else "dashed"
            lw = 4 if name == "PAH ionised" else None
            axs[1].loglog(
                a_grid[ok],
                ((a_grid[ok] * micron_to_cm) ** 4) * n_tot[ok],
                label=name,
                ls=ls,
                lw=lw,
                **style,
            )

        axs[1].set_ylim(1e-30, 1e-26)
        axs[1].set_xlabel("a (um)", fontsize=12)
        axs[1].set_ylabel(r"a$^{4}$ n(a) / (grains/H cm$^3$)", fontsize=12)
        axs[1].grid(which="both", ls="--", alpha=0.4)
        axs[1].legend(fontsize=11, ncol=2)

        title = f"""Extinction ({args.mode}) — sil DTG={DTG_sil:.3g},
        gra DTG={DTG_gra:.3g}, PAH DTG={DTG_PAH:.3g}"""
        fig.suptitle(title)

        plt.tight_layout()
        png_name = f"{out_prefix}_{args.mode}.png"
        plt.savefig(png_name, dpi=200)
        print(f"Saved plot: {png_name}")
        if args.show_plot:
            print("Showing plot")
            plt.show()
        else:
            print("Closing plot")
            plt.close()
