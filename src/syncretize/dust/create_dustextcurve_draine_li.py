#!/usr/bin/env python3
"""
Create dust mass opacities (extinction curves) from Draine optical properties.

Parse Draine's Sil_81, Gra_81, PAHion_30 and PAHneu_30 tables and compute,
on a grid of grain-size distribution parameters, the mass opacity of each
material listed in a YAML parameter file,

    kappa(lam) = integral pi a^2 Q(a, lam) n(a) da    (cm^2 per g of dust),

with n(a) normalised to one gram of dust over the radii where Q is tabulated.

Truncation to the optical table's radii (silicate, graphite: 0.001 - 10 um;
PAHs: 3.548e-4 - 0.01 um): Q is never extrapolated in radius. Any part of a
size distribution outside the table is cut off and the remainder rescaled
to one gram, so the full dust mass is kept but the mass outside the table
is effectively given the opacity per gram of the grains inside it. kappa is
therefore biased for distributions with significant mass outside the table,
e.g. a lognormal centred at a table edge has half its mass outside it
(the default PAH lognormal, a = 0.001 um and sigma = 0.75, has 8.5%).
How much is affected is stored in every output:
  - lognormal: untruncated_mass_fraction, the fraction of each
    untruncated lognormal's mass inside the table (1 = no truncation).
  - powerlaw: integrated_a_min/integrated_a_max, the radius range actually
    integrated after clipping [a_min, a_max] to the table.
Check these before using cells near or beyond the table edges.

The extinction scales linearly with the dust mass: for a line of sight with
dust mass column Sigma_dust,m of each material m, the optical depth is
tau(lam) = sum_m kappa_m(lam) Sigma_dust,m and A(lam) = 1.086 tau(lam).

Two opacities are written for every material:
  - extinction: computed with Q_ext, i.e. every scattered photon is removed
    from the beam (point source behind a dust screen).
  - attenuation: computed with Q_abs + (1 - g) Q_sca, i.e. forward-scattered
    light is kept in the beam, a simple stand-in for unresolved scattering
    geometry.

One parameter file makes one grid: a single size-distribution shape, the
materials of one family (grains: silicate, graphite; pah: pah_ionised,
pah_neutral, whose optical tables cover different radii and wavelengths)
and axes shared by all of them. Shapes (see params/*.yaml):
  - lognormal: axes (a, sigma). The lognormal is defined in number,
        n(a) = M0 / (sqrt(2 pi) sigma a) exp(-(ln a - mu)^2 / (2 sigma^2)),
    and the a axis is its mass-median radius (micron), so
    mu = ln(a / 1 micron) - 3 sigma^2 and the number-median radius is
    a_n = e^mu = a exp(-3 sigma^2).
  - powerlaw: axes (a_min, a_max), dn/da propto a^power on [a_min, a_max]
    (micron). Covers MRN-like size ranges and discrete grain-size bins.
    Cells with a_min >= a_max, or with no overlap with the optical table,
    are NaN.

Output layout (readable by synthesizer's Grid):
  attrs axes                       axis names, in array-dimension order
  /axes/<axis>                     axis values (radii in micron, read in log)
  /wavelength                      (micron), shared by both curve types
  /<type>_curves/wavelength        soft link to /wavelength
  /extinction_curves/<material>    kappa from Q_ext, (*axes, n_wavelength)
  /attenuation_curves/<material>   kappa from Q_abs + (1 - g) Q_sca
  lognormal: /untruncated_mass_fraction/<material>, (*axes)
  powerlaw:  /integrated_a_min/<material>, /integrated_a_max/<material>,
             the radius range actually integrated in each cell, (*axes)

Usage:
  python create_dustextcurve_draine_li.py \
      --paramfile params/draine_li_grains_lognormal.yaml \
      [--plot-example] [--show-plot]
"""

import argparse
import gzip
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
import yaml
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.special import erf
from synthesizer import GRID_DIR
from synthesizer.emission_models.attenuation import Calzetti2000, GrainModels
from unyt import Angstrom, cm, dimensionless, g, um

from syncretize.grid_io import GridFile

DRAINE_URL = "https://www.astro.princeton.edu/~draine/dust/diel"
MODEL_URL = "https://www.astro.princeton.edu/~draine/dust/dust.diel.html"

# Draine optical tables: table name, dataset-name label, default material
# density (g cm^-3) and family. Densities can be overridden in the
# parameter file. A grid holds materials of one family only.
MATERIALS = {
    "silicate": {
        "table": "Sil_81",
        "label": "silicate",
        "rho": 3.5,
        "family": "grains",
    },
    "graphite": {
        "table": "Gra_81",
        "label": "graphite",
        "rho": 2.24,
        "family": "grains",
    },
    "pah_ionised": {
        "table": "PAHion_30",
        "label": "pahionised",
        "rho": 2.24,
        "family": "pah",
    },
    "pah_neutral": {
        "table": "PAHneu_30",
        "label": "pahneutral",
        "rho": 2.24,
        "family": "pah",
    },
}

# The two efficiencies written for every material
CURVE_TYPES = {
    "extinction": {
        "efficiency": "Q_ext",
        "description": "Extinction mass opacity computed with Q_ext",
    },
    "attenuation": {
        "efficiency": "Q_abs + (1 - g) Q_sca",
        "description": (
            "Effective attenuation mass opacity computed with "
            "Q_abs + (1 - g) Q_sca"
        ),
    },
}

MICRON_TO_CM = (1.0 * um).to("cm").value  # 1e-4

# Grid axes of each shape, in array-dimension order
SHAPE_AXES = {
    "lognormal": ("a", "sigma"),
    "powerlaw": ("a_min", "a_max"),
}
# Radius axes (micron): log-spaced when given as {min, max, n}, stored in
# micron and interpolated in log by synthesizer. Other axes are linear.
RADIUS_AXES = {"a", "a_min", "a_max"}
AXIS_DESCRIPTIONS = {
    "a": (
        "Mass-median grain radius of the lognormal (micron). The lognormal "
        "is defined in number, n(a) propto (1/a) exp(-(ln a - mu)^2 / "
        "(2 sigma^2)), so mu = ln(a / 1 micron) - 3 sigma^2 and the "
        "number-median radius is a_n = e^mu = a exp(-3 sigma^2)"
    ),
    "sigma": "Width of the lognormal in ln(a)",
    "a_min": "Lower radius cut-off of the power law (micron)",
    "a_max": "Upper radius cut-off of the power law (micron)",
}
TOP_LEVEL_KEYS = {
    "output",
    "data_dir",
    "n_a_int",
    "densities",
    "shape",
    "materials",
    "axes",
    "power",
}


def download_bytes(url: str, timeout: float = 30.0) -> bytes:
    """
    Download bytes from URL with timeout.

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


def read_gz_lines(gzbytes: bytes, savename: str) -> str:
    """
    Decompress gzipped bytes and save the text to savename.

    Args:
        gzbytes (bytes)
            Gzipped content
        savename (string)
            Filename to save uncompressed text
    Returns:
        the uncompressed text
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


def load_optical_table(material: str, data_dir: str) -> Dict[str, NDArray]:
    """
    Load a material's Draine optical table, downloading and caching the
    uncompressed text in data_dir if it is not already there.

    Args:
        material (string)
            key of MATERIALS
        data_dir (string)
            directory where the uncompressed tables are cached
    Returns:
        dict with radii (micron), wav (micron), Qext, Qabs, Qsca and g, the
        last four of shape (len(wav), len(radii))
    """
    table = MATERIALS[material]["table"]
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    savename = str(Path(data_dir) / f"{table}.txt")
    if not Path(savename).exists():
        gz_bytes = download_bytes(f"{DRAINE_URL}/{table}.gz")
        read_gz_lines(gz_bytes, savename=savename)
    radii, wav, Qext, Qabs, Qsca, gcos = parse_draine_file_lines(savename)
    return {
        "radii": radii,
        "wav": wav,
        "Qext": Qext,
        "Qabs": Qabs,
        "Qsca": Qsca,
        "g": gcos,
    }


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


def efficiencies_on_grid(
    table: Dict[str, NDArray], wav: NDArray
) -> Dict[str, NDArray]:
    """
    Interpolate a material's efficiencies onto a common wavelength grid and
    form the efficiency used for each curve type.

    Args:
        table (dict)
            output of load_optical_table
        wav (NDArray)
            target wavelength grid (micron)
    Returns:
        dict curve type -> Q on (len(wav), len(table radii))
    """
    radii, wav_t = table["radii"], table["wav"]
    Qext = interp_Q_to_grid(wav_t, radii, table["Qext"], wav)
    Qabs = interp_Q_to_grid(wav_t, radii, table["Qabs"], wav)
    Qsca = interp_Q_to_grid(wav_t, radii, table["Qsca"], wav)
    gcos = interp_Q_to_grid(wav_t, radii, table["g"], wav)
    return {
        "extinction": Qext,
        "attenuation": Qabs + (1.0 - gcos) * Qsca,
    }


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


def trapezoid_weights(x: NDArray) -> NDArray:
    """
    Weights w such that sum(w * y) is the trapezoid-rule integral of y(x).

    Args:
        x (NDArray)
            sample points, increasing
    Returns:
        w (NDArray)
            quadrature weights, same shape as x
    """
    dx = np.diff(x)
    w = np.zeros_like(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def restricted_trapezoid_weights(
    x_micron: NDArray, lo_micron: Any, hi_micron: Any
) -> NDArray:
    """
    Trapezoid weights (in cm) for integrating over [lo, hi] only, where lo
    and hi are nodes of x. Segments outside [lo, hi] get no weight, so a
    sharp cut-off at a node is integrated without spilling half a segment
    beyond it. lo and hi broadcast against each other; cells with lo >= hi
    (or NaN) get all-zero weights.

    Args:
        x_micron (NDArray)
            sample points (micron), increasing
        lo_micron, hi_micron (float or NDArray)
            integration limits (micron), nodes of x_micron
    Returns:
        w (NDArray)
            weights, shape broadcast(lo, hi) + (len(x_micron),)
    """
    lo = np.asarray(lo_micron, dtype=float)[..., np.newaxis]
    hi = np.asarray(hi_micron, dtype=float)[..., np.newaxis]
    inside = (x_micron[:-1] >= lo) & (x_micron[1:] <= hi)
    half = 0.5 * np.diff(x_micron * MICRON_TO_CM) * inside
    w = np.zeros(half.shape[:-1] + (x_micron.size,))
    w[..., :-1] += half
    w[..., 1:] += half
    return w


def integration_grid(
    radii_micron: NDArray, n_points: int, extra_nodes: Any = ()
) -> NDArray:
    """
    Integration grid for a material: n_points log-spaced radii spanning the
    optical table, plus any extra nodes (e.g. power-law cut-offs) inside it.

    Args:
        radii_micron (NDArray)
            radius grid (micron) of the optical table
        n_points (int)
            number of log-spaced points
        extra_nodes (array-like)
            radii (micron) to add as nodes; those outside the table are
            ignored
    Returns:
        grid (NDArray)
            sorted integration grid (micron)
    """
    grid = table_radius_grid(radii_micron, n_points)
    extra = np.asarray(extra_nodes, dtype=float).ravel()
    extra = extra[(extra > grid[0]) & (extra < grid[-1])]
    return np.unique(np.concatenate([grid, extra]))


def build_kernel(
    radii_micron: NDArray, Q: NDArray, a_grid_micron: NDArray
) -> NDArray:
    """
    Kernel K[lam, a] = pi a^2 Q(a, lam) on the integration grid, so that
    kappa = (n * w) @ K.T with w the quadrature weights.

    Args:
        radii_micron (NDArray)
            radius grid (micron) of the optical table
        Q (NDArray)
            efficiency on (n_wavelength, len(radii_micron))
        a_grid_micron (NDArray)
            integration grid (micron); must lie within radii_micron
    Returns:
        K (NDArray)
            kernel (cm^2), shape (n_wavelength, len(a_grid_micron))
    """
    # Never extrapolate Q in radius (raise instead)
    Q_a = interp1d(radii_micron, Q, axis=1, bounds_error=True)(a_grid_micron)
    a_cm = a_grid_micron * MICRON_TO_CM
    return math.pi * a_cm**2 * Q_a


# Size distributions
def lognormal_shape(
    a_grid_micron: NDArray, mass_median_micron: Any, sigma: Any
) -> NDArray:
    """
    Unnormalised lognormal size distribution defined in number,
    n(a) propto (1 / a) exp(-(ln a - mu)^2 / (2 sigma^2)), with
    mu = ln(a_m) - 3 sigma^2 for mass-median radius a_m. a_m and sigma
    broadcast against each other; the radius axis is appended last.

    Args:
        a_grid_micron (NDArray)
            radius grid (micron)
        mass_median_micron (float or NDArray)
            mass-median radius (micron)
        sigma (float or NDArray)
            width in ln(a)
    Returns:
        shape (NDArray)
            unnormalised n(a), shape broadcast(a_m, sigma) + (n_a,)
    """
    a_m = np.asarray(mass_median_micron, dtype=float)[..., np.newaxis]
    sig = np.asarray(sigma, dtype=float)[..., np.newaxis]
    if np.any(sig <= 0) or np.any(a_m <= 0):
        raise ValueError("mass median and sigma must be > 0")
    mu = np.log(a_m) - 3 * sig**2
    x = np.log(a_grid_micron) - mu
    return np.exp(-0.5 * (x / sig) ** 2) / a_grid_micron


def lognormal_mass_fraction(
    a_lo_micron: float,
    a_hi_micron: float,
    mass_median_micron: Any,
    sigma: Any,
) -> NDArray:
    """
    Fraction of an untruncated lognormal's mass lying in [a_lo, a_hi].
    The mass per ln(a) is a Gaussian in ln(a) of width sigma centred on the
    mass-median radius.

    Args:
        a_lo_micron, a_hi_micron (float)
            radius range (micron)
        mass_median_micron (float or NDArray)
            mass-median radius (micron)
        sigma (float or NDArray)
            width in ln(a)
    Returns:
        mass fraction in [a_lo, a_hi], broadcast shape of the inputs
    """
    sig = np.asarray(sigma, dtype=float)
    ln_am = np.log(np.asarray(mass_median_micron, dtype=float))
    z_lo = (math.log(a_lo_micron) - ln_am) / (sig * math.sqrt(2))
    z_hi = (math.log(a_hi_micron) - ln_am) / (sig * math.sqrt(2))
    return 0.5 * (erf(z_hi) - erf(z_lo))


def axis_values(spec: Any, name: str) -> NDArray:
    """
    Axis values from {min, max, n} (log-spaced for radius axes, linear
    otherwise) or from an explicit value or increasing list.

    Args:
        spec (dict, list or float)
            axis specification from the parameter file
        name (string)
            axis name
    Returns:
        values (NDArray)
    """
    what = f"axes.{name}"
    if isinstance(spec, dict):
        if set(spec) != {"min", "max", "n"}:
            raise ValueError(f"{what} needs exactly min, max and n")
        lo, hi, n = float(spec["min"]), float(spec["max"]), int(spec["n"])
        if n < 2 or not hi > lo or lo <= 0:
            raise ValueError(f"{what} needs n >= 2 and max > min > 0")
        spacing = np.geomspace if name in RADIUS_AXES else np.linspace
        values = spacing(lo, hi, n)
        values[0], values[-1] = lo, hi
        return values
    values = np.atleast_1d(np.asarray(spec, dtype=float))
    if values.ndim != 1 or values.size < 1 or np.any(np.diff(values) <= 0):
        raise ValueError(
            f"{what} must be a value or a strictly increasing list"
        )
    if np.any(values <= 0):
        raise ValueError(f"{what} values must be > 0")
    return values


def parse_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the parameter file and build the grid axes.

    Args:
        params (dict)
            parsed YAML parameter file
    Returns:
        dict with shape, materials, family, axes (name -> values, in
        array-dimension order), power (powerlaw only) and densities
    """
    unknown = set(params) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown top-level parameter file keys: {sorted(unknown)}"
        )

    shape = params.get("shape")
    if shape not in SHAPE_AXES:
        raise ValueError(f"shape must be one of {sorted(SHAPE_AXES)}")

    materials = params.get("materials")
    if not isinstance(materials, list) or not materials:
        raise ValueError("materials must be a non-empty list")
    bad = [m for m in materials if m not in MATERIALS]
    if bad:
        raise ValueError(
            f"Unknown materials {bad}; must be one of {sorted(MATERIALS)}"
        )
    if len(set(materials)) != len(materials):
        raise ValueError("materials must not repeat")
    families = {MATERIALS[m]["family"] for m in materials}
    if len(families) > 1:
        groups = {
            f: [m for m in MATERIALS if MATERIALS[m]["family"] == f]
            for f in sorted({v["family"] for v in MATERIALS.values()})
        }
        raise ValueError(
            "materials must all be from one family, since their optical "
            f"tables share radii and wavelengths: {groups}. Use one "
            "parameter file per family."
        )

    axes_spec = params.get("axes")
    expected = SHAPE_AXES[shape]
    if not isinstance(axes_spec, dict) or set(axes_spec) != set(expected):
        raise ValueError(f"axes for shape {shape} must be exactly {expected}")
    axes = {name: axis_values(axes_spec[name], name) for name in expected}

    if shape == "powerlaw":
        if "power" not in params:
            raise ValueError("power is required for shape powerlaw")
        power = float(params["power"])
    else:
        if "power" in params:
            raise ValueError("power is only used by shape powerlaw")
        power = None

    densities = params.get("densities") or {}
    unknown = set(densities) - set(MATERIALS)
    if unknown:
        raise ValueError(f"Unknown materials in densities: {sorted(unknown)}")

    return {
        "shape": shape,
        "materials": materials,
        "family": families.pop(),
        "axes": axes,
        "power": power,
        "densities": densities,
    }


def size_distribution(
    cfg: Dict[str, Any], radii: NDArray, n_a_int: int
) -> Dict[str, Any]:
    """
    Unnormalised size distributions of every grid cell for a material, with
    the quadrature weights used to integrate them.

    Args:
        cfg (dict)
            output of parse_params
        radii (NDArray)
            radius grid (micron) of the material's optical table
        n_a_int (int)
            number of log-spaced integration points
    Returns:
        dict with a_grid (micron), n (unnormalised n(a), broadcastable to
        (*axes, n_a)), w (weights, (*axes, n_a)), valid (cells that
        should have a finite opacity, (*axes)) and extra (name ->
        ((*axes) array, units, description)) datasets to write
    """
    lo, hi = float(radii[0]), float(radii[-1])
    if cfg["shape"] == "lognormal":
        a_grid = integration_grid(radii, n_a_int)
        a_m, sig = np.meshgrid(
            cfg["axes"]["a"], cfg["axes"]["sigma"], indexing="ij"
        )
        n = lognormal_shape(a_grid, a_m, sig)
        w = np.broadcast_to(trapezoid_weights(a_grid * MICRON_TO_CM), n.shape)
        frac = lognormal_mass_fraction(lo, hi, a_m, sig)
        return {
            "a_grid": a_grid,
            "n": n,
            "w": w,
            # an untruncated lognormal has mass at every radius, but cells
            # far outside the table can underflow to zero (checked later)
            "valid": None,
            "extra": {
                "untruncated_mass_fraction": (
                    frac,
                    dimensionless,
                    "Fraction of the untruncated lognormal's mass within "
                    "the optical table's radius range (before "
                    "renormalisation onto it)",
                ),
            },
        }

    # powerlaw: put every cut-off on the integration grid so each cell is
    # integrated exactly between its own limits
    a_min_axis, a_max_axis = cfg["axes"]["a_min"], cfg["axes"]["a_max"]
    a_grid = integration_grid(
        radii, n_a_int, np.concatenate([a_min_axis, a_max_axis])
    )
    a_min, a_max = np.meshgrid(a_min_axis, a_max_axis, indexing="ij")
    int_lo, int_hi = np.maximum(a_min, lo), np.minimum(a_max, hi)
    valid = (a_min < a_max) & (int_lo < int_hi)
    int_lo = np.where(valid, int_lo, np.nan)
    int_hi = np.where(valid, int_hi, np.nan)
    return {
        "a_grid": a_grid,
        "n": a_grid ** cfg["power"],
        "w": restricted_trapezoid_weights(a_grid, int_lo, int_hi),
        "valid": valid,
        "extra": {
            "integrated_a_min": (
                int_lo,
                um,
                "Lower limit of the radius range integrated in each cell: "
                "a_min clipped to the optical table (NaN for empty cells)",
            ),
            "integrated_a_max": (
                int_hi,
                um,
                "Upper limit of the radius range integrated in each cell: "
                "a_max clipped to the optical table (NaN for empty cells)",
            ),
        },
    }


def compute_opacities(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the extinction and attenuation mass opacities of every material
    on the parameter file's grid.

    Args:
        params (dict)
            parsed YAML parameter file
    Returns:
        dict with wav (micron), shape, family, axes, power and materials
        (material -> dict with kappa (curve type -> (*axes, n_wavelength)
        in cm^2 g^-1), a_grid (micron), n_per_gram ((*axes, n_a), grains
        per gram per cm), rho, table range and extra datasets)
    """
    cfg = parse_params(params)
    data_dir = params.get("data_dir", ".")
    n_a_int = int(params.get("n_a_int", 500))
    grid_shape = tuple(v.size for v in cfg["axes"].values())

    tables = {m: load_optical_table(m, data_dir) for m in cfg["materials"]}

    # Common wavelength grid: union of this family's tables
    wav = np.unique(np.concatenate([t["wav"] for t in tables.values()]))

    out: Dict[str, Any] = {}
    for material in cfg["materials"]:
        table = tables[material]
        radii = table["radii"]
        rho = float(cfg["densities"].get(material, MATERIALS[material]["rho"]))
        dist = size_distribution(cfg, radii, n_a_int)
        a_grid = dist["a_grid"]

        # Normalise every cell to one gram of dust with its own weights
        a_cm = a_grid * MICRON_TO_CM
        grain_mass = (4.0 / 3.0) * math.pi * rho * a_cm**3
        nw = dist["n"] * dist["w"]
        mass = nw @ grain_mass
        has_mass = mass > 0
        if dist["valid"] is not None and np.any(dist["valid"] & ~has_mass):
            raise RuntimeError(
                f"{material}: cells with a valid radius range have no mass "
                "on the integration grid"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(has_mass, 1.0 / mass, np.nan)
        nw_per_gram = nw * scale[..., np.newaxis]
        n_per_gram = (
            np.where(dist["w"] > 0, dist["n"], 0.0) * scale[..., np.newaxis]
        )

        # kappa(lam) = integral pi a^2 Q(a, lam) n(a) da. With the
        # trapezoid rule this is sum_i w_i pi a_i^2 Q(a_i, lam) n(a_i).
        # pi a^2 Q is the same for every cell and is the per-material
        # kernel K[lam, i] (build_kernel); the weights w_i depend on the
        # cell (each power-law cell only integrates between its own
        # limits) and are folded into n. The integral for every cell is
        # then one matrix product, (n w) (*axes, n_a) @ K.T (n_a,
        # n_wavelength), rather than a loop over cells or a
        # (*axes, n_wavelength, n_a) array (GBs for a typical grid). The
        # mass normalisation above uses the same weights, so the
        # normalised mass and the kappa integral share one quadrature.
        Q = efficiencies_on_grid(table, wav)
        kappa = {
            ct: nw_per_gram @ build_kernel(radii, Q[ct], a_grid).T
            for ct in CURVE_TYPES
        }

        n_nan = int(np.sum(~has_mass))
        print(
            f"{material}: {grid_shape} grid on table radii "
            f"[{radii[0]:.4g}, {radii[-1]:.4g}] um, {n_nan} empty (NaN) "
            "cells"
        )
        if cfg["shape"] == "lognormal":
            frac = dist["extra"]["untruncated_mass_fraction"][0]
            print(
                f"  untruncated mass fraction in table range "
                f"{100 * frac.min():.3g}%-{100 * frac.max():.3g}%"
            )
        out[material] = {
            "kappa": kappa,
            "a_grid": a_grid,
            "n_per_gram": n_per_gram,
            "rho": rho,
            "table_a_min_um": float(radii[0]),
            "table_a_max_um": float(radii[-1]),
            "extra": dist["extra"],
        }

    return {
        "wav": wav,
        "shape": cfg["shape"],
        "family": cfg["family"],
        "axes": cfg["axes"],
        "power": cfg["power"],
        "materials": out,
    }


def write_grid(
    path: str, params: Dict[str, Any], results: Dict[str, Any]
) -> None:
    """
    Write the opacities to an HDF5 file (layout in the module docstring).

    Args:
        path (string)
            output file path
        params (dict)
            parsed YAML parameter file (stored in the Model metadata)
        results (dict)
            output of compute_opacities
    """
    model: Dict[str, Any] = {
        "model_name": "Draine & Li dust mass opacities",
        "url": MODEL_URL,
        "opacity_units": "cm**2/g",
        "usage": (
            "tau(lam) = sum_m kappa_m(lam) * Sigma_dust,m, where "
            "Sigma_dust,m is the dust mass column of material m; "
            "A(lam) = 1.086 tau(lam)"
        ),
        "shape": results["shape"],
        "family": results["family"],
        "materials": ", ".join(results["materials"]),
        "paramfile": yaml.safe_dump(params, sort_keys=False),
    }
    if results["shape"] == "lognormal":
        model["lognormal_definition"] = (
            "n(a) = M0 / (sqrt(2 pi) sigma a) exp(-(ln a - mu)^2 / "
            "(2 sigma^2)); axis a is the mass-median radius, "
            "mu = ln(a / 1 micron) - 3 sigma^2"
        )
    else:
        model["power"] = results["power"]
        model["powerlaw_definition"] = (
            "dn/da propto a^power on [a_min, a_max]; NaN where "
            "a_min >= a_max or the range misses the optical table"
        )

    # How the size distributions are truncated to the optical table (the
    # materials of a family share the same table radii)
    first = next(iter(results["materials"].values()))
    check = (
        "untruncated_mass_fraction (fraction of each untruncated "
        "lognormal's mass inside the table, 1 = no truncation)"
        if results["shape"] == "lognormal"
        else "integrated_a_min/integrated_a_max (the radius range actually "
        "integrated after clipping [a_min, a_max] to the table)"
    )
    model["truncation"] = (
        "Q is never extrapolated in radius. Any part of a size "
        "distribution outside the optical table's radii "
        f"[{first['table_a_min_um']:.4g}, {first['table_a_max_um']:.4g}] um "
        "is cut off and the remainder rescaled to one gram, so the full "
        "dust mass is kept but the mass outside the table is effectively "
        "given the opacity per gram of the grains inside it; kappa is "
        "biased for distributions with significant mass outside the "
        f"table. Check {check} before using cells near or beyond the "
        "table edges."
    )

    out_grid = GridFile(path)
    out_grid.write_model_metadata(model)

    # Axes in the layout synthesizer's Grid expects
    out_grid.write_attribute("/", "axes", list(results["axes"]))
    for name, values in results["axes"].items():
        radius = name in RADIUS_AXES
        out_grid.write_dataset(
            key=f"axes/{name}",
            data=values * (um if radius else dimensionless),
            description=AXIS_DESCRIPTIONS[name],
            log_on_read=radius,
        )

    out_grid.write_dataset(
        key="wavelength",
        data=results["wav"] * um,
        description="Wavelength of the opacities",
        log_on_read=False,
    )

    for material, res in results["materials"].items():
        label = MATERIALS[material]["label"]
        attrs = {
            "material": material,
            "rho_g_cm3": res["rho"],
            "table_a_min_um": res["table_a_min_um"],
            "table_a_max_um": res["table_a_max_um"],
        }
        for ct, info in CURVE_TYPES.items():
            out_grid.write_dataset(
                key=f"{ct}_curves/{label}",
                data=res["kappa"][ct] * cm**2 / g,
                description=(
                    f"{info['description']} for {material}: optical depth "
                    "per unit dust mass column (tau = kappa * Sigma_dust; "
                    "A = 1.086 tau), shape (*axes, wavelength)"
                ),
                log_on_read=False,
                **attrs,
            )
        for key, (values, units, description) in res["extra"].items():
            out_grid.write_dataset(
                key=f"{key}/{label}",
                data=values * units,
                description=f"{description}, shape (*axes)",
                log_on_read=False,
            )

    # Attributes go on the groups, which now exist
    for ct, info in CURVE_TYPES.items():
        out_grid.write_attribute(
            f"{ct}_curves", "efficiency", info["efficiency"]
        )

    # Each curve group also exposes the shared wavelength through a soft
    # link to /wavelength, so the data is stored only once
    with h5py.File(path, "a") as hf:
        for ct in CURVE_TYPES:
            hf[f"{ct}_curves/wavelength"] = h5py.SoftLink("/wavelength")


def load_paramfile(path: str) -> Dict[str, Any]:
    """Read the YAML parameter file."""
    with open(path, "r") as stream:
        params = yaml.safe_load(stream)
    if not isinstance(params, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return params


def plot_extinction_curve(
    wav_micron: NDArray,
    curves_over_Av: Dict[str, NDArray],
    ax: Optional[Axes] = None,
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
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
        styles (dict)
            optional legend label -> matplotlib line keyword arguments
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    styles = styles or {}

    wav_angstrom = wav_micron * 1e4
    for label, A_over_Av in curves_over_Av.items():
        ax.plot(wav_angstrom, A_over_Av, label=label, **styles.get(label, {}))
    curve = Calzetti2000(ampl=10.0)
    ax.plot(
        wav_angstrom,
        curve.get_tau(wav_angstrom * Angstrom),
        ls="--",
        color="k",
        label="Calzetti+2000",
    )
    for c, color in zip(["MW", "LMC", "SMC"], ["0.3", "0.5", "0.7"]):
        curve_wd01 = GrainModels(submodel=c)
        ax.plot(
            wav_angstrom,
            curve_wd01.get_tau(wav_angstrom * Angstrom),
            ls=":",
            color=color,
            label=f"{c}",
        )
    ax.set_xlabel("Wavelength (AA)", fontsize=12)
    ax.set_ylabel(r"A($\lambda$) / A(V)", fontsize=12)
    ax.set_xlim(800, 1e4)
    ax.set_xticks(np.arange(1000, 10001, 1000))
    ax.grid(which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    return ax


def example_cells(results: Dict[str, Any], n_cells: int) -> List[Tuple]:
    """
    A few grid cells to plot: up to n_cells nodes spread along the first
    axis, with the second axis at its middle node (lognormal) or its
    largest value (powerlaw, where small a_max leaves cells empty). Cells
    that are NaN for any material are skipped.
    """
    sizes = [v.size for v in results["axes"].values()]
    j = sizes[1] // 2 if results["shape"] == "lognormal" else sizes[1] - 1
    rows = np.unique(np.linspace(0, sizes[0] - 1, min(n_cells, sizes[0])))
    cells = [(int(round(i)), j) for i in rows]
    return [
        cell
        for cell in cells
        if all(
            np.all(np.isfinite(res["kappa"]["extinction"][cell]))
            for res in results["materials"].values()
        )
    ]


def plot_example(
    results: Dict[str, Any],
    material_dtg: Dict[str, float],
    png_name: str,
    show: bool,
    v_wavelength_um: float = 0.55,
    n_cells: int = 4,
) -> None:
    """
    Plot A(λ)/A(V) and the size distributions of an example dust mix of the
    grid's materials at a few grid cells (see example_cells).

    Args:
        results (dict)
            output of compute_opacities
        material_dtg (dict)
            dust-to-gas mass ratio of each material (key of MATERIALS);
            materials of the grid that are missing here are left out
        png_name (string)
            output image filename
        show (bool)
            whether to show the plot interactively
        v_wavelength_um (float)
            wavelength (micron) used to normalise A(λ)/A(V)
        n_cells (int)
            maximum number of grid cells to plot
    """
    materials = [m for m in results["materials"] if m in material_dtg]
    cells = example_cells(results, n_cells)
    if not materials or not cells:
        print("Nothing to plot for the example dust mix; skipping")
        return

    wav = results["wav"]
    iv = np.argmin(np.abs(wav - v_wavelength_um))
    names = list(results["axes"])

    # Gas mass per H (g) with helium correction, used only to turn the
    # example DTGs into dust mass per H for the Av/NH comparison
    MU = 1.4
    M_H = 1.6738e-24  # in grams
    dust_mass_per_H = {m: material_dtg[m] * MU * M_H for m in materials}
    MAG_PER_TAU = 2.5 * np.log10(np.e)  # A = 1.086 tau

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ct_ls = {"extinction": "-", "attenuation": "--"}
    curves_over_Av, styles = {}, {}
    fig, axs = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    for k, cell in enumerate(cells):
        color = colors[k % len(colors)]
        cell_label = ", ".join(
            f"{name}={results['axes'][name][i]:.3g}"
            for name, i in zip(names, cell)
        )
        for ct in CURVE_TYPES:
            # A(lam)/N_H in mag cm^2 per H
            A_total = MAG_PER_TAU * sum(
                results["materials"][m]["kappa"][ct][cell] * dust_mass_per_H[m]
                for m in materials
            )
            label = f"{ct}: {cell_label}"
            curves_over_Av[label] = A_total / A_total[iv]
            styles[label] = {"color": color, "ls": ct_ls[ct]}
            print(f"{label}: Av/NH = {A_total[iv]:.3e} mag cm^2")

        # Size distributions, a^4 n(a) per H
        for m in materials:
            res = results["materials"][m]
            n = res["n_per_gram"][cell] * dust_mass_per_H[m]
            ok = n > 0
            a = res["a_grid"][ok]
            axs[1].loglog(
                a,
                (a * MICRON_TO_CM) ** 4 * n[ok],
                color=color,
                ls="-" if m == materials[0] else ":",
                label=f"{m}: {cell_label}",
            )
    print("Milky Way sightline (extinction), Rv=3.1:")
    print("Av/NH = 5.3E-22 mag cm^2")

    plot_extinction_curve(wav, curves_over_Av, axs[0], styles)
    axs[1].set_xlabel("a (um)", fontsize=12)
    axs[1].set_ylabel(r"a$^{4}$ n(a) / (grains/H cm$^3$)", fontsize=12)
    axs[1].set_ylim(1e-30, 1e-26)
    axs[1].grid(which="both", ls="--", alpha=0.4)
    axs[1].legend(fontsize=7, ncol=2)

    dtgs = ", ".join(f"{m}={material_dtg[m]:.3g}" for m in materials)
    fig.suptitle(f"Example dust mix (DTG): {dtgs}", fontsize=9, wrap=True)
    plt.tight_layout()
    plt.savefig(png_name, dpi=200)
    print(f"Saved plot: {png_name}")
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build a grid of dust mass opacities from Draine optical tables "
            "for the materials and size-distribution axes in a YAML "
            "parameter file."
        )
    )
    parser.add_argument(
        "--paramfile",
        type=str,
        required=True,
        help="path to the YAML parameter file (see params/)",
    )
    parser.add_argument(
        "--plot-example",
        action="store_true",
        help="plot the curves for an example dust mix (EXAMPLE_DTG)",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="show the extinction curve plot",
    )
    args = parser.parse_args()

    params = load_paramfile(args.paramfile)
    output = params.get("output") or {}
    grid_loc = output.get("grid_loc", GRID_DIR)
    grid_name = output.get("grid_name", Path(args.paramfile).stem)

    results = compute_opacities(params)

    Path(grid_loc).mkdir(parents=True, exist_ok=True)
    path = f"{grid_loc}/{grid_name}.hdf5"
    write_grid(path, params, results)
    print(f"Saved dust opacity grid to: {path}")

    if args.plot_example:
        # Example dust-to-gas mass ratio of each material; only the grid's
        # materials are used
        EXAMPLE_DTG = {
            "silicate": 0.006,
            "graphite": 0.003,
            "pah_ionised": 0.00005,
            "pah_neutral": 0.00005,
        }
        plot_example(
            results, EXAMPLE_DTG, f"{grid_name}_example.png", args.show_plot
        )
