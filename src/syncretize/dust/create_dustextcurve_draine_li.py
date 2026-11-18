#!/usr/bin/env python3
"""
Create dust mass opacities (extinction curves) from Draine optical properties.

Parse Draine's Sil_81, Gra_81, PAHion_30 and PAHneu_30 tables, build the
grain-size distributions listed in a YAML parameter file, and compute the
mass opacity of each grain component,

    kappa(lam) = integral pi a^2 Q(a, lam) n(a) da    (cm^2 per g of dust),

with n(a) normalised to one gram of dust over the radii where Q is tabulated
(any part of a distribution outside the table is cut off and the mass
renormalised onto the table's radii).

The extinction scales linearly with the dust mass: for a line of sight with
dust mass column Sigma_dust,c of each component c, the optical depth is
tau(lam) = sum_c kappa_c(lam) Sigma_dust,c and A(lam) = 1.086 tau(lam).

Two opacities are written for every component:
  - extinction: computed with Q_ext, i.e. every scattered photon is removed
    from the beam (point source behind a dust screen).
  - attenuation: computed with Q_abs + (1 - g) Q_sca, i.e. forward-scattered
    light is kept in the beam, a simple stand-in for unresolved scattering
    geometry.

Component kinds (see params/*.yaml for examples):
  - lognormal: fixed lognormal components, one per median radius. The
    lognormal is defined in number,
        n(a) = M0 / (sqrt(2 pi) sigma a) exp(-(ln a - mu)^2 / (2 sigma^2)),
    so the number-median radius is a_n = e^mu and the mass-median radius is
    a_m = a_n exp(3 sigma^2). Medians are given explicitly as either
    number_median or mass_median (micron).
  - powerlaw: dn/da propto a^power on one or more radius ranges, one
    component per range. Covers MRN-like ranges and discrete grain-size bins.
  - lognormal_table: kappa tabulated on a grid of (log10 a_n, sigma), for
    moment-based lognormal modes whose parameters vary continuously.

Output layout:
  /wavelength                        (micron)
  /components/<name>/extinction      kappa, shape (*axes, n_wavelength)
  /components/<name>/attenuation     kappa, shape (*axes, n_wavelength)
  /components/<name>/axes/<axis>     only for tabulated components
  Each component group carries attributes describing its material, kind,
  axes (empty for fixed components) and size-distribution parameters.

Usage:
  python create_dustextcurve_draine_li.py \
      --paramfile params/draine_li_lognormal.yaml \
      [--plot_example] [--show_plot]
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

# Draine optical tables: table name, dataset-name label and default
# material density (g cm^-3). Densities can be overridden in the parameter
# file.
MATERIALS = {
    "silicate": {"table": "Sil_81", "label": "silicate", "rho": 3.5},
    "graphite": {"table": "Gra_81", "label": "graphite", "rho": 2.24},
    "pah_ionised": {"table": "PAHion_30", "label": "pahionised", "rho": 2.24},
    "pah_neutral": {"table": "PAHneu_30", "label": "pahneutral", "rho": 2.24},
}

# The two efficiencies written for every component
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

# Allowed keys for each component kind (besides "kind" and "material")
COMPONENT_KEYS = {
    "lognormal": {"number_median", "mass_median", "sigma", "names"},
    "powerlaw": {"edges", "ranges", "power", "names"},
    "lognormal_table": {"log10_number_median", "sigma", "name"},
}
TOP_LEVEL_KEYS = {
    "output",
    "data_dir",
    "n_a_int",
    "densities",
    "components",
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


def build_kernel(
    radii_micron: NDArray, Q: NDArray, a_grid_micron: NDArray
) -> NDArray:
    """
    Kernel K such that kappa = n_per_gram @ K.T, i.e.
    K[lam, a] = pi a^2 Q(a, lam) w(a), with w the trapezoid weights in a.

    Args:
        radii_micron (NDArray)
            radius grid (micron) of the optical table
        Q (NDArray)
            efficiency on (n_wavelength, len(radii_micron))
        a_grid_micron (NDArray)
            integration grid (micron); must lie within radii_micron
    Returns:
        K (NDArray)
            kernel (cm^2 * cm), shape (n_wavelength, len(a_grid_micron))
    """
    # Never extrapolate Q in radius (raise instead)
    Q_a = interp1d(radii_micron, Q, axis=1, bounds_error=True)(a_grid_micron)
    a_cm = a_grid_micron * MICRON_TO_CM
    return math.pi * a_cm**2 * Q_a * trapezoid_weights(a_cm)


# Size distributions
def lognormal_shape(
    a_grid_micron: NDArray, number_median_micron: Any, sigma: Any
) -> NDArray:
    """
    Unnormalised lognormal size distribution defined in number,
    n(a) propto (1 / a) exp(-(ln a - mu)^2 / (2 sigma^2)), mu = ln(a_n).
    number_median_micron and sigma broadcast against each other; the
    radius axis is appended last.

    Args:
        a_grid_micron (NDArray)
            radius grid (micron)
        number_median_micron (float or NDArray)
            number-median radius a_n = e^mu (micron)
        sigma (float or NDArray)
            width in ln(a)
    Returns:
        shape (NDArray)
            unnormalised n(a), shape broadcast(a_n, sigma) + (n_a,)
    """
    a_n = np.asarray(number_median_micron, dtype=float)[..., np.newaxis]
    sig = np.asarray(sigma, dtype=float)[..., np.newaxis]
    if np.any(sig <= 0):
        raise ValueError("sigma must be > 0")
    x = np.log(a_grid_micron / a_n)
    return np.exp(-0.5 * (x / sig) ** 2) / a_grid_micron


def powerlaw_shape(
    a_grid_micron: NDArray,
    a_min: float,
    a_max: float,
    power: float,
    include_a_max: bool = True,
) -> NDArray:
    """
    Unnormalised truncated power law, n(a) propto a^power on
    [a_min, a_max) (micron) and zero elsewhere. The upper edge is only
    included when include_a_max is set, so a radius shared by two
    adjacent bins belongs to the upper bin alone.

    Args:
        a_grid_micron (NDArray)
            radius grid (micron)
        a_min, a_max (float)
            radius range (micron)
        power (float)
            power-law exponent (dn/da propto a^power)
        include_a_max (bool)
            include a_max itself, i.e. integrate over [a_min, a_max]
    Returns:
        shape (NDArray)
            unnormalised n(a) on a_grid_micron
    """
    shape = np.zeros_like(a_grid_micron)
    below_max = (
        a_grid_micron <= a_max if include_a_max else a_grid_micron < a_max
    )
    mask = (a_grid_micron >= a_min) & below_max
    shape[mask] = a_grid_micron[mask] ** power
    return shape


def normalise_per_gram(
    a_grid_micron: NDArray, shape: NDArray, rho: float
) -> NDArray:
    """
    Normalise size distributions to one gram of dust on a_grid_micron,
    i.e. integral (4/3 pi a^3 rho n(a) da) = 1 g, using the same trapezoid
    weights as the kappa integral.

    Args:
        a_grid_micron (NDArray)
            radius grid (micron)
        shape (NDArray)
            unnormalised n(a), radius axis last
        rho (float)
            material density (g cm^-3)
    Returns:
        n(a) (NDArray)
            grains per gram of dust per cm, same shape as shape
    """
    a_cm = a_grid_micron * MICRON_TO_CM
    grain_mass = (4.0 / 3.0) * math.pi * rho * a_cm**3
    mass = shape @ (grain_mass * trapezoid_weights(a_cm))
    if not np.all(mass > 0):
        raise ValueError(
            "Size distribution has no mass on the integration grid; "
            "check it overlaps the radii covered by the optical table."
        )
    return shape / mass[..., np.newaxis]


def lognormal_mass_fraction(
    a_lo_micron: float,
    a_hi_micron: float,
    number_median_micron: Any,
    sigma: Any,
) -> NDArray:
    """
    Fraction of an untruncated lognormal's mass lying in [a_lo, a_hi].
    The mass per ln(a) is a Gaussian in ln(a) of width sigma centred on the
    mass-median radius a_m = a_n exp(3 sigma^2).

    Args:
        a_lo_micron, a_hi_micron (float)
            radius range (micron)
        number_median_micron (float or NDArray)
            number-median radius (micron)
        sigma (float or NDArray)
            width in ln(a)
    Returns:
        mass fraction in [a_lo, a_hi], broadcast shape of the inputs
    """
    sig = np.asarray(sigma, dtype=float)
    ln_am = np.log(np.asarray(number_median_micron, dtype=float)) + 3 * sig**2
    z_lo = (math.log(a_lo_micron) - ln_am) / (sig * math.sqrt(2))
    z_hi = (math.log(a_hi_micron) - ln_am) / (sig * math.sqrt(2))
    return 0.5 * (erf(z_hi) - erf(z_lo))


def format_radius(value: float) -> str:
    """Format a radius for a dataset name without exponent notation."""
    return np.format_float_positional(float(value), trim="-")


def _as_list(value: Any, n: int, what: str) -> List[Any]:
    """Broadcast a scalar to a list of length n, or check a list's length."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{what} has {len(value)} entries, expected {n}")
        return list(value)
    return [value] * n


def _axis_values(spec: Any, what: str) -> NDArray:
    """Axis from {min, max, n} (linearly spaced) or an explicit list."""
    if isinstance(spec, dict):
        if set(spec) != {"min", "max", "n"}:
            raise ValueError(f"{what} needs exactly min, max and n")
        if int(spec["n"]) < 2 or not spec["max"] > spec["min"]:
            raise ValueError(f"{what} needs n >= 2 and max > min")
        return np.linspace(
            float(spec["min"]), float(spec["max"]), int(spec["n"])
        )
    values = np.asarray(spec, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0):
        raise ValueError(f"{what} must be strictly increasing, >= 2 values")
    return values


def expand_components(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate the parameter file's component entries and expand them into
    one spec per output component.

    Args:
        params (dict)
            parsed YAML parameter file
    Returns:
        list of component specs, each a dict with name, material, kind and
        the kind's parameters
    """
    unknown = set(params) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown top-level parameter file keys: {sorted(unknown)}"
        )
    if not params.get("components"):
        raise ValueError("The parameter file must list at least one component")

    specs = []
    for i, entry in enumerate(params["components"]):
        where = f"components[{i}]"
        kind = entry.get("kind")
        material = entry.get("material")
        if kind not in COMPONENT_KEYS:
            raise ValueError(
                f"{where}: kind must be one of {sorted(COMPONENT_KEYS)}"
            )
        if material not in MATERIALS:
            raise ValueError(
                f"{where}: material must be one of {sorted(MATERIALS)}"
            )
        unknown = set(entry) - COMPONENT_KEYS[kind] - {"kind", "material"}
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        label = MATERIALS[material]["label"]

        if kind == "lognormal":
            has_n = "number_median" in entry
            has_m = "mass_median" in entry
            if has_n == has_m:
                raise ValueError(
                    f"{where}: give exactly one of number_median or "
                    "mass_median (micron)"
                )
            medians = entry["number_median" if has_n else "mass_median"]
            medians = medians if isinstance(medians, list) else [medians]
            sigmas = _as_list(entry.get("sigma"), len(medians), "sigma")
            names = entry.get("names") or [
                f"{label}_a{format_radius(m)}um" for m in medians
            ]
            names = _as_list(names, len(medians), f"{where} names")
            for name, median, sigma in zip(names, medians, sigmas):
                if sigma is None or not float(sigma) > 0:
                    raise ValueError(f"{where}: sigma must be > 0")
                sigma = float(sigma)
                # Everything is computed in the number convention
                factor = math.exp(3 * sigma**2)
                a_n = float(median) if has_n else float(median) / factor
                specs.append(
                    {
                        "name": name,
                        "material": material,
                        "kind": kind,
                        "number_median_um": a_n,
                        "mass_median_um": a_n * factor,
                        "sigma": sigma,
                    }
                )

        elif kind == "powerlaw":
            if ("edges" in entry) == ("ranges" in entry):
                raise ValueError(f"{where}: give exactly one of edges/ranges")
            if "power" not in entry:
                raise ValueError(f"{where}: power is required")
            if "edges" in entry:
                edges = entry["edges"]
                if not isinstance(edges, list) or any(
                    isinstance(e, (list, tuple)) for e in edges
                ):
                    raise ValueError(
                        f"{where}: edges must be a flat list of bin edges, "
                        "e.g. [0.001, 0.01, 0.1]; use ranges for a list of "
                        "[min, max] pairs"
                    )
                edges = [float(e) for e in edges]
                if len(edges) < 2 or np.any(np.diff(edges) <= 0):
                    raise ValueError(
                        f"{where}: edges must be strictly increasing"
                    )
                ranges = list(zip(edges[:-1], edges[1:]))
            else:
                ranges = entry["ranges"]
                if not isinstance(ranges, list) or not all(
                    isinstance(r, (list, tuple)) and len(r) == 2
                    for r in ranges
                ):
                    raise ValueError(
                        f"{where}: ranges must be a list of [min, max] pairs, "
                        "e.g. [[0.001, 0.01], [0.1, 1.0]]; use edges for a "
                        "flat list of bin edges"
                    )
                ranges = [(float(lo), float(hi)) for lo, hi in ranges]
                if any(not hi > lo for lo, hi in ranges):
                    raise ValueError(f"{where}: each range needs max > min")
            names = entry.get("names") or [
                f"{label}_a{format_radius(lo)}-{format_radius(hi)}um"
                for lo, hi in ranges
            ]
            names = _as_list(names, len(ranges), f"{where} names")
            # upper edges are exclusive so adjacent bins don't share a
            # radius, except for the entry's topmost bin
            top = max(hi for _, hi in ranges)
            for name, (lo, hi) in zip(names, ranges):
                specs.append(
                    {
                        "name": name,
                        "material": material,
                        "kind": kind,
                        "a_min_um": lo,
                        "a_max_um": hi,
                        "include_a_max": hi == top,
                        "power": float(entry["power"]),
                    }
                )

        # lognormal_table: useful for moment-based lognormal modes whose
        # parameters can vary continuously
        else:
            for key in ("log10_number_median", "sigma"):
                if key not in entry:
                    raise ValueError(f"{where}: {key} is required")
            log10_a_n = _axis_values(
                entry["log10_number_median"], f"{where} log10_number_median"
            )
            sigmas = _axis_values(entry["sigma"], f"{where} sigma")
            if np.any(sigmas <= 0):
                raise ValueError(f"{where}: sigma values must be > 0")
            specs.append(
                {
                    "name": entry.get("name", f"{label}_lognormal_table"),
                    "material": material,
                    "kind": kind,
                    "log10_number_median_um": log10_a_n,
                    "sigma": sigmas,
                }
            )

    names = [s["name"] for s in specs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Duplicate component names: {duplicates}")
    return specs


def load_paramfile(path: str) -> Dict[str, Any]:
    """Read the YAML parameter file."""
    with open(path, "r") as stream:
        params = yaml.safe_load(stream)
    if not isinstance(params, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return params


def compute_component(
    spec: Dict[str, Any], a_grid: NDArray, rho: float
) -> Dict[str, Any]:
    """
    Build a component's size distribution(s), normalised to one gram of
    dust on the material's integration grid, plus its output metadata.

    Args:
        spec (dict)
            one entry of expand_components
        a_grid (NDArray)
            the material's integration grid (micron)
        rho (float)
            material density (g cm^-3)
    Returns:
        dict with n_per_gram (shape (*axes, n_a)), axes (dict name ->
        values), attrs (dict) and datasets (dict name -> (array,
        description)) to write alongside the opacities
    """
    lo, hi = a_grid[0], a_grid[-1]
    attrs: Dict[str, Any] = {
        "material": spec["material"],
        "kind": spec["kind"],
        "rho_g_cm3": rho,
        "table_a_min_um": lo,
        "table_a_max_um": hi,
    }
    axes: Dict[str, NDArray] = {}
    datasets: Dict[str, Tuple[NDArray, str]] = {}

    if spec["kind"] == "lognormal":
        a_n, sigma = spec["number_median_um"], spec["sigma"]
        shape = lognormal_shape(a_grid, a_n, sigma)
        frac = float(lognormal_mass_fraction(lo, hi, a_n, sigma))
        attrs.update(
            number_median_um=a_n,
            mass_median_um=spec["mass_median_um"],
            sigma=sigma,
            untruncated_mass_fraction=frac,
        )
        print(
            f"{spec['name']}: {100 * frac:.2f}% of the lognormal mass lies "
            f"in the table range [{lo:.4g}, {hi:.4g}] um; renormalised"
        )

    elif spec["kind"] == "powerlaw":
        a_min, a_max = spec["a_min_um"], spec["a_max_um"]
        # nothing lies above the table's last radius, so keep it
        include_a_max = bool(spec["include_a_max"] or a_max >= hi)
        shape = powerlaw_shape(
            a_grid, a_min, a_max, spec["power"], include_a_max
        )
        attrs.update(
            a_min_um=a_min,
            a_max_um=a_max,
            include_a_max=include_a_max,
            power=spec["power"],
            integrated_a_min_um=max(a_min, lo),
            integrated_a_max_um=min(a_max, hi),
        )
        print(
            f"{spec['name']}: power law on [{a_min:.4g}, {a_max:.4g}"
            f"{']' if include_a_max else ')'} um, "
            f"integrated over [{max(a_min, lo):.4g}, {min(a_max, hi):.4g}] "
            "um (table range)"
        )
    # lognormal_table
    else:
        log10_a_n, sigmas = spec["log10_number_median_um"], spec["sigma"]
        a_n_mesh, sig_mesh = np.meshgrid(10**log10_a_n, sigmas, indexing="ij")
        shape = lognormal_shape(a_grid, a_n_mesh, sig_mesh)
        frac = lognormal_mass_fraction(lo, hi, a_n_mesh, sig_mesh)
        axes = {"log10_number_median_um": log10_a_n, "sigma": sigmas}
        datasets["untruncated_mass_fraction"] = (
            frac,
            "Fraction of the untruncated lognormal's mass within the "
            "table's radius range (before renormalisation)",
        )
        print(
            f"{spec['name']}: table of {log10_a_n.size} x {sigmas.size} "
            f"(log10 a_n, sigma); untruncated mass fraction in table range "
            f"{100 * frac.min():.2f}%-{100 * frac.max():.2f}%"
        )

    return {
        "n_per_gram": normalise_per_gram(a_grid, shape, rho),
        "axes": axes,
        "attrs": attrs,
        "datasets": datasets,
    }


def compute_opacities(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load the optical tables, build every component's size distribution and
    compute its extinction and attenuation mass opacities.

    Args:
        params (dict)
            parsed YAML parameter file
    Returns:
        dict with wav (micron), a_grids (material -> grid), and components
        (name -> output of compute_component plus kappa, a dict curve type
        -> kappa of shape (*axes, n_wavelength) in cm^2 g^-1)
    """
    specs = expand_components(params)
    data_dir = params.get("data_dir", ".")
    n_a_int = int(params.get("n_a_int", 500))
    densities = params.get("densities") or {}
    unknown = set(densities) - set(MATERIALS)
    if unknown:
        raise ValueError(f"Unknown materials in densities: {sorted(unknown)}")

    materials = sorted({s["material"] for s in specs})
    tables = {
        material: load_optical_table(material, data_dir)
        for material in materials
    }

    # Common wavelength grid: union of the tables in use
    wav = np.unique(
        np.concatenate([tables[material]["wav"] for material in materials])
    )

    # Per material: integration grid spanning the table's radii, and the
    # kernel K[lam, a] for each curve type so that kappa = n @ K.T
    a_grids, kernels = {}, {}
    for material in materials:
        radii = tables[material]["radii"]
        a_grids[material] = table_radius_grid(radii, n_a_int)
        Q = efficiencies_on_grid(tables[material], wav)
        kernels[material] = {
            ct: build_kernel(radii, Q[ct], a_grids[material])
            for ct in CURVE_TYPES
        }

    components = {}
    for spec in specs:
        material = spec["material"]
        rho = float(densities.get(material, MATERIALS[material]["rho"]))
        comp = compute_component(spec, a_grids[material], rho)
        comp["material"] = material
        # kappa(lam) = integral pi a^2 Q(a, lam) n(a) da. With the trapezoid
        # rule this is sum_i w_i pi a_i^2 Q(a_i, lam) n(a_i), where the
        # weights w_i depend only on the radius grid. Everything except n(a)
        # is folded into the per-material kernel K[lam, i] (build_kernel), so
        # the integral for any number of distributions is one matrix product,
        # n_per_gram (*axes, n_a) @ K.T (n_a, n_wavelength). For a
        # lognormal_table this avoids looping over every (a_n, sigma) node or
        # broadcasting to a (*axes, n_wavelength, n_a) array (GBs for a
        # typical table). normalise_per_gram uses the same weights, so the
        # normalised mass and the kappa integral share one quadrature.
        comp["kappa"] = {
            ct: comp["n_per_gram"] @ kernels[material][ct].T
            for ct in CURVE_TYPES
        }
        components[spec["name"]] = comp

    return {"wav": wav, "a_grids": a_grids, "components": components}


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
    out_grid = GridFile(path)
    out_grid.write_model_metadata(
        {
            "model_name": "Draine & Li dust mass opacities",
            "url": MODEL_URL,
            "opacity_units": "cm**2/g",
            "usage": (
                "tau(lam) = sum_c kappa_c(lam) * Sigma_dust,c, where "
                "Sigma_dust,c is the dust mass column of component c; "
                "A(lam) = 1.086 tau(lam)"
            ),
            "lognormal_definition": (
                "n(a) = M0 / (sqrt(2 pi) sigma a) "
                "exp(-(ln a - mu)^2 / (2 sigma^2)), a_n = e^mu, "
                "a_mass_median = a_n exp(3 sigma^2)"
            ),
            "paramfile": yaml.safe_dump(params, sort_keys=False),
        }
    )
    out_grid.write_dataset(
        key="wavelength",
        data=results["wav"] * um,
        description="Wavelength of the opacities",
        log_on_read=False,
    )

    for name, comp in results["components"].items():
        group = f"components/{name}"
        for axis, values in comp["axes"].items():
            out_grid.write_dataset(
                key=f"{group}/axes/{axis}",
                data=values * dimensionless,
                description=f"{axis} axis of the opacity table",
                log_on_read=False,
            )
        for ct, info in CURVE_TYPES.items():
            out_grid.write_dataset(
                key=f"{group}/{ct}",
                data=comp["kappa"][ct] * cm**2 / g,
                description=(
                    f"{info['description']} for component {name}: optical "
                    "depth per unit dust mass column (tau = kappa * "
                    "Sigma_dust; A = 1.086 tau), shape (*axes, wavelength)"
                ),
                log_on_read=False,
                efficiency=info["efficiency"],
            )
        for dset, (values, description) in comp["datasets"].items():
            out_grid.write_dataset(
                key=f"{group}/{dset}",
                data=values * dimensionless,
                description=description,
                log_on_read=False,
            )
        # Attributes go on the group, which now exists
        out_grid.write_attribute(
            group,
            "axes",
            np.array(list(comp["axes"]), dtype=h5py.string_dtype()),
        )
        for key, value in comp["attrs"].items():
            out_grid.write_attribute(group, key, value)


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
        _, ax = plt.subplots(figsize=(8, 5))

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
    ax.grid(which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=11, ncol=2)
    return ax


def plot_example(
    results: Dict[str, Any],
    material_dtg: Dict[str, float],
    png_name: str,
    show: bool,
    v_wavelength_um: float = 0.55,
) -> None:
    """
    Plot the A(λ)/A(V) curves and size distributions for an example dust
    mix. Each material's dust-to-gas mass ratio is split evenly across that
    material's fixed (non-tabulated) components; materials without fixed
    components are skipped.

    Args:
        results (dict)
            output of compute_opacities
        material_dtg (dict)
            dust-to-gas mass ratio of each material (key of MATERIALS)
        png_name (string)
            output image filename
        show (bool)
            whether to show the plot interactively
        v_wavelength_um (float)
            wavelength (micron) used to normalise A(λ)/A(V)
    """
    components = results["components"]
    dtg = {}
    for material, material_total in material_dtg.items():
        names = [
            name
            for name, comp in components.items()
            if comp["material"] == material and not comp["axes"]
        ]
        for name in names:
            dtg[name] = material_total / len(names)
    if not dtg:
        print(
            "No fixed components in the parameter file; skipping the "
            "example plot"
        )
        return

    wav = results["wav"]
    iv = np.argmin(np.abs(wav - v_wavelength_um))

    # Gas mass per H (g) with helium correction, used only to turn the
    # example DTGs into dust mass per H for the Av/NH comparison
    MU = 1.4
    M_H = 1.6738e-24  # in grams
    dust_mass_per_H = {name: d * MU * M_H for name, d in dtg.items()}
    MAG_PER_TAU = 2.5 * np.log10(np.e)  # A = 1.086 tau

    plot_labels = {
        "extinction": r"Extinction ($Q_{\rm ext}$)",
        "attenuation": r"Attenuation ($Q_{\rm abs} + (1-g)\,Q_{\rm sca}$)",
    }
    curves_over_Av = {}
    for ct in CURVE_TYPES:
        # A(lam)/N_H in mag cm^2 per H
        A_total = MAG_PER_TAU * sum(
            components[name]["kappa"][ct] * m
            for name, m in dust_mass_per_H.items()
        )
        curves_over_Av[plot_labels[ct]] = A_total / A_total[iv]
        print(f"{ct}: Av/NH = {A_total[iv]:.3e} mag cm^2")
    print("Milky Way sightline (extinction), Rv=3.1:")
    print("Av/NH = 5.3E-22 mag cm^2")

    fig, axs = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    plot_extinction_curve(wav, curves_over_Av, axs[0])

    # Size distributions summed per material, a^4 n(a) per H
    by_material: Dict[str, NDArray] = {}
    for name, m in dust_mass_per_H.items():
        mat = components[name]["material"]
        n = components[name]["n_per_gram"] * m
        by_material[mat] = by_material.get(mat, 0) + n
    for mat, n_tot in by_material.items():
        a_grid = results["a_grids"][mat]
        ok = n_tot != 0
        axs[1].loglog(
            a_grid[ok],
            ((a_grid[ok] * MICRON_TO_CM) ** 4) * n_tot[ok],
            label=mat,
            ls="dotted" if mat.startswith("pah") else "dashed",
            # Thicker so identical ionised/neutral PAH curves both show
            lw=4 if mat == "pah_ionised" else None,
        )
    axs[1].set_ylim(1e-30, 1e-26)
    axs[1].set_xlabel("a (um)", fontsize=12)
    axs[1].set_ylabel(r"a$^{4}$ n(a) / (grains/H cm$^3$)", fontsize=12)
    axs[1].grid(which="both", ls="--", alpha=0.4)
    axs[1].legend(fontsize=11, ncol=2)

    dtgs = ", ".join(f"{k}={v:.3g}" for k, v in dtg.items())
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
            "Build dust mass opacities from Draine optical tables for the "
            "grain components listed in a YAML parameter file."
        )
    )
    parser.add_argument(
        "--paramfile",
        type=str,
        required=True,
        help="path to the YAML parameter file (see params/)",
    )
    parser.add_argument(
        "--plot_example",
        action="store_true",
        help="plot the curves for an example dust mix (EXAMPLE_DTG)",
    )
    parser.add_argument(
        "--show_plot",
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
        # Example dust-to-gas mass ratio of each material, split evenly
        # across that material's fixed components in the parameter file
        EXAMPLE_DTG = {
            "silicate": 0.006,
            "graphite": 0.003,
            "pah_ionised": 0.00005,
            "pah_neutral": 0.00005,
        }
        plot_example(
            results, EXAMPLE_DTG, f"{grid_name}_example.png", args.show_plot
        )
