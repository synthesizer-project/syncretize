"""
Collect Cloudy outputs from Sobol-sampled models and add them to the HDF5 grid
created by create_cloudy_input_sobol.py.

Reads, for each sample, the raw cloudy files kept in the ``cont/`` subdirectory
by the submission script:
  - ``{index}.cont``: continuum spectra, read with synthesizer's
    ``read_continuum`` (column mapping, wavelength ordering and units);
  - ``{index}.emergent_elin`` (and optionally ``.intrinsic_elin``): line
    luminosities for the line list given by ``output_linelist``.

Usage:
    python collect_sobol_outputs.py --output-dir /path/to/cloudy_output_dir
    python collect_sobol_outputs.py --output-dir /path/to/dir \
        --spec-names nebular linecont --no-lines
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

from syncretize.cloudy import cloudy17, cloudy23, cloudy25

# All spectra exposed by synthesizer.read_continuum(return_dict=True).
AVAILABLE_SPECTRA = (
    "incident",
    "transmitted",
    "nebular",
    "nebular_continuum",
    "total",
    "linecont",
)

# Base spectra to store; total and nebular_continuum are derived from these
# within synthesizer and so are not saved by default.
DEFAULT_SPECTRA = (
    "incident",
    "transmitted",
    "nebular",
    "linecont",
)

# Continuum values stored at each line wavelength (as in synthesizer grids)
LINE_CONTINUA = (
    "incident",
    "transmitted",
    "nebular_continuum",
    "total_continuum",
)

H_PLANCK = 6.62607015e-27  # erg s
C_AA = 2.99792458e18  # A/s
LAM_HI = 911.76  # A, H ionisation edge


def _select_cloudy_module(cloudy_version):
    """Return the synthesizer cloudy module for a given version string."""
    major = cloudy_version.split(".")[0]
    if major == "c23":
        return cloudy23
    elif major == "c25":
        return cloudy25
    elif major == "c17":
        return cloudy17
    raise ValueError(f"Unknown Cloudy version: {cloudy_version}")


def ionising_rate(lam, lnu):
    """Q(H) [1/s] of L_nu [erg/s/Hz] on ascending wavelengths lam [A]."""
    m = lam <= LAM_HI
    nu = C_AA / lam[m]
    return np.abs(np.trapezoid(lnu[..., m] / (H_PLANCK * nu), nu, axis=-1))


def read_elin(path):
    """Read a Cloudy `save line list ... column` file.

    Returns the line ids (single-spaced) and luminosities, scaled like
    synthesizer's cloudy.read_linelist (divided by 1e7) so that they are
    consistent with read_continuum.
    """
    ids, lums = [], []
    with open(path) as f:
        for row in f:
            parts = row.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            try:
                lum = float(parts[1])
            except ValueError:  # header row
                continue
            ids.append(" ".join(parts[0].split()))
            lums.append(lum)
    return ids, np.array(lums) / 1e7


def _write(group, key, data, units, description, **dset_kw):
    """Create dataset with synthesizer's attributes, replacing any old one."""
    if key in group:
        del group[key]
    d = group.create_dataset(key, data=data, **dset_kw)
    d.attrs["Units"] = units
    d.attrs["Description"] = description
    d.attrs["log_on_read"] = False
    return d


def collect_sobol_outputs(
    output_dir,
    output_file=None,
    spec_names=DEFAULT_SPECTRA,
    lines=True,
    intrinsic=False,
    normalise=True,
    block=100,
):
    """
    Collect all Sobol output spectra (and lines) into the parameter HDF5.

    Args:
        output_dir (str): Path to Cloudy output directory
        output_file (str): Output HDF5 filename (optional)
        spec_names (tuple): Spectra to save (any of AVAILABLE_SPECTRA).
        lines (bool): Also collect the line luminosities (.emergent_elin).
        intrinsic (bool): Also store the intrinsic line luminosities.
        normalise (bool): Normalise by the incident ionising photon rate.
        block (int): Samples read before each write to the HDF5 file.
    """
    output_dir = Path(output_dir)

    for spec_name in spec_names:
        if spec_name not in AVAILABLE_SPECTRA:
            raise ValueError(
                f"Unknown spectrum type: {spec_name}. "
                f"Choose from {AVAILABLE_SPECTRA}."
            )

    # Determine HDF5 grid file path. If not specified, find the single .hdf5
    # in the output dir (created by create_cloudy_input_sobol.py).
    if output_file is None:
        hdf5_files = sorted(output_dir.glob("*.hdf5"))
        if len(hdf5_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .hdf5 in {output_dir}, found "
                f"{len(hdf5_files)}. Run create_cloudy_input_sobol.py first."
            )
        output_file = hdf5_files[0]
    else:
        output_file = Path(output_file)
        if not output_file.exists():
            raise FileNotFoundError(f"Grid HDF5 file not found: {output_file}")

    # Load grid parameters from YAML for cloudy version
    param_file = output_dir / "grid_parameters.yaml"
    with open(param_file, "r") as f:
        grid_params = yaml.safe_load(f)

    cloudy = _select_cloudy_module(grid_params["cloudy_version"])

    # Read n_samples and the stored incident spectra (for the normalisation)
    with h5py.File(output_file, "r") as hf:
        n_samples = int(hf.attrs["n_samples"])
        has_incident = "incident_lnu" in hf and "incident_lam" in hf
        if normalise and has_incident:
            inc_lam = hf["incident_lam"][:]
            order = np.argsort(inc_lam)
            inc_lam = inc_lam[order]
            q_ref = np.concatenate(
                [
                    ionising_rate(
                        inc_lam, hf["incident_lnu"][s : s + 1000][:, order]
                    )
                    for s in range(0, n_samples, 1000)
                ]
            )
    if normalise and not has_incident:
        print("No incident_lnu in the grid file: spectra are not normalised")
        normalise = False

    print(f"Collecting {n_samples} Sobol samples from {output_dir}")
    print(f"Cloudy version: {grid_params['cloudy_version']}")
    print(f"Spectra to save: {spec_names}")

    # The submission script saves the raw cloudy files for each sample as
    # cont/{index}.<ext> (synthesizer's read_continuum appends the .cont).
    cont_dir = output_dir / "cont"
    available = sorted(cont_dir.glob("*.cont"))
    if not available:
        raise FileNotFoundError(f"No .cont files found in {cont_dir}")

    # Read the (ascending, Angstrom) wavelength grid from the first file
    first_index = available[0].stem  # "{index}.cont" -> "{index}"
    lam = cloudy.read_wavelength(str(cont_dir / first_index))
    n_lambda = len(lam)
    print(f"Wavelength grid size: {n_lambda}")

    # Reference line list from the first sample that has one
    line_exts = ["emergent_elin"] + (["intrinsic_elin"] if intrinsic else [])
    if lines:
        elins = sorted(cont_dir.glob("*.emergent_elin"))
        if not elins:
            print("No .emergent_elin files in cont/: lines are not collected")
            lines = False
    if lines:
        line_ids, _ = read_elin(elins[0])
        line_lam = np.array(
            [
                cloudy.convert_cloudy_wavelength(i.split(" ")[-1])
                for i in line_ids
            ]
        )
        line_order = np.argsort(line_lam, kind="stable")
        n_lines = len(line_ids)
        print(f"Lines: {n_lines} (from {elins[0].name})")

    # Create the output datasets up front and fill them block by block
    missing, bad = [], []
    valid_samples = np.zeros(n_samples, dtype=bool)
    norm_all = np.ones(n_samples)
    spec_units = "erg/(Hz*s)" if normalise else "Cloudy units / 1e7"
    with h5py.File(output_file, "a") as hf:
        sg = hf.require_group("spectra")
        rows = min(block, n_samples)
        for spec_name in spec_names:
            _write(
                sg,
                spec_name,
                None,
                spec_units,
                f"{spec_name} spectrum",
                shape=(n_samples, n_lambda),
                dtype="f8",
                chunks=(rows, min(n_lambda, 16384)),
                compression="gzip",
            )
        if lines:
            if "lines" in hf:
                del hf["lines"]
            lg = hf.create_group("lines")
            _write(
                lg,
                "wavelength",
                line_lam[line_order],
                "Å",
                "Wavelength of each emission line",
            )
            ids_sorted = np.array(line_ids, dtype=object)[line_order]
            d = lg.create_dataset(
                "id", data=ids_sorted, dtype=h5py.string_dtype()
            )
            d.attrs["Description"] = "Cloudy ID of each emission line"
            lum_units = "erg/s" if normalise else "Cloudy units / 1e7"
            cont_units = "erg/(Hz*s)" if normalise else "Cloudy units / 1e7"
            line_dsets = {
                "luminosity": (lum_units, "Line luminosity"),
                **(
                    {
                        "intrinsic_luminosity": (
                            lum_units,
                            "Intrinsic line luminosity",
                        )
                    }
                    if intrinsic
                    else {}
                ),
                **{
                    k: (cont_units, f"{' '.join(k.split('_'))} luminosity")
                    for k in LINE_CONTINUA
                },
            }
            for key, (units, desc) in line_dsets.items():
                _write(
                    lg,
                    key,
                    None,
                    units,
                    desc,
                    shape=(n_samples, n_lines),
                    dtype="f8",
                    chunks=(rows, min(n_lines, 16384)),
                    compression="gzip",
                )

        for start in range(0, n_samples, block):
            stop = min(start + block, n_samples)
            buf = {k: np.zeros((stop - start, n_lambda)) for k in spec_names}
            if lines:
                lbuf = {
                    k: np.zeros((stop - start, n_lines)) for k in line_dsets
                }
            for j, i in enumerate(range(start, stop)):
                cont_prefix = cont_dir / str(i)
                if not cont_prefix.with_suffix(".cont").exists() or (
                    lines
                    and not all(
                        (cont_dir / f"{i}.{e}").exists() for e in line_exts
                    )
                ):
                    missing.append(i)
                    continue

                try:
                    spec_dict = cloudy.read_continuum(
                        str(cont_prefix), return_dict=True
                    )
                    if lines:
                        elin = {
                            e: read_elin(cont_dir / f"{i}.{e}")
                            for e in line_exts
                        }
                except Exception as exc:  # noqa: BLE001 - skip any bad file
                    print(f"Warning: could not read sample {i}: {exc}")
                    bad.append(i)
                    continue

                # Reject samples whose wavelength grid differs from the
                # reference in either length or values; a same-length but
                # shifted grid would otherwise be silently mis-aligned.
                if len(spec_dict["lam"]) != n_lambda or not np.allclose(
                    spec_dict["lam"], lam
                ):
                    print(
                        f"Warning: sample {i} wavelength grid does not match "
                        "the reference grid; skipping"
                    )
                    bad.append(i)
                    continue
                if lines and any(ids != line_ids for ids, _ in elin.values()):
                    print(f"Warning: sample {i} line list differs; skipping")
                    bad.append(i)
                    continue

                norm = 1.0
                if normalise:
                    q_cloudy = ionising_rate(lam, spec_dict["incident"])
                    norm = q_ref[i] / q_cloudy
                norm_all[i] = norm
                for spec_name in spec_names:
                    buf[spec_name][j] = spec_dict[spec_name] * norm

                if lines:
                    lbuf["luminosity"][j] = (
                        elin["emergent_elin"][1][line_order] * norm
                    )
                    if intrinsic:
                        lbuf["intrinsic_luminosity"][j] = (
                            elin["intrinsic_elin"][1][line_order] * norm
                        )
                    cont = {
                        "incident": spec_dict["incident"],
                        "transmitted": spec_dict["transmitted"],
                        "nebular_continuum": spec_dict["nebular_continuum"],
                    }
                    cont["total_continuum"] = (
                        cont["nebular_continuum"] + cont["transmitted"]
                    )
                    for k in LINE_CONTINUA:
                        lbuf[k][j] = (
                            np.interp(line_lam[line_order], lam, cont[k])
                            * norm
                        )
                valid_samples[i] = True

            for spec_name in spec_names:
                sg[spec_name][start:stop] = buf[spec_name]
            if lines:
                for k in line_dsets:
                    lg[k][start:stop] = lbuf[k]
            print(f"  {stop}/{n_samples} samples read", flush=True)

        # Save wavelength grid (Angstroms, ascending)
        if "wavelength" in hf:
            del hf["wavelength"]
        hf.create_dataset("wavelength", data=lam, compression="gzip")

        # Valid samples mask: missing and unreadable/mismatched files are
        # excluded so all-zero rows never leak into downstream training.
        if "valid_samples" in hf:
            del hf["valid_samples"]
        hf.create_dataset("valid_samples", data=valid_samples)

        _write(
            hf,
            "normalisation",
            norm_all,
            "dimensionless",
            "Q(H) of incident_lnu / Q(H) of the Cloudy incident spectrum"
            if normalise
            else "not normalised (unity)",
        )

        # Update metadata
        n_valid = int(valid_samples.sum())
        hf.attrs["n_wavelength"] = n_lambda
        hf.attrs["n_valid_samples"] = n_valid
        hf.attrs["spec_names"] = list(spec_names)
        hf.attrs["normalised_by_q"] = normalise
        hf.attrs["n_lines"] = n_lines if lines else 0

    if missing:
        print(
            f"\n{len(missing)} samples with missing files (failed/absent "
            "runs): "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if bad:
        print(
            f"{len(bad)} unreadable/inconsistent samples: "
            f"{bad[:10]}{'...' if len(bad) > 10 else ''}"
        )
    print(
        f"\nValid samples: {n_valid}/{n_samples} ({n_samples - n_valid} "
        "excluded)"
    )
    print(
        f"Successfully added spectra ({', '.join(spec_names)})"
        f"{f' and {n_lines} lines' if lines else ''} to {output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect Cloudy Sobol outputs into HDF5"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to Cloudy output directory",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default=None,
        help="HDF5 grid file path (default: auto-detected from "
        "output_dir name)",
    )

    parser.add_argument(
        "--spec-names",
        type=str,
        nargs="+",
        required=False,
        default=list(DEFAULT_SPECTRA),
        choices=list(AVAILABLE_SPECTRA),
        help="Spectra to save (default: incident transmitted nebular "
        "linecont).",
    )
    parser.add_argument(
        "--no-lines",
        action="store_true",
        help="Do not collect line luminosities (.emergent_elin).",
    )
    parser.add_argument(
        "--intrinsic",
        action="store_true",
        help="Also store intrinsic line luminosities (.intrinsic_elin).",
    )
    parser.add_argument(
        "--no-normalise",
        action="store_true",
        help="Keep raw Cloudy values instead of normalising by Q(H).",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=100,
        help="Samples read between writes (memory ~ block x n_wavelength).",
    )

    args = parser.parse_args()

    collect_sobol_outputs(
        args.output_dir,
        args.output_file,
        spec_names=tuple(args.spec_names),
        lines=not args.no_lines,
        intrinsic=args.intrinsic,
        normalise=not args.no_normalise,
        block=args.block,
    )
