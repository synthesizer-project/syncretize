"""Build a 'full' Cloudy line list: every line Cloudy transfers.

The procedure (see ../README.md) has three steps, run from this directory
(create_linelist/full_linelist/):

1. Run line_labels.in (`sh run.sh line_labels`), a one-zone model that saves
   Cloudy's line labels, and build the unfiltered list from them:
       python full_linelist.py build line_labels --unfiltered \
           --add ../linelist-standard.dat

2. Run the reference models in ../stellar_standard/ and ../agn_standard/
   (which save their line luminosities for the unfiltered list), then find
   the labels to exclude:
       python full_linelist.py check linelist-unfiltered.dat \
           ../stellar_standard/{1..4} ../agn_standard/{1..3} \
           --keep ../linelist-standard.dat

3. Build the final list:
       python full_linelist.py build line_labels \
           --add ../linelist-standard.dat --exclude exclude-c25.00.txt

The Cloudy version (e.g. c25.00, as in the grid parameter files) is read from
the header of the runs' main output ({run}.out). It names the outputs
(exclude-{version}.txt here, linelist-full-{version}.dat in the cloudy
package directory next to the other line lists), and runs or
exclusion files from another version are refused. Builds that do not print a
release number (git builds print e.g. 'Cloudy (master, abc123)') need
--version.

`build` keeps every transferred line ('# type: t'), i.e. the real lines whose
sum is the line contribution (`linecont`) of the saved continuum. It drops:
  - information / heating / cooling / recombination-contribution entries
    (types i, h, c, r);
  - the 'bands' and 'continua' sections (sums over lines or continuum bands,
    which would double count);
  - labels without a defined wavelength ('?');
  - repeated labels: Cloudy cannot distinguish lines whose labels coincide at
    the printed precision (mostly inner-shell satellites), so only one copy is
    kept;
  - labels whose species has a qualifier (e.g. 'H  1 M1', 'Ne10 M1':
    magnetic-dipole components of H-/He-like Lyman alpha; 'O  2R':
    recombination-only components), which Cloudy's line-list parser cannot
    read back ("failed to read wavelength");
  - labels in the --exclude files.
Lines from --add files (e.g. synthesizer's standard list) are always included,
even if excluded, as in earlier grids.

`check` finds the labels to exclude, of two kinds:
  - ambiguous: labels that Cloudy matches to several lines, or to no exact
    line, at the printed precision. Their luminosity would be one arbitrary
    component. These are read from the warnings in the runs' main output
    ("multiple matching lines found", "no exact matching lines found").
  - duplicates: labels that Cloudy resolves to the same line as another label
    (e.g. c25's 'N  2 121.769m' and the standard list's 'N  2 121.767m'),
    which would count that line twice in sums over lines. They cannot be
    predicted from the labels alone. Two labels are duplicates if they have
    the same species, wavelengths within rtol (relative) and identical
    luminosities in every run, with a non-zero luminosity in at least one.
    From each group one label is kept: one from --keep (so that the familiar
    ids are retained), or otherwise the first. Ambiguous labels are not
    considered, unless they are in --keep (those are always included).
The runs should span different physical conditions, so that coincidental
equalities are ruled out; lines that are zero in every run cannot be tested.

Labels are written exactly as Cloudy prints them (fixed-width species field,
e.g. 'NO                  2183.94c'), in Cloudy's label order. The parser
needs the species padded to 4 characters: 'NO 2001.47m' fails and
'NO  2001.47m' works. 'H 1' for 'H  1' logs a warning per line. Species
longer than 4 characters (molecules such as CH3OH, ^13CO) must be quoted
('"CH3OH" 8.84723c'). Checked with c25.00.

The labels use the wavelength convention of the labels run (air above
2000 A, Cloudy's default, for line_labels.in), so the list is for runs in
that convention: the reference runs with `print line vacuum` cannot use it.

Labels are compared in their single-spaced form without quotes, so the
standard list's '^13CO 679.978m' matches '"^13CO" 679.978m'. This is also how
the line ids are stored in the grid (as in synthesizer).
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The cloudy package directory, where the grid line lists live
CLOUDY_DIR = HERE.parents[1]

# Cloudy c25.00 warnings for a line-list label without a unique match
# (LineSave::findline in lines.cpp)
AMBIGUOUS = re.compile(
    r"WARNING: (?:multiple matching lines found in search for"
    r"|no exact matching lines found for) (.*)$"
)
# Version in the header of a Cloudy main output: 'Cloudy 25.00' for a release
# (version.cpp); git builds print '(branch, commit)' instead of the number
VERSION = re.compile(r"^\s*Cloudy (\d\d\.\d\d)\b")
# Version recorded in the header of an exclusion file written by check
EXCLUDE_VERSION = re.compile(r"^# cloudy_version: (\S+)")


def norm(label):
    """Single-spaced label without quotes.

    For example '"^13CO" 2719.67m' -> '^13CO 2719.67m'.
    """
    return " ".join(label.replace('"', " ").split())


def read_labels(path):
    """Normalised labels of a list file (one per line, '#' comments)."""
    return {norm(r) for r in open(path) if r.strip() and not r.startswith("#")}


def wavelength(label):
    """Wavelength in Angstrom of a normalised label."""
    w = label.split()[-1]
    return float(w[:-1]) * {"A": 1.0, "m": 1e4, "c": 1e8}[w[-1]]


def read_elin(path):
    """Luminosities of a `save line list ... column` file, in list order."""
    return np.array([float(r.split("\t")[1]) for r in open(path) if "\t" in r])


def cloudy_version(runs, version=None):
    """Cloudy version (e.g. 'c25.00') of the runs, from their main output.

    Raises an error if the runs were made with different versions, or if the
    version is not printed and none is given.
    """
    found = {}
    for run in runs:
        with open(f"{run}.out") as f:
            for _, row in zip(range(200), f):
                m = VERSION.match(row)
                if m:
                    found[run] = f"c{m.group(1)}"
                    break
    versions = set(found.values())
    if len(versions) > 1:
        sys.exit(f"runs made with different Cloudy versions: {found}")
    if version is not None:
        if versions and versions != {version}:
            sys.exit(f"--version {version}, but the runs used {found}")
        return version
    if len(found) < len(runs) or not versions:
        sys.exit(
            "no Cloudy release number in the main output of "
            f"{sorted(set(runs) - set(found))}: give --version"
        )
    return versions.pop()


def build(
    labels_run,
    out=None,
    add=(),
    exclude=(),
    unfiltered=False,
    version=None,
):
    """Write the line list from the labels saved by labels_run.

    Args:
        labels_run (str): run that saved Cloudy's labels: reads
            {labels_run}.labels and the version from {labels_run}.out.
        out (str): list to write. Default linelist-full-{version}.dat in
            the cloudy package directory, or
            linelist-unfiltered.dat here if unfiltered.
        add (list): line lists whose lines are always included.
        exclude (list): files of labels to drop (output of check).
        unfiltered (bool): build the list for check (no exclusions).
        version (str): Cloudy version, if not printed in {labels_run}.out.

    Returns:
        Path of the list written.
    """
    version = cloudy_version([labels_run], version)
    if unfiltered and exclude:
        sys.exit("an unfiltered list takes no exclusion files")
    if not unfiltered and not exclude:
        sys.exit(
            "no exclusion files: give the output of check with --exclude, "
            "or build the unfiltered list with --unfiltered"
        )
    if out is None:
        out = (
            HERE / "linelist-unfiltered.dat"
            if unfiltered
            else CLOUDY_DIR / f"linelist-full-{version}.dat"
        )

    excluded = set()
    for path in exclude:
        for row in open(path):
            m = EXCLUDE_VERSION.match(row)
            if m and m.group(1) != version:
                sys.exit(f"{path} is for Cloudy {m.group(1)}, not {version}")
        excluded |= read_labels(path)

    # seen: labels written; dropped: excluded labels (kept apart from seen so
    # that add can still include them)
    section, seen, dropped, lines = "", set(), set(), []
    n_t = n_dup = n_nowl = n_qual = n_exc = 0
    for row in open(f"{labels_run}.labels"):
        if row.startswith("####"):
            section = row.split("\t")[1].strip(". \n")
            continue
        if row.startswith("#") or "\t" not in row:
            continue
        m = re.search(r"# type: (\w)", row)
        if not m or m.group(1) != "t" or section in ("bands", "continua"):
            continue
        n_t += 1
        raw = row.split("\t")[1].strip()
        species, wl = raw.rsplit(None, 1)
        species = species.strip()
        label = norm(raw)
        if wl.endswith("?"):
            n_nowl += 1
            continue
        tok = species.split()
        if len(tok) > 2 or any(not t.isdigit() for t in tok[1:]):
            n_qual += 1
            continue
        if label in seen or label in dropped:
            n_dup += 1
            continue
        if label in excluded:
            n_exc += 1
            dropped.add(label)
            continue
        seen.add(label)
        # species longer than 4 characters (molecules, e.g. CH3OH, ^13CO)
        # must be quoted
        lines.append(
            f'"{species}" {wl}'
            if len(species) > 4 and not species.startswith('"')
            else raw
        )
    n_from_labels = len(lines)

    for path in add:
        for row in open(path):
            label = norm(row)
            if label and label not in seen:
                seen.add(label)
                species, wl = label.strip().rsplit(None, 1)
                if len(species.split()) == 2:
                    # pad to Cloudy's 4-character species field (avoids a
                    # warning per line)
                    el, ion = species.split()
                    species = f"{el:<2s}{ion:>2s}"
                elif len(species) > 4:
                    # molecules must be quoted (norm strips the quotes)
                    species = f'"{species}"'
                lines.append(f"{species} {wl}")

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(
        f"Cloudy {version}: {n_t} transferred lines; dropped {n_nowl} "
        f"without wavelength, {n_qual} with a species qualifier, {n_dup} "
        f"repeated labels and {n_exc} excluded; {n_from_labels} kept, "
        f"{len(lines) - n_from_labels} added from {list(add)}; "
        f"wrote {len(lines)} to {out}"
    )
    return Path(out)


def find_ambiguous(runs):
    """Labels with an ambiguity warning in any run's main output."""
    found = set()
    for run in runs:
        for row in open(f"{run}.out"):
            m = AMBIGUOUS.match(row.strip())
            if m:
                found.add(norm(m.group(1)))
    return found


def find_duplicates(labels, lum, keep, skip, rtol):
    """Labels resolving to the same line as another label (see docstring)."""
    by_species = defaultdict(list)
    for i, lab in enumerate(labels):
        by_species[" ".join(lab.split()[:-1])].append(i)

    drop, n_groups = [], 0
    for idx in by_species.values():
        idx = sorted(idx, key=lambda i: wavelength(labels[i]))
        used = set()
        for a_pos, a in enumerate(idx):
            if a in used or labels[a] in skip or not lum[:, a].any():
                continue
            group = [a]
            for b in idx[a_pos + 1 :]:
                if wavelength(labels[b]) > wavelength(labels[a]) * (1 + rtol):
                    break
                # identical labels are repeats, which build removes itself
                if labels[b] == labels[a] or labels[b] in skip:
                    continue
                if b not in used and np.array_equal(lum[:, a], lum[:, b]):
                    group.append(b)
            if len(group) > 1:
                n_groups += 1
                used.update(group)
                kept = next(
                    (i for i in group if labels[i] in keep), min(group)
                )
                drop += [labels[i] for i in sorted(group) if i != kept]
                print(
                    f"kept {labels[kept]!r}; same line as "
                    + ", ".join(repr(labels[i]) for i in group if i != kept)
                )
    return drop, n_groups


def check(linelist, runs, out=None, keep=(), rtol=1e-3, version=None):
    """Write the labels to exclude, found from runs made with linelist.

    Args:
        linelist (str): list used for the runs (normally the unfiltered one).
        runs (list): runs made with it: reads {run}.out and
            {run}.emergent_elin.
        out (str): file to write. Default exclude-{version}.txt here.
        keep (list): line lists whose labels are preferred when keeping one.
        rtol (float): relative wavelength tolerance for duplicates.
        version (str): Cloudy version, if not printed in the {run}.out.

    Returns:
        Path of the file written.
    """
    version = cloudy_version(runs, version)
    if out is None:
        out = HERE / f"exclude-{version}.txt"
    labels = [norm(r) for r in open(linelist) if r.strip()]
    lum = np.array([read_elin(f"{run}.emergent_elin") for run in runs])
    if lum.shape[1] != len(labels):
        sys.exit(
            f"{runs[0]}.emergent_elin has {lum.shape[1]} lines, "
            f"{linelist} has {len(labels)}"
        )
    keep_labels = set()
    for path in keep:
        keep_labels |= read_labels(path)

    ambiguous = find_ambiguous(runs)
    duplicates, n_groups = find_duplicates(
        labels, lum, keep_labels, ambiguous - keep_labels, rtol
    )
    # ambiguous labels in Cloudy's label order
    ambiguous = [lab for lab in dict.fromkeys(labels) if lab in ambiguous]

    with open(out, "w") as f:
        f.write(
            "# Labels to exclude from the full line list (full_linelist.py "
            "check).\n"
            f"# cloudy_version: {version}\n"
            f"# {len(runs)} runs with {Path(linelist).name}, rtol {rtol}: "
            f"{' '.join(str(r) for r in runs)}\n#\n"
            f"# Ambiguous ({len(ambiguous)}): Cloudy matches them to several "
            "lines, or to no exact\n# line, at the printed precision.\n"
        )
        f.write("".join(f"{lab}\n" for lab in ambiguous))
        f.write(
            f"#\n# Duplicates ({len(duplicates)}, in {n_groups} groups): "
            "Cloudy resolves them to the same\n# line as another label "
            "of the list.\n"
        )
        f.write("".join(f"{lab}\n" for lab in duplicates))
    print(
        f"Cloudy {version}: {len(ambiguous)} ambiguous labels and "
        f"{len(duplicates)} duplicates in {len(runs)} runs; wrote {out}"
    )
    return Path(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="write the line list")
    b.add_argument(
        "labels_run",
        help="run that saved the labels: reads {run}.labels and {run}.out",
    )
    b.add_argument(
        "-o",
        "--out",
        help="list to write (default linelist-full-{version}.dat in the "
        "cloudy package directory, or "
        "linelist-unfiltered.dat with --unfiltered)",
    )
    b.add_argument(
        "--add",
        nargs="*",
        default=[],
        help="line lists whose lines are always included",
    )
    b.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="files of labels to drop (the output of check)",
    )
    b.add_argument(
        "--unfiltered",
        action="store_true",
        help="build the unfiltered list for check",
    )

    c = sub.add_parser("check", help="write the labels to exclude")
    c.add_argument("linelist", help="line list used for the runs")
    c.add_argument(
        "runs",
        nargs="+",
        help="runs made with this list: reads {run}.out and "
        "{run}.emergent_elin",
    )
    c.add_argument(
        "-o", "--out", help="file to write (default exclude-{version}.txt)"
    )
    c.add_argument(
        "--keep",
        nargs="*",
        default=[],
        help="line lists whose labels are preferred when keeping one",
    )
    c.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="relative wavelength tolerance (default 1e-3)",
    )

    for p in (b, c):
        p.add_argument(
            "--version",
            help="Cloudy version (e.g. c25.00), if the runs do not print it",
        )

    args = vars(ap.parse_args())
    command = args.pop("command")
    {"build": build, "check": check}[command](**args)


if __name__ == "__main__":
    main()
