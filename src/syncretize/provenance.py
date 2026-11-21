"""What a grid records about how it was made, so it can be remade from it.

Every grid written with ``GridFile`` records:

    Software          the Python version, the platform and the version of
                      each package that shapes a grid; for syncretize and
                      synthesizer also their git commit and, when they had
                      uncommitted changes, the changes themselves (as a diff)
    SourceData        each source data file the incident model read, with
                      its size and SHA-256 checksum (incident grids)

and a Cloudy grid also:

    IncidentGrid      a copy of its incident grid's provenance (Model,
                      ParameterFile, RunConfig, SourceData and Software, and
                      its root attributes), with the incident grid's SHA-256
    CloudyBuild       the Cloudy version (and its date) the models ran with,
                      and (from the Runner) the build directory, with the
                      SHA-256 of its executable and of its data directory
    CloudyParams      the Cloudy parameter files, verbatim

The source data files are found by watching what the process opens for
reading (with an audit hook, see ``record_reads``), so the incident models
don't each have to say what they read.
"""

import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

#: The packages whose versions every grid records
PACKAGES = (
    "syncretize",
    "synthesizer",
    "numpy",
    "scipy",
    "h5py",
    "unyt",
    "astropy",
    "fsps",
)

#: The packages whose git commits (and uncommitted changes) every grid also
#: records, when run from a git checkout
CHECKOUTS = ("syncretize", "synthesizer")

#: Files the process reads that aren't source data: code, and configs (which
#: the grid stores verbatim anyway)
NOT_SOURCE_DATA = (".py", ".pyc", ".pyi", ".so", ".pth", ".yaml", ".yml")

#: Every file the process has opened for reading, in order (see record_reads)
_reads = {}


def _audit(event, args):
    """Note each file opened for reading (an audit hook; see record_reads)."""
    if event != "open":
        return
    path, mode, flags = args
    if mode is None:
        # os.open gives flags rather than a mode
        reading = isinstance(flags, int) and not flags & (
            os.O_WRONLY | os.O_RDWR
        )
    else:
        reading = isinstance(mode, str) and not set(mode) & set("wax+")
    if reading and isinstance(path, (str, bytes, os.PathLike)):
        _reads[os.fsdecode(path)] = None


def record_reads():
    """Start noting every file the process opens for reading.

    Python's audit hooks see every file opened with open, io.open or os.open,
    whichever library opens it (files read by compiled code, e.g. HDF5 or
    Fortran, aren't seen). A hook can't be removed, so this is done once.
    """
    if not getattr(record_reads, "started", False):
        sys.addaudithook(_audit)
        record_reads.started = True


def source_files(exclude=()):
    """The source data files the process has read.

    Code, configs, and files inside Python's installation or the recorded
    packages aren't source data, nor are files that no longer exist.

    Args:
        exclude (iterable of str)
            Files to leave out (e.g. the parameter file).

    Returns:
        list of str
            Their absolute, resolved paths, in the order first read.
    """
    skip = {os.path.realpath(path) for path in exclude}
    installed = tuple(
        os.path.realpath(prefix) + os.sep
        for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix}
    ) + tuple(_package_dir(name) + os.sep for name in CHECKOUTS)
    files = []
    for path in list(_reads):
        real = os.path.realpath(path)
        if (
            real in skip
            or real in files
            or real.endswith(NOT_SOURCE_DATA)
            or real.startswith(installed + ("/proc/", "/sys/", "/dev/"))
            or not os.path.isfile(real)
        ):
            continue
        files.append(real)
    return files


def sha256(path):
    """A file's SHA-256 checksum (hex)."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(directory):
    """One SHA-256 checksum for every file in a directory (hex).

    The checksum covers each file's path (relative to the directory) and
    contents, in path order, so it changes if any file is added, removed,
    renamed or changed.
    """
    digest = hashlib.sha256()
    paths = sorted(
        os.path.relpath(os.path.join(root, name), directory)
        for root, _, names in os.walk(directory)
        for name in names
    )
    for path in paths:
        digest.update(path.encode() + b"\0")
        digest.update(sha256(os.path.join(directory, path)).encode())
    return digest.hexdigest()


def _package_dir(name):
    """A package's directory (empty if it isn't installed)."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return ""
    if spec is None or not spec.submodule_search_locations:
        return ""
    return os.path.realpath(list(spec.submodule_search_locations)[0])


def _git(directory, *args):
    """Run git in a directory, returning its output (None if it fails)."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


@functools.cache
def checkout(name):
    """A package's git commit and uncommitted changes, if run from git.

    Args:
        name (str)
            The package, e.g. "syncretize".

    Returns:
        dict or None
            "commit" (the full hash) and "diff" (``git diff HEAD``: the
            changes to tracked files not yet committed, empty if none);
            None when the package isn't in a git checkout (e.g. installed
            from a wheel, whose version names its commit).
    """
    directory = _package_dir(name)
    # The package must be tracked in the checkout, not just inside one (as
    # a virtual environment inside a repository is)
    tracked = directory and _git(
        directory, "ls-files", "--error-unmatch", "__init__.py"
    )
    if not tracked:
        return None
    commit = _git(directory, "rev-parse", "HEAD")
    diff = _git(directory, "diff", "HEAD")
    if commit is None or diff is None:
        return None
    return {"commit": commit.strip(), "diff": diff}


def version(name):
    """A package's version, from git when run from a checkout.

    A checkout's version file is written when it is installed, so goes out
    of date as commits are made; there the version is worked out from git
    (with setuptools_scm, as the version file is), and failing that the
    commit is added to the installed version.

    Args:
        name (str)
            The package.

    Returns:
        str or None
            Its version (None if it isn't installed or in a checkout).
    """
    # The code imported (which can differ from the package installed, e.g.
    # with PYTHONPATH) holds its own version, but importing other packages
    # can have side effects (fsps needs SPS_HOME), so theirs come from the
    # installed package's metadata
    try:
        if name in CHECKOUTS:
            installed = importlib.import_module(f"{name}._version").__version__
        else:
            installed = importlib.metadata.version(name)
    except (ImportError, importlib.metadata.PackageNotFoundError):
        # Not installed (a checkout used through PYTHONPATH has no version
        # file, which is written on installing)
        installed = None
    git = checkout(name)
    if git is None:
        return installed
    try:
        from setuptools_scm import get_version

        return get_version(
            root=_git(
                _package_dir(name), "rev-parse", "--show-toplevel"
            ).strip()
        )
    except Exception:
        commit = f"commit {git['commit'][:9]}"
        commit += " with uncommitted changes" if git["diff"] else ""
        return f"{installed} ({commit})" if installed else commit


@functools.cache
def software():
    """Everything ``Software`` records (see the module docstring).

    Returns:
        dict
            "python", "platform", each installed package's version (by its
            name) and, for each package run from a git checkout,
            "<name>_commit" and "<name>_diff".
    """
    info = {"python": sys.version, "platform": platform.platform()}
    for name in PACKAGES:
        found = version(name)
        if found is not None:
            info[name] = found
    for name in CHECKOUTS:
        git = checkout(name)
        if git is not None:
            info[f"{name}_commit"] = git["commit"]
            info[f"{name}_diff"] = git["diff"]
    return info


def cloudy_version(out_file):
    """The Cloudy version (and its date) a Cloudy output was made with.

    Args:
        out_file (str)
            A Cloudy .out file, which starts with e.g. "Cloudy 23.01" and a
            line of asterisks holding the version's date ("**23Jul25**").

    Returns:
        dict
            "version" and "date" (each "unknown" if not found).
    """
    with open(out_file, errors="replace") as file:
        head = "".join(file.readline() for _ in range(10))
    found = re.search(r"Cloudy\s+(\S+)", head)
    date = re.search(r"\*{5,}\s*(\w+)\s*\*{5,}", head)
    return {
        "version": found.group(1) if found else "unknown",
        "date": date.group(1) if date else "unknown",
    }


def find_out_file(directory):
    """A Cloudy .out file under a directory (None if there is none)."""
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            if name.endswith(".out"):
                return str(Path(root) / name)
    return None
