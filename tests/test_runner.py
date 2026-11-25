"""Tests for the Runner's config and its local stages."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from unyt import Angstrom, Hz, dimensionless, erg, s, yr

from syncretize.incident.builder import (
    GridSpec,
    Output,
    build_grid,
    table_point,
)
from syncretize.runner import Runner, perturb
from syncretize.runner.cloudy import work
from syncretize.runner.config import load_run, seconds, slurm_time
from syncretize.runner.queue import Queue

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "parameter_files" / "runs" / "run_template.yaml"
BLACKBODY = (
    ROOT
    / "parameter_files"
    / "incident"
    / "blackholes"
    / "blackbody"
    / "generate_incident_grid_template.yaml"
)


def write_run(tmp_path, **sections):
    """Write a run config with a blackbody incident stage.

    Args:
        tmp_path (pathlib.Path)
            The directory to write run.yaml in.
        **sections
            Sections to add to the config, or to replace its Run and
            Incident sections with, each a dict of its values.

    Returns:
        pathlib.Path
            The path of the run.yaml written.
    """
    config = {
        "Run": {"name": "test", "work_dir": "work", "grid_dir": "grids"},
        "Incident": {
            "model": "blackholes/blackbody/generate_incident_grid",
            "config": str(BLACKBODY),
        },
    }
    for section, values in sections.items():
        config[section] = values
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_template_loads():
    """The shipped run template loads, with its defaults."""
    config = load_run(TEMPLATE)
    assert config.run["cloudy"]["walltime_growth"] == 1.5
    assert config.sections == {}
    assert config.incident["config"].endswith(
        "blackbody/generate_incident_grid_template.yaml"
    )


def test_defaults_and_relative_paths(tmp_path):
    """Defaults are filled in and paths are made relative to the config."""
    config = load_run(write_run(tmp_path))
    assert config.run["incident"]["cores"] == 1
    assert config.run["cloudy"]["max_retries"] == 2
    assert config.run["grid_dir"] == str(tmp_path / "grids")
    assert config.run["input_dir"] is None


@pytest.mark.parametrize(
    "sections, message",
    [
        ({"Run": {"name": "x", "grid_dir": "g"}}, "needs work_dir"),
        (
            {"Run": {"name": "x", "work_dir": "w", "grid_dir": "g", "y": 1}},
            "unknown keys",
        ),
        (
            {
                "Run": {
                    "name": "x",
                    "work_dir": "w",
                    "grid_dir": "g",
                    "incident": {"cores": 0},
                }
            },
            "at least 1",
        ),
        ({"Incident": {"model": "blackholes/nope"}}, "needs model and config"),
        (
            {"Incident": {"model": "blackholes/nope", "config": "c.yaml"}},
            "no module",
        ),
        ({"Incident": {"grid": "g.hdf5"}}, "needs a Cloudy section"),
        ({"Cloudy": {"config": "c23.01-sps"}}, "needs executable"),
        (
            {
                "Run": {
                    "name": "x",
                    "work_dir": "w",
                    "grid_dir": "g",
                    "cloudy": {"margin": "15 minutes"},
                }
            },
            "margin",
        ),
        (
            {
                "Run": {
                    "name": "x",
                    "work_dir": "w",
                    "grid_dir": "g",
                    "cloudy": {"walltime": "00:10:00"},
                }
            },
            "no time",
        ),
        (
            {
                "Run": {
                    "name": "x",
                    "work_dir": "w",
                    "grid_dir": "g",
                    "cloudy": {"margin": "00:01:00"},
                }
            },
            "longer than 00:02:00",
        ),
        ({"Extra": {}}, "unknown"),
        ({"Cloudy0": {}}, "unknown"),
        (
            {
                "Cloudy": {"config": "c23.01-sps", "executable": "/c"},
                "Cloudy1": {"config": "c23.01-nlr", "executable": "/c"},
            },
            "the same section",
        ),
        (
            {
                "Cloudy": {"config": "c23.01-sps", "executable": "/c"},
                "Cloudy2": {"config": "c23.01-sps", "executable": "/c"},
            },
            "would both make the grid c23.01-sps",
        ),
        ({"Cloudy3": {"config": "c23.01-sps"}}, "Cloudy3 needs executable"),
    ],
)
def test_bad_configs(tmp_path, sections, message):
    """Each bad run config is refused, with a message saying why."""
    with pytest.raises(ValueError, match=message):
        load_run(write_run(tmp_path, **sections))


def test_seconds():
    """Walltimes convert to seconds and back to Slurm's format."""
    assert seconds("00:15:00") == 900
    assert seconds("3-00:00:00") == 3 * 86400
    assert seconds(43200) == 43200  # YAML's reading of 12:00:00
    assert slurm_time(seconds("1-02:03:04")) == "1-02:03:04"


def test_local_incident_run(tmp_path):
    """A local run makes the incident grid, with its run config."""
    path = write_run(
        tmp_path,
        Run={
            "name": "test",
            "work_dir": "work",
            "grid_dir": "grids",
            "incident": {"cores": 2},
        },
    )
    subprocess.run(
        [sys.executable, "-m", "syncretize.runner", "run", str(path)]
        + ["--local"],
        check=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    with h5py.File(tmp_path / "grids" / "blackbody.hdf5", "r") as hdf:
        assert list(hdf.attrs["axes"]) == ["blackbody_temperature"]
        assert hdf["spectra/incident"].shape[0] == 5
        assert "blackbody" in hdf["ParameterFile/contents"][()].decode()
        assert hdf["RunConfig/Run"].attrs["name"] == "test"
        assert hdf["RunConfig"].attrs["Cloudy"] == "None"


def fake_incident_grid(grid_dir):
    """Write a 2 x 2 (age, metallicity) incident grid.

    Args:
        grid_dir (str or pathlib.Path)
            The directory to write the grid in.

    Returns:
        str
            The path of the grid written.
    """
    lam = np.geomspace(10, 1e6, 2000)
    ages, metallicities = np.array([1e6, 1e7]), np.array([0.001, 0.02])
    spectra = np.array(
        [
            [
                1e20
                * (lam / 1e3) ** 2
                * np.exp(-912e6 * (1 + 10 * z) / a / lam)
                for z in metallicities
            ]
            for a in ages
        ]
    )
    spec = GridSpec(
        grid_name="fake-sps",
        model={"sps_name": "fake"},
        axes={
            "ages": ages * yr,
            "metallicities": metallicities * dimensionless,
        },
        wavelength=lam * Angstrom,
        outputs={"incident": Output(units=erg / s / Hz)},
        log_on_read={"ages": True, "metallicities": False},
        data={"outputs": {"incident": spectra}},
    )
    return build_grid(spec, table_point, str(grid_dir), verbose=False)


def test_cloudy_setup_fills_the_queue(tmp_path):
    """Cloudy setup writes the inputs and queues every model, kept on redo."""
    grid = fake_incident_grid(tmp_path)
    runner = Runner(
        write_run(
            tmp_path,
            Incident={"grid": grid},
            Cloudy={
                "config": "c23.01-sps-grid_limited",
                "executable": "/cloudy",
            },
        )
    )
    (section,) = runner.sections
    section.setup(grid)

    # 2 x 2 incident models, each with 2 x 2 x 3 x 2 x 2 photoionisation
    # models, in the section's own directory
    work = tmp_path / "work" / "c23.01-sps-grid_limited"
    assert Queue(work / "queue").counts()["todo"] == 4 * 48
    cloudy_dir = work / "cloudy" / "fake-sps_cloudy-c23.01-sps-grid_limited"
    assert (cloudy_dir / "3" / "47.in").is_file()
    assert (cloudy_dir / "0" / "linelist.dat").is_file()
    state = yaml.safe_load((work / "state.yaml").read_text())
    assert state["cloudy_dir"] == str(cloudy_dir)
    assert state["models"]["todo"] == 4 * 48
    models = json.loads((work / "models.json").read_text())
    assert len(models) == 4 * 48
    assert models["3_47"] == {"state": "todo", "attempts": []}

    # Setting up again keeps each model's state
    queue = Queue(work / "queue")
    (model,) = queue.claim("1.0")
    queue.done(model, "1.0")
    section.setup(grid)
    assert queue.counts() == {
        "todo": 4 * 48 - 1,
        "claimed": 0,
        "done": 1,
        "failed": 0,
    }


def test_cloudy_build_must_match_the_config(tmp_path):
    """A Cloudy build named for another version than the config's fails."""
    with pytest.raises(ValueError, match="is Cloudy c25 but"):
        load_run(
            write_run(
                tmp_path,
                Cloudy={"config": "c23.01-sps", "executable": "/cloudy/c25"},
            )
        )
    # A build named after its version, or not named like one, is fine
    for executable in ("/cloudy/c23.01", "/cloudy/build"):
        load_run(
            write_run(
                tmp_path,
                Cloudy={"config": "c23.01-sps", "executable": executable},
            )
        )


def test_unknown_cloudy_config(tmp_path):
    """A Cloudy config that isn't shipped is refused."""
    with pytest.raises(ValueError, match="shipped Cloudy config"):
        load_run(
            write_run(
                tmp_path, Cloudy={"config": "nope", "executable": "/cloudy"}
            )
        )


FAKE_CLOUDY = f"""#!{sys.executable}
# A stand-in for cloudy.exe -r <j>, run in <cloudy_dir>/<i>. FAKE_CLOUDY maps
# a model "<i>_<j>" (or a pattern, e.g. "*" for every model; the first match
# applies) to what it does: "fail" (crash), "fail_once" (crash the first run
# only), "qheat" (crash on Cloudy's grain quantum heating assertion unless
# its input turns quantum heating off), "warn" (finish with a warning),
# "nzone" (finish, stopped by the zone limit), "no_lines" (exit OK without
# lines), "no_output" (stop before writing anything, as when killed early),
# "sleep" (take 0.5 s) or "hang" (take a minute).
import fnmatch, json, os, sys, time
j = sys.argv[2]
model = os.path.basename(os.getcwd()) + "_" + j
behaviours = json.loads(os.environ.get("FAKE_CLOUDY", "{{}}"))
behaviour = next(
    (b for key, b in behaviours.items() if fnmatch.fnmatch(model, key)),
    None,
)
if behaviour == "sleep":
    time.sleep(0.5)
if behaviour == "hang":
    time.sleep(60)
if behaviour == "no_output":
    sys.exit(1)
if behaviour == "fail_once":
    behaviour = "ok" if os.path.exists(j + ".out") else "fail"
if behaviour == "qheat":
    with open(j + ".in") as file:
        behaviour = "ok" if "no qheat" in file.read() else "qheat"
with open(j + ".out", "w") as file:
    file.write(os.environ["CLOUDY_DATA_PATH"] + "\\n")
    if behaviour == "fail":
        file.write(" [Stop in cdMain, something went wrong]\\n")
        sys.exit(1)
    if behaviour == "qheat":
        file.write(" PROBLEM DISASTER\\n An assert has been thrown.\\n")
        file.write(" Failed: wid == 0.\\n It happened in the file ")
        file.write("grains_qheat.cpp at line number 2047\\n")
        file.write("[0x4a0715] _start -- ??:?\\n")
        sys.exit(1)
    file.write(" Start Iteration Number 2\\n Iteration 2 of 10.\\n")
    if behaviour == "nzone":
        file.write("W-Calculation stopped because default number of zones ")
        file.write("reached.\\n >>>> The calculation stopped for ")
        file.write("unintended reasons.\\n")
    if behaviour in ("warn", "nzone"):
        file.write(" Cloudy ends: 1400 zones, 2 iterations, 1 warning.\\n")
        file.write(" [Stop in cdMain, something went wrong]\\n")
    else:
        file.write(" Cloudy ends: 10 zones, 2 iterations.\\n")
        file.write(" [Stop in cdMain, Cloudy exited OK]\\n")
# Continua (.cont: wavelength and 8 nu L_nu columns, blue last) and the
# albedo (.refl, column 4)
import numpy as np
lam = np.geomspace(1e6, 10, 500)
incident = 1e40 * (lam / 1e3) ** -1 * np.exp(-300 / lam)
columns = [lam, incident, 0.5 * incident, 0.1 * incident, 0.6 * incident,
           0.01 * incident, 0.61 * incident, 0, 0.01 * incident]
np.savetxt(j + ".cont", np.column_stack(np.broadcast_arrays(*columns)),
           delimiter="\\t")
np.savetxt(j + ".refl", np.column_stack([lam, lam, lam, lam, 0 * lam + 0.01]),
           delimiter="\\t")
if behaviour != "no_lines":
    # Every line of the model's line list, as "<id>\\t<luminosity>"
    with open("linelist.dat") as file:
        ids = file.read().split("\\n")
    with open(j + ".emergent_elin", "w") as file:
        for n, id in enumerate(filter(None, ids)):
            file.write(f"{{id}}\\t{{1e38 * (n + 1)}}\\n")
"""


def cloudy_run(
    tmp_path,
    monkeypatch,
    behaviours,
    extra=None,
    perturb=None,
    more=None,
    options=None,
    **cloudy,
):
    """A run's Cloudy section that runs the fake Cloudy, and its queue.

    The run's incident grid is the fake one, written to tmp_path; nothing
    is set up or run yet.

    Args:
        tmp_path (pathlib.Path)
            The directory to put the fake Cloudy, the grid and the run in.
        monkeypatch (pytest.MonkeyPatch)
            Sets $FAKE_CLOUDY for the fake Cloudy.
        behaviours (dict)
            What the fake Cloudy does for each model or pattern (see
            FAKE_CLOUDY), e.g. {"1_3": "fail"}.
        extra (dict)
            Cloudy parameters to add, written as an extra_config, or None.
        perturb (dict)
            The Cloudy section's perturb settings, or None (the default),
            so failed models aren't perturbed.
        more (dict)
            More Cloudy sections (e.g. {"Cloudy2": {...}}), whose
            executable is the fake Cloudy, or None.
        options (dict)
            More keys of the Cloudy section (e.g. review), or None.
        **cloudy
            The run's cloudy settings (Run.cloudy), e.g. max_retries=0.

    Returns:
        (Section, Queue)
            The run's first Cloudy section, and its queue.
    """
    exe = tmp_path / "cloudy" / "source" / "cloudy.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text(FAKE_CLOUDY)
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_CLOUDY", json.dumps(behaviours))
    grid = fake_incident_grid(tmp_path)
    extra_config = None
    if extra is not None:
        extra_config = str(tmp_path / "extra.yaml")
        Path(extra_config).write_text(yaml.safe_dump(extra))
    runner = Runner(
        write_run(
            tmp_path,
            Run={
                "name": "test",
                "work_dir": "work",
                "grid_dir": "grids",
                "cloudy": cloudy,
            },
            Incident={"grid": grid},
            Cloudy={
                "config": "c23.01-sps-grid_limited",
                "extra_config": extra_config,
                "executable": str(exe.parents[1]),
                "perturb": perturb,
                **(options or {}),
            },
            **{
                key: {"executable": str(exe.parents[1]), **values}
                for key, values in (more or {}).items()
            },
        )
    )
    section = runner.sections[0]
    return section, section.queue


def test_local_workers_run_every_model(tmp_path, monkeypatch):
    """A local run runs, records and assembles every model, failures too."""
    section, queue = cloudy_run(
        tmp_path,
        monkeypatch,
        {
            "1_3": "fail",
            "2_5": "no_lines",
            "0_9": "fail_once",
            "3_1": "no_output",
        },
    )
    (grid,) = section.runner.run(local=True)

    # Within one round, 0_9 worked when retried and the others failed all
    # 1 + 2 tries
    assert queue.counts() == {
        "todo": 0,
        "claimed": 0,
        "done": 4 * 48 - 3,
        "failed": 3,
    }
    failed = Path(queue.root) / "failed"
    assert sorted(os.listdir(failed)) == ["1_3", "2_5", "3_1"]
    assert (failed / "1_3").read_text() == "3"
    state = section._load_state()
    assert (state["round"], state["next"], state["grid"]) == (1, "done", grid)
    assert state["failures"] == {
        "1_3": "[Stop in cdMain, something went wrong]",
        "2_5": "no .emergent_elin",
        "3_1": "no .out",
    }
    (summary,) = state["rounds"]
    assert (summary["done"], summary["failed"]) == (4 * 48 - 3, 10)

    # Every model's attempts are kept, and the workers' records cleared
    models = section._load_models()
    assert [a["result"] for a in models["0_9"]["attempts"]] == [
        "failed",
        "done",
    ]
    assert models["0_9"]["state"] == "done"
    # with how far Cloudy got
    assert models["0_9"]["attempts"][-1]["iterations"] == 2
    assert len(models["1_3"]["attempts"]) == 3
    assert models["1_3"]["state"] == "failed"
    assert not os.listdir(section._path("state"))

    # run.log tells the run's story; each model's line is only in the job log
    run_log = (tmp_path / "work" / "run.log").read_text()
    for text in (
        "|S|Y|N|C|R|E|T|I|Z|E|",
        "max_retries: 2",
        "48 photoionisation models per incident point",
        "Wrote 192 Cloudy inputs",
        "| round | time limit |",
        "1_3: [Stop in cdMain, something went wrong]",
        "Done: 189 of 192 grid points from Cloudy, 0 interpolated from "
        "perturbed runs, 3 failed",
    ):
        assert text in run_log
    assert "0_9 done" not in run_log

    check_watch(str(tmp_path / "work"))

    # A finished run is resumed by doing nothing, and isn't run again
    assert section.runner.resume(local=True) is None
    with pytest.raises(ValueError, match="use resume"):
        section.runner.run(local=True)
    cloudy_dir = section._load_state()["cloudy_dir"]
    with open(os.path.join(cloudy_dir, "0", "7.out")) as file:
        assert file.readline() == f"{tmp_path}/cloudy/data/:./\n"

    # The final grid loads in synthesizer, with the failures marked
    assert grid == str(
        tmp_path / "grids" / "fake-sps_cloudy-c23.01-sps-grid_limited.hdf5"
    )
    from synthesizer.grid import Grid

    loaded = Grid(os.path.basename(grid), grid_dir=str(tmp_path / "grids"))
    assert loaded.shape[:2] == (2, 2)
    assert np.prod(loaded.shape[:-1]) == 4 * 48  # the last is wavelength
    assert len(loaded.line_ids) > 0
    assert np.nanmax(loaded.spectra["nebular"]) > 0
    with h5py.File(grid, "r") as hdf:
        failures = hdf["failures"][()]
        assert failures.sum() == 3
        # One each at incident points 1, 2 and 3 (the photoionisation index
        # isn't in C order, so isn't checked)
        assert list(failures.reshape(4, 48).sum(axis=1)) == [0, 1, 1, 1]
        # The failed points have no spectra or lines, rather than zeros
        failed = failures.astype(bool)
        nebular = hdf["spectra/nebular"][()]
        lines = hdf["lines/luminosity"][()]
        assert np.isnan(nebular[failed]).all()
        assert np.isnan(lines[failed]).all()
        assert np.isfinite(nebular[~failed]).all()
        assert np.isfinite(lines[~failed]).all()
        assert hdf["RunConfig/Cloudy"].attrs["include_spectra"]
        # How it was made: the Cloudy build, the Cloudy config verbatim and
        # the incident grid's provenance (with its source data)
        build = hdf["CloudyBuild"].attrs
        assert build["build"] == os.path.realpath(tmp_path / "cloudy")
        assert len(build["executable_sha256"]) == 64
        files = hdf["CloudyParams/files"]
        assert "cloudy_version" in files["0"][()].decode()
        incident = hdf["IncidentGrid"]
        assert incident.attrs["file"] == "fake-sps.hdf5"
        # (made here without a parameter file or a run, so without theirs)
        for name in ("Model", "SourceData", "Software"):
            assert name in incident
        assert "syncretize_commit" in hdf["Software"].attrs
        assert "fake-sps" in hdf["RunConfig/contents"][()].decode()


def run_worker(section, deadline, stop_at, max_retries=2):
    """Run one worker of 2 loops with deadlines in seconds from now.

    Args:
        section (Section)
            The set up Cloudy section whose queue the worker runs.
        deadline (float)
            Seconds from now after which no model is claimed.
        stop_at (float)
            Seconds from now at which running models are stopped.
        max_retries (int)
            How many times a failed model is run again.
    """
    now = time.monotonic()
    work(
        section.queue.root,
        section._load_state()["cloudy_dir"],
        section.cloudy["executable"],
        "1",
        slots=2,
        claim=1,
        extensions=["emergent_elin"],
        max_retries=max_retries,
        record_file=section._path("state", "1.jsonl"),
        deadline=now + deadline,
        stop_at=now + stop_at,
    )


def test_workers_stop_claiming_at_their_deadline(tmp_path, monkeypatch):
    """Workers claim no models after their deadline."""
    section, queue = cloudy_run(tmp_path, monkeypatch, {"*": "sleep"})
    section.setup(section.config.incident["grid"])
    start = time.monotonic()
    run_worker(section, deadline=2, stop_at=10)

    # Each loop starts models for 2 s, so runs about 4 of 0.5 s, and the
    # last ends by 2.5 s
    assert time.monotonic() - start < 3
    counts = queue.counts()
    assert 2 <= counts["done"] <= 12
    assert counts["claimed"] == 0 and counts["failed"] == 0


def test_workers_claim_only_models_with_time_to_finish(tmp_path, monkeypatch):
    """Workers don't claim models they likely haven't time to finish."""
    section, queue = cloudy_run(tmp_path, monkeypatch, {"*": "sleep"})
    section.setup(section.config.incident["grid"])
    run_worker(section, deadline=10, stop_at=0.9)

    # After one run of ~0.5 s, under 1.5 x that is left before the stop, so
    # each loop claims nothing more and nothing is stopped
    assert queue.counts()["done"] == 2
    assert queue.counts()["todo"] == 4 * 48 - 2


def test_workers_stop_their_cloudy_runs_at_the_time_limit(
    tmp_path, monkeypatch
):
    """Workers stop Cloudy runs at the time limit and put them back."""
    section, queue = cloudy_run(tmp_path, monkeypatch, {"*": "hang"})
    section.setup(section.config.incident["grid"])
    start = time.monotonic()
    run_worker(section, deadline=0.5, stop_at=1)

    # Both runs stopped at 1 s and were put back without an attempt
    assert time.monotonic() - start < 5
    assert queue.counts() == {
        "todo": 4 * 48,
        "claimed": 0,
        "done": 0,
        "failed": 0,
    }
    # so the next round's time limit is longer
    assert section.finish(lambda owner: False) == "work"
    assert section._load_state()["walltime"] == "0-18:00:00"
    stopped = [
        entry["attempts"][0]
        for entry in section._load_models().values()
        if entry["attempts"]
    ]
    assert len(stopped) == 2
    assert stopped[0]["result"] == "stopped" and stopped[0]["requeued"]

    # Even when the limit can't grow: running out of time isn't a failure
    section._save_state(walltime="3-00:00:00")
    run_worker(section, deadline=0.5, stop_at=1, max_retries=0)
    assert queue.counts()["failed"] == 0


def test_finish_requeues_stopped_workers_models_with_a_longer_limit(
    tmp_path, monkeypatch
):
    """Finish requeues lost claims with a longer time limit, up to the most."""
    section, queue = cloudy_run(tmp_path, monkeypatch, {}, max_retries=0)
    section.setup(section.config.incident["grid"])
    queue.claim("7_1.0", n=2)
    queue.claim("7_2.0", n=1)

    assert section.finish(lambda owner: False) == "work"
    assert queue.counts()["todo"] == 4 * 48
    state = section._load_state()
    assert (state["round"], state["walltime"]) == (2, "0-18:00:00")
    assert state["rounds"][0]["lost"] == 3
    lost = [
        entry["attempts"]
        for entry in section._load_models().values()
        if entry["attempts"]
    ]
    assert len(lost) == 3
    assert all(attempts[0]["result"] == "lost" for attempts in lost)

    # Up to max_walltime, where the section stops (rather than failing
    # them) until max_walltime is raised
    queue.claim("8_1.0")
    section.finish(lambda owner: False)
    assert section._load_state()["walltime"] == "1-03:00:00"
    section._save_state(walltime="3-00:00:00")
    queue.claim("9_1.0")
    assert section.finish(lambda owner: False) == "walltime"
    assert queue.counts()["failed"] == 0
    assert queue.counts()["todo"] == 4 * 48
    state = section._load_state()
    assert (state["next"], state["walltime"]) == ("walltime", "3-00:00:00")


def test_finish_moves_on_when_the_queue_is_empty(tmp_path, monkeypatch):
    """Finish moves on to assembly once every model is done."""
    section, queue = cloudy_run(tmp_path, monkeypatch, {})
    section.setup(section.config.incident["grid"])
    for model in queue.claim("1.0", n=4 * 48):
        queue.done(model, "1.0")
    assert section.finish(lambda owner: False) == "assemble"


def test_finish_refuses_while_workers_run(tmp_path, monkeypatch):
    """Finish refuses while a running worker still holds a claim."""
    section, queue = cloudy_run(tmp_path, monkeypatch, {})
    section.setup(section.config.incident["grid"])
    queue.claim("1.0")
    with pytest.raises(RuntimeError, match="still claimed"):
        section.finish(lambda owner: owner == "1.0")


def check_watch(work_dir):
    """Check syncretize watch shows a finished local run, in every view.

    Args:
        work_dir (str)
            The finished run's work directory.
    """
    import asyncio

    from rich.console import Console

    from syncretize.runner.watch import (
        WatchApp,
        Watcher,
        _Details,
        _squeue_rows,
        summary,
    )

    # --once prints a summary
    view = Watcher(work_dir).read()
    console = Console(record=True, width=120)
    console.print(summary(view))
    text = console.export_text()
    for expected in (
        "189/192 grid points",
        "Failed models (3)",
        "1_3",
        "grid_limited",
    ):
        assert expected in text
    # The Slurm queue, without squeue
    (snapshot,) = view["sections"]
    assert _squeue_rows(snapshot) == ([], "Slurm queue (no squeue here)")
    jobs = [["1", "serial", f"R1-{view['run']['name']}-C1-Cloudy"]]
    jobs[0] += ["RUNNING", "1:00", "20:00", "1", "m1"]
    rows, title = _squeue_rows({**snapshot, "squeue": jobs})
    assert len(rows) == 1 and "1 running" in title

    async def drive():
        """Press through the watch app's views and panels of the run."""
        app = WatchApp(work_dir, interval=0.5)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(1.5)
            section = app._current()
            assert section.panel("failed").row_count == 3
            assert section.panel("rounds").row_count == 1
            assert len(section.panel("log").lines) > 10
            # The Slurm queue opens, Enter shows a job, and it closes
            app.watcher._all_jobs = lambda: jobs
            await pilot.press("s")
            await pilot.pause(0.5)
            assert app.watcher.show_squeue
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert isinstance(app.screen, _Details)
            await pilot.press("escape")
            await pilot.pause(0.2)
            await pilot.press("s")
            await pilot.pause(0.2)
            assert not app.watcher.show_squeue
            await pilot.press("r")
            await pilot.pause(0.2)
            assert app.screen.maximized is section.panel("running")
            await pilot.press("r")
            await pilot.press("l")
            await pilot.pause(0.2)
            assert app.screen.maximized is section.panel("log")
            await pilot.press("l")
            # Enter on a failed model shows its details over the page, and
            # on the log shows it full screen; rounds aren't focused
            assert not section.panel("rounds").can_focus
            section.panel("failed").focus()
            await pilot.pause(0.1)
            enter = app.screen.active_bindings["enter"].binding
            assert enter.description == "Model details"
            await pilot.press("enter")
            await pilot.pause(0.5)
            assert isinstance(app.screen, _Details)
            console = Console(record=True, width=120)
            console.print(app.screen.build())
            details = console.export_text()
            assert "Its attempts (" in details and "failed" in details
            await pilot.press("escape")
            await pilot.pause(0.2)
            assert not isinstance(app.screen, _Details)
            section.panel("log").focus()
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert app.screen.maximized is section.panel("log")
            await pilot.press("enter", "q")

    asyncio.run(drive())


def test_points_below_the_ionising_threshold_arent_run(tmp_path, monkeypatch):
    """Points below the ionising threshold aren't run; light passes through."""
    # A threshold between the fake grid's lowest and highest log10 Q(HI)
    (tmp_path / "probe").mkdir()
    grid = fake_incident_grid(tmp_path / "probe")
    with h5py.File(grid) as hdf:
        log10_q = hdf["log10_specific_ionising_luminosity/HI"][()]
    threshold = float(np.mean([log10_q.min(), log10_q.max()]))
    below = log10_q < threshold
    assert 0 < below.sum() < below.size

    section, queue = cloudy_run(
        tmp_path,
        monkeypatch,
        {},
        extra={"ionising_luminosity_threshold": threshold},
    )
    (grid,) = section.runner.run(local=True)

    # Only the points above it were run
    assert sum(queue.counts().values()) == (~below).sum() * 48
    with h5py.File(grid) as hdf:
        mask = hdf["below_threshold"][()]
        assert mask.sum() == below.sum() * 48
        assert hdf["failures"][()].sum() == 0
        quiet = mask.astype(bool)
        # their light passes through, with no nebular emission or lines
        np.testing.assert_array_equal(
            hdf["spectra/transmitted"][()][quiet],
            hdf["spectra/incident"][()][quiet],
        )
        assert hdf["spectra/incident"][()][quiet].max() > 0
        assert not hdf["spectra/nebular"][()][quiet].any()
        assert not hdf["lines/luminosity"][()][quiet].any()
        assert hdf["lines/luminosity"][()][~quiet].all()
    assert (
        "below the ionising luminosity threshold"
        in (tmp_path / "work" / "run.log").read_text()
    )


def test_failed_models_are_interpolated_from_perturbed_runs(
    tmp_path, monkeypatch
):
    """Failed models are interpolated from perturbed runs where they can be."""
    section, queue = cloudy_run(
        tmp_path,
        monkeypatch,
        {
            # 1_3 fails, but runs either side of it don't
            "1_3": "fail",
            # 2_5 fails wherever it's run
            "2_5*": "fail",
            # 3_1 fails, and so do the runs below it on every axis
            "3_1": "fail",
            "3_1p?-*": "fail",
        },
        perturb={"step": 0.1, "max_steps": 2},
    )
    (grid,) = section.runner.run(local=True)

    cloudy_dir = Path(section._load_state()["cloudy_dir"])
    plan = json.loads((cloudy_dir / "perturb.json").read_text())
    assert sorted(plan) == ["1_3", "2_5", "3_1"]
    # Along every Cloudy axis of the grid (C/O, N/O, depletion, U, nH)
    assert len(plan["1_3"]) == 5
    side = plan["1_3"]["ionisation_parameter"]
    assert side["log"] and side["minus"]["x"] < side["x"] < side["plus"]["x"]
    # A side that failed was tried again further out (up to max_steps)
    assert plan["2_5"]["hydrogen_density"]["minus"]["level"] == 2
    assert (cloudy_dir / "1" / f"{side['minus']['name']}.in").is_file()

    with h5py.File(grid) as hdf:
        interpolated = hdf["interpolated"][()].reshape(4, 48)
        failures = hdf["failures"][()].reshape(4, 48)
        luminosity = hdf["lines/luminosity"][()].reshape(4, 48, -1)
    # 1_3 is interpolated; 2_5 (no successful run) and 3_1 (one sided)
    # stay failed
    assert interpolated.sum() == 1 and list(interpolated.sum(axis=1)) == [
        0,
        1,
        0,
        0,
    ]
    assert failures.sum() == 2
    assert luminosity[1][interpolated[1] == 1].all()

    run_log = (tmp_path / "work" / "run.log").read_text()
    assert "Perturbing 3 failed models" in run_log
    assert "Retrying 15 perturbed runs that failed, twice as far" in run_log
    assert "1 interpolated from perturbed runs, 2 failed" in run_log
    assert sorted(section._load_state()["failures"]) == ["2_5", "3_1"]
    # The perturbed runs' rounds are the PerturbCloudy phase
    state = section._load_state()
    assert [r.get("perturb", False) for r in state["rounds"]] == [
        False,
        True,
        True,
    ]
    assert section._job_name("work") == "R3-test-C1-PerturbCloudy"
    assert "[C1 perturb]" in run_log


def test_cloudy_caveats_are_flagged_and_crashes_remedied(
    tmp_path, monkeypatch
):
    """Runs Cloudy finished with warnings or an unintended stop are done
    (flagged), and a crash with a remedy is run again with it."""
    section, queue = cloudy_run(
        tmp_path,
        monkeypatch,
        {"1_3": "warn", "2_5": "nzone", "0_0": "qheat", "3_1": "fail"},
    )
    (grid,) = section.runner.run(local=True)

    models = section._load_models()
    assert [models[m]["attempts"][-1]["outcome"] for m in ("1_3", "2_5")] == [
        "warnings",
        "unintended",
    ]
    assert models["2_5"]["attempts"][-1]["reason"] == (
        "default number of zones reached"
    )
    # The crash was run again with grain quantum heating off, and filled it
    assert models["0_0"]["attempts"][-1]["reason"] == (
        "assert wid == 0 (grains_qheat.cpp:2047)"
    )
    assert models["0_0"]["state"] == "done"
    assert models["0_0"]["remedied_by"] == "0_0r1"
    cloudy_dir = Path(section._load_state()["cloudy_dir"])
    assert "no qheat" in (cloudy_dir / "0" / "0r1.in").read_text()
    remedies = json.loads((cloudy_dir / "remedies.json").read_text())
    assert remedies["0_0"][0]["changes"] == {"grain_qheat": False}
    # A crash no remedy matches stays failed
    assert sorted(section._load_state()["failures"]) == ["3_1"]

    with h5py.File(grid) as hdf:
        quality = hdf["quality"][()].reshape(4, 48)
        luminosity = hdf["lines/luminosity"][()].reshape(4, 48, -1)
        remedies_used = json.loads(hdf["quality"].attrs["remedies"])
    # One point of each (the photoionisation points aren't in j's order)
    assert sorted(quality[quality > 0]) == [1, 2, 4, 16]
    assert quality[0].max() == 4 and quality[3].max() == 16
    assert np.isfinite(luminosity[quality < 16]).all()
    assert np.isnan(luminosity[quality == 16]).all()
    assert remedies_used == [{"grain_qheat": False}]
    run_log = (tmp_path / "work" / "run.log").read_text()
    assert "Running 1 crashed grid point again with a remedy" in run_log
    assert "Filled 1 crashed grid point from runs with a remedy" in run_log


def test_review_pauses_when_many_fail(tmp_path, monkeypatch):
    """A section pauses for a review when too many models failed with no
    remedy left, then carries on (once) when resumed."""
    section, queue = cloudy_run(
        tmp_path,
        monkeypatch,
        {"0_*": "fail", "1_*": "fail", "2_0": "qheat"},
        max_retries=0,
        options={"review": {"min_models": 4}},
    )
    section.setup(section.config.incident["grid"])
    run_worker(section, 60, 60, max_retries=0)
    assert section.finish(lambda owner: False) == "review"
    state = section._load_state()
    assert state["next"] == "review"
    # The crash with a remedy waits for it, rather than counting as failed
    assert queue.states()["2_0r1"] == "todo"
    why, where = section._review()
    assert why.startswith("96 of 191 grid points run have failed")
    assert "ages = index 0: 96 of 96 failed (100%)" in where

    # Resuming carries on with the failures, and won't pause again
    monkeypatch.setattr(section, "submit_round", lambda: "9")
    assert section.resume() == "9"
    state = section._load_state()
    assert state["reviewed"] and state["next"] == "finish"
    assert state["round"] == 2
    assert section._review() is None


def test_recheck_finishes_runs_cloudy_finished(tmp_path, monkeypatch):
    """Recheck makes failed runs Cloudy finished done, drops the queued
    runs they no longer need, and carries the section on."""
    section, queue = cloudy_run(
        tmp_path,
        monkeypatch,
        {"1_3": "fail", "2_5": "warn"},
        perturb={"step": 0.1, "max_steps": 1},
        max_retries=0,
    )
    section.setup(section.config.incident["grid"])
    run_worker(section, 60, 60, max_retries=0)
    # As the older rule did: a run with a warning failed
    os.rename(queue._path("done", "2_5"), queue._path("failed", "2_5"))
    Path(queue._path("failed", "2_5")).write_text("1")
    section.finish(lambda owner: False)
    assert queue.states()["2_5"] == "failed"
    # Its perturbed runs are queued
    assert any(m.startswith("2_5p") for m in queue.names("todo"))

    section._save_state(next="done")
    section.runner._save_state(next="done")
    assert section.runner.recheck()[0] == "work"
    states = queue.states()
    assert states["2_5"] == "done"
    assert not any(m.startswith("2_5p") for m in states)
    assert "2_5" not in perturb.load(section._load_state()["cloudy_dir"])
    assert section._load_models()["2_5"]["recheck"]["outcome"] == "warnings"
    assert section.runner._load_state()["next"] == "cloudy"
    run_log = (tmp_path / "work" / "run.log").read_text()
    assert "1 failed run finished with warnings" in run_log
    assert "dropped 10 queued runs they no longer need" in run_log

    # Nothing to change: a done section stays done
    section._save_state(next="done")
    assert section.recheck() is None


def test_recheck_remedies_crashes_older_rules_misread(tmp_path, monkeypatch):
    """A crash recorded with an older reason gets its remedy on recheck."""
    section, queue = cloudy_run(
        tmp_path, monkeypatch, {"1_3": "qheat"}, max_retries=0
    )
    section.setup(section.config.incident["grid"])
    section._save_state(next="finish")
    run_worker(section, 60, 60, max_retries=0)
    section._gather_records()
    section._remove_records()
    models = section._load_models()
    models["1_3"]["attempts"][-1]["reason"] = "[0x4a0715] _start -- ??:?"
    section._save_models(models)
    assert section.recheck() == "work"
    assert queue.states()["1_3r1"] == "todo"


def test_remedy_config_is_checked(tmp_path):
    """Remedies default to the qheat one, and bad ones are refused."""
    executable = tmp_path / "c23.01"
    run = load_run(
        write_run(
            tmp_path,
            Cloudy={"config": "c23.01-sps", "executable": str(executable)},
        )
    )
    assert run.sections[1]["remedies"][0]["changes"] == {"grain_qheat": False}
    assert run.sections[1]["review"]["fraction"] == 0.1
    for remedies, message in (
        ([{"match": "("}], "mapping of match"),
        ([{"match": "(", "changes": {"nend": 3000}}], "remedies\\[0\\].match"),
    ):
        with pytest.raises(ValueError, match=message):
            load_run(
                write_run(
                    tmp_path,
                    Cloudy={
                        "config": "c23.01-sps",
                        "executable": str(executable),
                        "remedies": remedies,
                    },
                )
            )
    run = load_run(
        write_run(
            tmp_path,
            Cloudy={
                "config": "c23.01-sps",
                "executable": str(executable),
                "review": False,
                "remedies": [],
            },
        )
    )
    assert run.sections[1]["review"] is None
    assert run.sections[1]["remedies"] == []


def test_several_cloudy_sections_share_the_incident_grid(
    tmp_path, monkeypatch
):
    """Several Cloudy sections share the incident grid, each making a grid."""
    density = tmp_path / "density.yaml"
    density.write_text(yaml.safe_dump({"hydrogen_density": 100.0}))
    first, _ = cloudy_run(
        tmp_path,
        monkeypatch,
        {"1_3": "fail"},
        more={
            "Cloudy2": {
                "config": "c23.01-sps-grid_limited",
                "extra_config": str(density),
            }
        },
    )
    runner = first.runner
    assert [(s.label, s.name) for s in runner.sections] == [
        ("C1", "c23.01-sps-grid_limited"),
        ("C2", "c23.01-sps-grid_limited-density"),
    ]
    grids = runner.run(local=True)

    # One grid per section, each from its own Cloudy runs, the run done once
    # both are
    assert [os.path.basename(grid) for grid in grids] == [
        "fake-sps_cloudy-c23.01-sps-grid_limited.hdf5",
        "fake-sps_cloudy-c23.01-sps-grid_limited-density.hdf5",
    ]
    state = runner._load_state()
    assert state["next"] == "done" and state["grids"] == grids
    first, second = runner.sections
    assert first._load_state()["failures"] == second._load_state()["failures"]
    assert second._job_name("work") == "R1-test-C2-Cloudy"

    # run.log says which section each line is about
    run_log = (tmp_path / "work" / "run.log").read_text()
    for text in ("[C1 assemble]", "[C2 assemble]", "Cloudy section C2"):
        assert text in run_log

    # watch shows one section at a time, each with its own log, and every
    # section on the all grids page
    import asyncio

    from syncretize.runner.watch import WatchApp, Watcher

    view = Watcher(str(tmp_path / "work")).read()
    assert [s["label"] for s in view["sections"]] == ["C1", "C2"]
    assert not any("[C1 " in line for line in view["sections"][1]["log"])

    async def drive():
        """Switch between the sections and pick one on the all grids page."""
        app = WatchApp(str(tmp_path / "work"), interval=0.5)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(1.5)
            switcher = app.query_one("#sections")
            assert app._current() is switcher.visible_content
            first = switcher.current
            await pilot.press("2")
            assert switcher.current != first
            await pilot.press("1")
            assert switcher.current == first
            # The all grids page lists both, and Enter shows the one picked
            await pilot.press("g")
            await pilot.pause(0.5)
            rows = app.grid_rows()
            assert [cells[7].split()[0] for _, cells, _ in rows] == [
                "C1",
                "C2",
            ]
            # o sorts by the next column, O reverses, and the place order
            # comes back after the last
            page = app.screen
            await pilot.press("o", "O")
            assert (page.sort, page.reverse) == (1, True)
            for _ in range(7):
                await pilot.press("o")
            assert (page.sort, page.reverse) == (0, False)
            await pilot.press("down", "enter")
            await pilot.pause(0.5)
            assert switcher.current == rows[1][0]
            await pilot.press("q")

    asyncio.run(drive())

    # Watching a directory holding runs: a tab per section of each run under
    # way, the shown run read in full
    from syncretize.runner.watch import find_runs

    work_dir = str(tmp_path / "work")
    runs = tmp_path / "runs"
    for name, run_work_dir in (("a", work_dir), ("b", "work")):
        (runs / name).mkdir(parents=True)
        (runs / name / "run.yaml").write_text(
            yaml.safe_dump({"Run": {"work_dir": run_work_dir}})
        )
    other = runs / "b"
    assert find_runs(str(runs)) == {
        "active": [],
        "done": [work_dir],
        "waiting": [str(other / "work")],
    }
    state = yaml.safe_load((tmp_path / "work" / "state.yaml").read_text())
    (tmp_path / "work" / "state.yaml").write_text(
        yaml.safe_dump({**state, "next": "cloudy"})
    )
    (other / "work").mkdir()
    (other / "work" / "state.yaml").write_text(
        yaml.safe_dump({"name": "other", "next": "start"})
    )

    async def drive_directory():
        """Check a directory of runs shows a tab per section, and the pick."""
        app = WatchApp(interval=0.5, directory=str(runs))
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause(1.5)
            grids = [
                (cells[6], cells[7].split()[:1])
                for _, cells, _ in app.grid_rows()
            ]
            assert grids == [("test", ["C1"]), ("test", ["C2"]), ("other", [])]
            assert app._shown() == work_dir
            await pilot.press("3")
            await pilot.pause(0.5)
            assert app._shown() == str(other / "work")
            await pilot.press("q")

    asyncio.run(drive_directory())


def test_watch_reads_state_again_only_once_changed(tmp_path):
    """A state file is parsed again only once it has been replaced."""
    from syncretize.runner.watch import _read_yaml

    path = tmp_path / "state.yaml"
    path.write_text("next: finish\n")
    first = _read_yaml(str(path))
    assert first == {"next": "finish"} and _read_yaml(str(path)) is first
    # Saved as the Runner does, by replacing the file
    (tmp_path / "state.yaml.tmp").write_text("next: assemble\n")
    os.replace(tmp_path / "state.yaml.tmp", path)
    assert _read_yaml(str(path)) == {"next": "assemble"}
    assert _read_yaml(str(tmp_path / "missing.yaml")) == {}


def test_watch_keeps_only_ended_jobs_from_sacct(monkeypatch):
    """An array with a task still going isn't recorded as ended."""
    from syncretize.runner import watch

    out = "1_0|COMPLETED\n1_1|SUSPENDED\n2|CANCELLED by 5\n2_1|TIMEOUT\n"
    monkeypatch.setattr(
        watch.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, out, ""),
    )
    assert watch._sacct(["1", "2"]) == {"2": {"CANCELLED": 1, "TIMEOUT": 1}}


def test_watch_failures_leave_out_perturbed_runs():
    """Perturbed runs given up on aren't listed as failed models.

    They're interpolated past, so aren't failures of the grid.
    """
    from syncretize.runner.watch import _failures

    records = [
        {"model": "3_1", "attempt": 3, "requeued": False, "reason": "x"},
        {"model": "3_1p0+1", "attempt": 3, "requeued": False, "reason": "y"},
        {"model": "4_2", "attempt": 1, "requeued": True, "reason": "z"},
    ]
    snapshot = {"state": {"failures": {"1_1": "w"}}, "records": records}
    assert _failures(snapshot) == {"1_1": "w", "3_1": "x"}


def test_worker_puts_back_a_claim_it_has_no_time_for(tmp_path):
    """Models claimed together don't start after the claiming deadline."""
    exe = tmp_path / "cloudy" / "source" / "cloudy.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\nsleep 3\n")
    exe.chmod(0o755)
    (tmp_path / "inputs" / "0").mkdir(parents=True)
    queue = Queue(str(tmp_path / "queue"))
    queue.add(["0_0", "0_1", "0_2"])
    now = time.monotonic()
    records = work(
        queue.root,
        str(tmp_path / "inputs"),
        str(exe.parents[1]),
        "1_0",
        slots=1,
        claim=3,
        extensions=["cont"],
        max_retries=0,
        record_file=str(tmp_path / "state" / "1_0.jsonl"),
        deadline=now + 0.5,
        stop_at=now + 1.0,
    )
    # The first was stopped at the time limit; the other two never ran
    assert [r["result"] for r in records] == ["stopped"]
    assert queue.counts() == {"todo": 3, "claimed": 0, "done": 0, "failed": 0}


def test_worker_records_a_model_it_cannot_run(tmp_path):
    """A model whose run raises fails rather than staying claimed.

    The failure counts as an attempt.
    """
    queue = Queue(str(tmp_path / "queue"))
    queue.add(["0_0"])
    records = work(
        queue.root,
        str(tmp_path / "inputs"),  # no 0/ to run it in
        str(tmp_path / "cloudy"),
        "1_0",
        slots=1,
        claim=1,
        extensions=[],
        max_retries=0,
        record_file=str(tmp_path / "state" / "1_0.jsonl"),
    )
    assert [r["result"] for r in records] == ["failed"]
    assert "couldn't run it" in records[0]["reason"]
    assert queue.counts()["failed"] == 1 and queue.counts()["claimed"] == 0


def test_queue_failure_counts_grow(tmp_path):
    """Each failure rewrites the claim's count, past one digit."""
    queue = Queue(str(tmp_path / "queue"))
    queue.add(["0_0"])
    for attempt in range(1, 12):
        assert queue.claim("w.0") == ["0_0"]
        assert queue.fail("0_0", "w.0", max_retries=20) == attempt


def test_finish_keeps_lost_claims_and_counts_given_up(tmp_path, monkeypatch):
    """Finish keeps count of lost claims and models given up on.

    Lost claims found by a finish that stops early are still counted, and
    requeuing old failures doesn't make given-up negative.
    """
    section, queue = cloudy_run(tmp_path, monkeypatch, {}, max_retries=0)
    section.setup(section.config.incident["grid"])
    queue.claim("7_1.0", n=2)
    queue.claim("7_2.0")
    with pytest.raises(RuntimeError, match="still claimed"):
        section.finish(lambda owner: owner == "7_2.0")
    section.finish(lambda owner: False)
    (summary,) = section._load_state()["rounds"]
    assert summary["lost"] == 3 and summary["given_up"] == 0

    # A model given up on, as a worker records it
    (model,) = queue.claim("8_1.0")
    attempt = queue.fail(model, "8_1.0")
    record = {"model": model, "result": "failed", "reason": "x", "seconds": 1}
    Path(section._path("state", "8_1.jsonl")).write_text(
        json.dumps({**record, "attempt": attempt, "requeued": False}) + "\n"
    )
    section.finish(lambda owner: False)
    assert section._load_state()["rounds"][-1]["given_up"] == 1
    # A larger max_retries puts it back, which gives up on nothing
    section.config.run["cloudy"]["max_retries"] = 2
    section.finish(lambda owner: False)
    assert section._load_state()["rounds"][-1]["given_up"] == 0
    assert not os.listdir(section._path("state"))


def test_walltime_is_at_most_max_walltime(tmp_path, monkeypatch):
    """A round's time limit is at most max_walltime."""
    section, _ = cloudy_run(tmp_path, monkeypatch, {})
    section._save_state(walltime="3-00:00:00")
    section.config.run["cloudy"]["max_walltime"] = "1-00:00:00"
    assert section._walltime() == 86400


def test_resume_submits_a_round_its_job_did_not(tmp_path, monkeypatch):
    """Resume carries on a round or an assembly its job didn't finish.

    A finish that died between starting a round and submitting it, and an
    assembly that died before marking the run done, are carried on.
    """
    section, _ = cloudy_run(tmp_path, monkeypatch, {})
    runner = section.runner
    runner._save_state(next="cloudy")
    section._save_state(
        next="finish", round=2, rounds=[{"round": 1, "array_job": "5"}]
    )
    submitted = []
    monkeypatch.setattr(
        section, "submit_round", lambda: submitted.append(1) or "9"
    )
    assert runner.resume() == ["9"] and submitted == [1]

    section._save_state(next="done", grid="g.hdf5")
    assert runner.resume() == [None]
    assert runner._load_state()["next"] == "done"


def test_perturb_queues_planned_sides_never_queued(tmp_path):
    """Sides a finish job planned but died before queuing are queued."""
    from syncretize.runner import perturb

    side = {"level": 1, "x": 0.0}
    plan = {
        "0_1": {
            "ionisation_parameter": {
                "log": True,
                "x": 0.0,
                "minus": {**side, "name": "1p0-1"},
                "plus": {**side, "name": "1p0+1"},
            }
        }
    }
    (tmp_path / perturb.PLAN).write_text(json.dumps(plan))
    settings = {"step": 0.1, "log_step": 0.05, "max_steps": 3}
    runs = perturb.advance(str(tmp_path), [], {}, settings, None)
    assert runs == ["0_1p0-1", "0_1p0+1"]
    # Once queued, they aren't again
    states = dict.fromkeys(runs, "todo")
    assert perturb.advance(str(tmp_path), [], states, settings, None) == []


def test_perturb_leaves_models_surrounded_by_failures(tmp_path):
    """A model whose every neighbour failed isn't perturbed."""
    from syncretize.runner import perturb

    # A 3 x 2 grid of (nH, U): j = 2 * nH index + U index
    values = [[nh, u] for nh in (1, 10, 100) for u in (0.01, 0.1)]
    (tmp_path / "grid_points.json").write_text(
        json.dumps(
            {
                "photoionisation": {
                    "axes": ["hydrogen_density", "ionisation_parameter"],
                    "values": values,
                }
            }
        )
    )
    written = []
    settings = {"step": 0.1, "log_step": 0.05, "max_steps": 3}
    # 0_5's neighbours (0_3 and 0_4) both failed; 0_0 has 0_2 done
    states = dict.fromkeys(["0_0", "0_1", "0_3", "0_4", "0_5"], "failed")
    states["0_2"] = "done"
    runs = perturb.advance(
        str(tmp_path),
        ["0_0", "0_5"],
        states,
        settings,
        lambda *args: written.append(args[2]),
    )
    assert {run.partition("p")[0] for run in runs} == {"0_0"}
    assert len(runs) == 4  # either side along both axes
    plan = perturb.load(str(tmp_path))
    assert plan["0_5"] == {} and set(plan["0_0"]) == {
        "hydrogen_density",
        "ionisation_parameter",
    }

    # Still surrounded while its neighbours' runs are under way
    states.update(dict.fromkeys(runs, "todo"))
    assert (
        perturb.advance(str(tmp_path), ["0_0", "0_5"], states, settings, None)
        == []
    )

    # Once 0_3 can be interpolated, 0_5 is perturbed after all
    states.update(dict.fromkeys(runs, "failed"))
    states["0_3"] = "failed"
    plan["0_3"] = {
        "ionisation_parameter": {
            "log": True,
            "x": -1.0,
            "minus": {"level": 1, "name": "3p1-1", "x": -1.1},
            "plus": {"level": 1, "name": "3p1+1", "x": -0.9},
        }
    }
    (tmp_path / perturb.PLAN).write_text(json.dumps(plan))
    states.update({"0_3p1-1": "done", "0_3p1+1": "done"})
    settings["max_steps"] = 1  # 0_0's failed sides aren't tried further out
    runs = perturb.advance(
        str(tmp_path),
        ["0_0", "0_5"],
        states,
        settings,
        lambda *args: written.append(args[2]),
    )
    assert {run.partition("p")[0] for run in runs} == {"0_5"}
    assert perturb.load(str(tmp_path))["0_5"]


def test_queue_claims_from_a_kept_listing(tmp_path, monkeypatch):
    """A claimer lists todo/ again only once its listing is used up.

    It lists again once its listing runs out or goes stale, and still finds
    models put back since.
    """
    queue = Queue(str(tmp_path / "queue"))
    queue.add([f"0_{j}" for j in range(50)])
    listings = []
    names = queue.names
    monkeypatch.setattr(
        queue, "names", lambda state: listings.append(state) or names(state)
    )
    claimed = [queue.claim("w.0")[0] for _ in range(10)]
    assert listings == ["todo"] and len(set(claimed)) == 10

    # Another worker takes the rest, so the kept listing has gone stale
    other = Queue(queue.root)
    taken = []
    while models := other.claim("v.0"):
        taken += models
    assert len(taken) == 40
    assert queue.claim("w.0") == []
    # A model put back since is found by listing again
    other.release(taken[0], "v.0")
    assert queue.claim("w.0") == [taken[0]]
