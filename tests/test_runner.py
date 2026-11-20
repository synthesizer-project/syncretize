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
from syncretize.runner import Runner
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
    """Write a run config with a blackbody incident stage."""
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
    config = load_run(TEMPLATE)
    assert config.run["cloudy"]["walltime_growth"] == 1.5
    assert config.cloudy is None
    assert config.incident["config"].endswith(
        "blackbody/generate_incident_grid_template.yaml"
    )


def test_defaults_and_relative_paths(tmp_path):
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
        ({"Extra": {}}, "unknown"),
    ],
)
def test_bad_configs(tmp_path, sections, message):
    with pytest.raises(ValueError, match=message):
        load_run(write_run(tmp_path, **sections))


def test_seconds():
    assert seconds("00:15:00") == 900
    assert seconds("3-00:00:00") == 3 * 86400
    assert seconds(43200) == 43200  # YAML's reading of 12:00:00
    assert slurm_time(seconds("1-02:03:04")) == "1-02:03:04"


def test_local_incident_run(tmp_path):
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
    """Write a 2 x 2 (age, metallicity) incident grid."""
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
    runner.cloudy_setup(grid)

    # 2 x 2 incident models, each with 2 x 2 x 3 x 2 x 2 photoionisation
    # models
    work = tmp_path / "work"
    assert Queue(work / "queue").counts()["todo"] == 4 * 48
    cloudy_dir = work / "cloudy" / "fake-sps_cloudy-c23.01-sps-grid_limited"
    assert (cloudy_dir / "3" / "47.in").is_file()
    assert (cloudy_dir / "0" / "linelist.dat").is_file()
    state = yaml.safe_load((work / "state.yaml").read_text())
    assert state == {"incident_grid": grid, "cloudy_dir": str(cloudy_dir)}

    # Setting up again keeps each model's state
    queue = Queue(work / "queue")
    (model,) = queue.claim("1.0")
    queue.done(model, "1.0")
    runner.cloudy_setup(grid)
    assert queue.counts() == {
        "todo": 4 * 48 - 1,
        "claimed": 0,
        "done": 1,
        "failed": 0,
    }


def test_unknown_cloudy_config(tmp_path):
    with pytest.raises(ValueError, match="shipped Cloudy config"):
        load_run(
            write_run(
                tmp_path, Cloudy={"config": "nope", "executable": "/cloudy"}
            )
        )


FAKE_CLOUDY = f"""#!{sys.executable}
# A stand-in for cloudy.exe -r <j>, run in <cloudy_dir>/<i>. FAKE_CLOUDY maps
# a model "<i>_<j>" (or "*", every model) to what it does: "fail" (exit with
# an error), "fail_once" (fail the first run only), "no_lines" (exit OK
# without lines) or "sleep" (take 0.5 s).
import json, os, sys, time
j = sys.argv[2]
model = os.path.basename(os.getcwd()) + "_" + j
behaviours = json.loads(os.environ.get("FAKE_CLOUDY", "{{}}"))
behaviour = behaviours.get(model, behaviours.get("*"))
if behaviour == "sleep":
    time.sleep(0.5)
if behaviour == "fail_once":
    behaviour = "ok" if os.path.exists(j + ".out") else "fail"
with open(j + ".out", "w") as file:
    file.write(os.environ["CLOUDY_DATA_PATH"] + "\\n")
    if behaviour == "fail":
        file.write(" [Stop in cdMain, something went wrong]\\n")
        sys.exit(1)
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


def cloudy_run(tmp_path, monkeypatch, behaviours, **cloudy):
    """A Runner set up to run the fake Cloudy on the fake incident grid."""
    exe = tmp_path / "cloudy" / "source" / "cloudy.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text(FAKE_CLOUDY)
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_CLOUDY", json.dumps(behaviours))
    grid = fake_incident_grid(tmp_path)
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
                "executable": str(exe.parents[1]),
            },
        )
    )
    return runner, Queue(tmp_path / "work" / "queue")


def test_local_workers_run_every_model(tmp_path, monkeypatch):
    runner, queue = cloudy_run(
        tmp_path,
        monkeypatch,
        {"1_3": "fail", "2_5": "no_lines", "0_9": "fail_once"},
    )
    grid = runner.run(local=True)

    # 0_9 worked when retried; the others failed all 1 + 2 tries

    assert queue.counts() == {
        "todo": 0,
        "claimed": 0,
        "done": 4 * 48 - 2,
        "failed": 2,
    }
    failed = tmp_path / "work" / "queue" / "failed"
    assert sorted(os.listdir(failed)) == ["1_3", "2_5"]
    assert (failed / "1_3").read_text() == "3"
    assert runner._load_state()["round"] == 3
    cloudy_dir = runner._load_state()["cloudy_dir"]
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
    assert loaded.spectra["nebular"].max() > 0
    with h5py.File(grid, "r") as hdf:
        failures = hdf["failures"][()]
        assert failures.sum() == 2
        # One each at incident points 1 and 2 (the photoionisation index
        # isn't in C order, so isn't checked)
        assert list(failures.reshape(4, 48).sum(axis=1)) == [0, 1, 1, 0]
        assert hdf["RunConfig/Cloudy"].attrs["include_spectra"]
        assert "fake-sps" in hdf["RunConfig/contents"][()].decode()


def test_workers_stop_at_their_deadline(tmp_path, monkeypatch):
    runner, queue = cloudy_run(
        tmp_path,
        monkeypatch,
        {"*": "sleep"},
        cores_per_worker=2,
        margin="00:00:01",
    )
    runner.cloudy_setup(runner.config.incident["grid"])
    start = time.monotonic()
    runner.work("1", walltime="00:00:03")

    # Each loop starts models for 2 s, so runs about 4 of 0.5 s, and the
    # last ends by 2.5 s
    assert time.monotonic() - start < 3
    counts = queue.counts()
    assert 2 <= counts["done"] <= 12
    assert counts["claimed"] == 0 and counts["failed"] == 0


def test_finish_requeues_stopped_workers_models_with_a_longer_limit(
    tmp_path, monkeypatch
):
    runner, queue = cloudy_run(tmp_path, monkeypatch, {}, max_retries=0)
    runner.cloudy_setup(runner.config.incident["grid"])
    queue.claim("7_1.0", n=2)
    queue.claim("7_2.0", n=1)

    assert runner.finish(lambda owner: False) == "work"
    assert queue.counts()["todo"] == 4 * 48
    state = runner._load_state()
    assert (state["round"], state["walltime"]) == (2, "0-18:00:00")

    # Up to max_walltime, and then a stopped model counts as failed
    queue.claim("8_1.0")
    runner.finish(lambda owner: False)
    assert runner._load_state()["walltime"] == "1-03:00:00"
    runner._save_state(walltime="3-00:00:00")
    (model,) = queue.claim("9_1.0")
    runner.finish(lambda owner: False)
    assert os.listdir(tmp_path / "work" / "queue" / "failed") == [model]
    assert runner._load_state()["walltime"] == "3-00:00:00"


def test_finish_moves_on_when_the_queue_is_empty(tmp_path, monkeypatch):
    runner, queue = cloudy_run(tmp_path, monkeypatch, {})
    runner.cloudy_setup(runner.config.incident["grid"])
    for model in queue.claim("1.0", n=4 * 48):
        queue.done(model, "1.0")
    assert runner.finish(lambda owner: False) == "assemble"


def test_finish_refuses_while_workers_run(tmp_path, monkeypatch):
    runner, queue = cloudy_run(tmp_path, monkeypatch, {})
    runner.cloudy_setup(runner.config.incident["grid"])
    queue.claim("1.0")
    with pytest.raises(RuntimeError, match="still claimed"):
        runner.finish(lambda owner: owner == "1.0")
