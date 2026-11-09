from pathlib import Path

import pytest
from synthesizer.abundances import Abundances

from syncretize.cloudy import cloudy25, submission_scripts
from syncretize.cloudy.utils import (
    cloudy_run_status,
    get_cloudy_module,
    get_cloudy_params,
    resolve_linelist,
    resolve_param_file,
)
from syncretize.parser import Parser, get_choice


def test_parser_cloudy_arguments(tmp_path):
    parser = Parser("cloudy test", cloudy_args=True)
    cloudy_dir = tmp_path / "cloudy"

    args = parser.parse_args(
        [
            "--grid-dir",
            str(tmp_path),
            "--cloudy-output-dir",
            str(cloudy_dir),
            "--incident-grid",
            "incident-grid",
        ]
    )

    assert args.cloudy_output_dir == str(cloudy_dir)
    assert args.incident_grid == "incident-grid"
    assert args.cloudy_paramfile == "c23.01-sps"
    assert not hasattr(args, "download")


def test_get_cloudy_params_splits_fixed_and_grid(tmp_path):
    param_dir = tmp_path / "params"
    param_dir.mkdir()
    param_file = param_dir / "custom.yaml"

    param_file.write_text(
        "\n".join(
            [
                "cloudy_version: c23.01",
                "hydrogen_density: [1, 2]",
                "abundance_scalings:",
                "  carbon_to_oxygen: [0.1, 0.2]",
                "  nitrogen_to_oxygen: 0.3",
                "fixed_value: 42",
            ]
        )
    )

    fixed, grid = get_cloudy_params(
        param_file=Path(param_file).name,
        param_dir=str(param_dir),
    )

    assert fixed["cloudy_version"] == "c23.01"
    assert fixed["fixed_value"] == 42
    assert fixed["abundance_scalings.nitrogen_to_oxygen"] == 0.3

    assert grid["hydrogen_density"].tolist() == [1.0, 2.0]
    assert grid["abundance_scalings.carbon_to_oxygen"].tolist() == [0.1, 0.2]


def test_get_choice_checks_options():
    assert (
        get_choice({"imf_type": "kroupa"}, "imf_type", ["kroupa"]) == "kroupa"
    )
    assert (
        get_choice({"rotation": 0.4}, "rotation", {0.0: "00", 0.4: "40"})
        == 0.4
    )

    with pytest.raises(ValueError, match="must be one of"):
        get_choice({"imf_type": "bad"}, "imf_type", ["kroupa", "salpeter"])
    with pytest.raises(ValueError, match="must be one of"):
        get_choice({}, "imf_type", ["kroupa"])


def test_resolve_param_file_takes_paths_and_names(tmp_path):
    # A path is used as given, with or without the extension
    param_file = tmp_path / "c23.01-custom.yaml"
    param_file.write_text("cloudy_version: c23.01\n")
    assert resolve_param_file(param_file) == param_file
    assert resolve_param_file(tmp_path / "c23.01-custom") == param_file

    # A bare name falls back to the ready-made parameter files
    assert resolve_param_file("c23.01-sps").exists()


def test_cloudy_run_status_reads_the_out_file(tmp_path):
    model = tmp_path / "0"
    assert cloudy_run_status(model) == "not run (no .out file)"

    model.with_suffix(".out").write_text("")
    assert cloudy_run_status(model) == "empty .out file"

    model.with_suffix(".out").write_text(
        "lots of output\n Cloudy exited OK]\n"
    )
    assert cloudy_run_status(model) is None

    model.with_suffix(".out").write_text("zone 3\n PROBLEM ABORT boom\n")
    assert cloudy_run_status(model).startswith("aborted")

    model.with_suffix(".out").write_text("zone 3\n zone 4\n")
    assert cloudy_run_status(model) == "did not finish: zone 4"


def test_cloudy_helpers_work_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert get_cloudy_module("c25.00") is cloudy25
    with pytest.raises(ValueError, match="Unknown Cloudy version"):
        get_cloudy_module("c99.00")
    assert resolve_linelist("linelist-standard.dat").exists()


def test_cloudy25_is_its_own_module_and_cmb_false_is_off():
    assert cloudy25.__name__ == "syncretize.cloudy.cloudy25"
    for cmb, expected in ((None, False), (False, False), (True, True)):
        lines = cloudy25.create_cloudy_input(
            "model",
            [],
            Abundances(),
            output_dir=None,
            CMB=cmb,
            ionisation_parameter=0.01,
            ionisation_parameter_model="fixed",
            hydrogen_density=100.0,
            geometry="planeparallel",
        )
        assert any(line.startswith("CMB") for line in lines) is expected


def test_slurm_script_is_written_beside_the_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submission_scripts.artemis(
        "grid_cloudy-test",
        number_of_incident_grid_points=3,
        number_of_photoionisation_models=2,
        cloudy_output_dir=str(tmp_path / "runs"),
        cloudy_executable_path="/cloudy",
    )
    grid_dir = tmp_path / "runs" / "grid_cloudy-test"
    script = (grid_dir / "grid_cloudy-test.slurm").read_text()
    assert (grid_dir / "logs").is_dir()
    assert f"python {submission_scripts.RUN_CLOUDY}" in script
    assert f"--output={grid_dir}/logs/" in script
