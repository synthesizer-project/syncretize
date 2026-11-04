from pathlib import Path

import pytest

from syncretize.cloudy.utils import get_cloudy_params, resolve_param_file
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
