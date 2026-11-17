from pathlib import Path

from syncretize.cloudy.utils import get_cloudy_params, resolve_param_file
from syncretize.parser import Parser


def test_parser_exposes_alpha_flags(tmp_path):
    parser = Parser("alpha test", with_alpha=True)

    args = parser.parse_args(
        [
            "--grid-dir",
            str(tmp_path),
            "--input-dir",
            str(tmp_path),
            "--full",
        ]
    )

    assert args.grid_dir == str(tmp_path)
    assert args.input_dir == str(tmp_path)
    assert args.full
    assert not args.individual


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


def test_resolve_param_file_takes_paths_and_names(tmp_path):
    # A path is used as given, with or without the extension
    param_file = tmp_path / "c23.01-custom.yaml"
    param_file.write_text("cloudy_version: c23.01\n")
    assert resolve_param_file(param_file) == param_file
    assert resolve_param_file(tmp_path / "c23.01-custom") == param_file
    other = tmp_path / "custom.yml"
    other.write_text("cloudy_version: c23.01\n")
    assert resolve_param_file(other) == other

    # A bare name falls back to the ready-made parameter files
    assert resolve_param_file("c23.01-sps").exists()

    # With a param_dir, a file there is used, and paths and ready-made
    # names still resolve when it isn't there
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    copied = run_dir / "c23.01-sps.yaml"
    copied.write_text("cloudy_version: c23.01\n")
    assert resolve_param_file("c23.01-sps", param_dir=run_dir) == copied
    assert resolve_param_file(param_file, param_dir=run_dir) == param_file
    assert resolve_param_file("c23.01-nlr", param_dir=run_dir).exists()
    assert resolve_param_file("missing", param_dir=run_dir) == (
        run_dir / "missing.yaml"
    )
