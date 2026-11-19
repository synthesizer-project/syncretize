from pathlib import Path

import numpy as np
import pytest

from syncretize.cloudy.utils import get_cloudy_params, resolve_param_file
from syncretize.parser import Parser, get_choice, load_params


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


def _write(tmp_path, text):
    path = tmp_path / "params.yaml"
    path.write_text(text)
    return path


def test_load_params_builds_axes_and_checks_keys(tmp_path):
    path = _write(
        tmp_path,
        "Model:\n  variant: bin\n"
        "Constants:\n  max_age: 1e8\n"
        "Axes:\n"
        "  - {name: a, type: values, values: [1.0, 3.0]}\n"
        "  - {name: b, type: linear-range, min: 0, max: 1, n: 3}\n"
        "  - {name: c, type: log-range, min: 1.0e+2, max: 1e4, n: 3}\n"
        "Sampling:\n",
    )
    params = load_params(path, ["variant"], ["max_age"], ["a", "b", "c"])
    assert params.model == {"variant": "bin"}
    # 1e8 is a float, not the string YAML 1.1 would give
    assert params.constants["max_age"] == 1e8
    assert params.axes["a"].tolist() == [1.0, 3.0]
    assert params.axes["b"].tolist() == [0.0, 0.5, 1.0]
    np.testing.assert_allclose(params.axes["c"], [1e2, 1e3, 1e4])

    with pytest.raises(ValueError, match="unknown Model keys"):
        load_params(path, [], ["max_age"], ["a", "b", "c"])
    with pytest.raises(ValueError, match="axis 'c' must be one of"):
        load_params(path, ["variant"], ["max_age"], ["a", "b"])


def test_load_params_rejects_bad_entries(tmp_path):
    base = "Model:\nConstants:\n"
    cases = {
        "Axes:\n  - {name: a, type: linear-range, min: 0}\nSampling:\n": (
            r"needs \['max', 'n'\]"
        ),
        "Axes:\n  - {name: a, type: sobol, min: 1, max: 2, scale: Log}\n"
        "Sampling:\n  n_samples: 4\n": "scale must be linear or log",
        "Axes:\nSampling:\n  nsamples: 4\n": "unknown Sampling keys",
        "Axes:\n  - a\nSampling:\n": "each axis must be a mapping",
    }
    for text, message in cases.items():
        with pytest.raises(ValueError, match=message):
            load_params(
                _write(tmp_path, base + text),
                axis_names=["a"],
                allow_sobol=True,
            )


def test_load_params_needs_every_section(tmp_path):
    path = _write(tmp_path, "Model:\nConstants:\nAxes:\n")
    with pytest.raises(ValueError, match="missing \\['Sampling'\\]"):
        load_params(path)


def test_load_params_sobol(tmp_path):
    path = _write(
        tmp_path,
        "Model:\nConstants:\n"
        "Axes:\n"
        "  - {name: mass, type: sobol, min: 1.0e+6, max: 1.0e+10, "
        "scale: log}\n"
        "  - {name: spin, type: sobol, min: 0.0, max: 0.998}\n"
        "Sampling:\n  n_samples: 100\n  seed: 1\n",
    )
    params = load_params(path, axis_names=["mass", "spin"], allow_sobol=True)
    assert params.axes["mass"].shape == params.axes["spin"].shape == (100,)
    assert 1e6 <= params.axes["mass"].min() < params.axes["mass"].max() <= 1e10
    assert (
        0.0 <= params.axes["spin"].min() < params.axes["spin"].max() <= 0.998
    )

    with pytest.raises(ValueError, match="can't use Sobol"):
        load_params(path, axis_names=["mass", "spin"])
