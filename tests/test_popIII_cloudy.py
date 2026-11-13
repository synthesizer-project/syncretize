"""Tests of the Cloudy set-up for Pop III stars and decoupled gas."""

from pathlib import Path

import pytest
import yaml
from synthesizer.exceptions import InconsistentParameter

from syncretize.cloudy import cloudy23, submission_scripts
from syncretize.cloudy.create_cloudy_input_grid import (
    create_cloudy_input,
    resolve_gas_metallicity,
)
from syncretize.cloudy.utils import get_cloudy_params

CLOUDY_DIR = Path(submission_scripts.__file__).parent
PARAM_DIR = CLOUDY_DIR / "params"


def _parameters(name, **overrides):
    """Return one model's parameters from a ready-made parameter file."""
    fixed, variable = get_cloudy_params(f"{name}.yaml", param_dir=PARAM_DIR)
    parameters = fixed | {key: values[0] for key, values in variable.items()}
    # The stars are metal free whatever the gas is
    parameters |= {"ages": 1e6, "metallicities": 0.0} | overrides
    # The line lists are named relative to the cloudy directory
    parameters["output_linelist"] = str(
        CLOUDY_DIR / parameters["output_linelist"]
    )
    return parameters


def _create_input(tmp_path, parameters):
    """Write a model's cloudy input and return its lines."""
    (tmp_path / "0").mkdir(exist_ok=True)
    create_cloudy_input(0, 0, parameters, None, str(tmp_path), cloudy23)
    return (tmp_path / "0" / "0.in").read_text().splitlines()


def test_gas_metallicity_defaults_to_the_incident_metallicity():
    """The gas has the stars' metallicity unless gas_metallicity is set."""
    assert resolve_gas_metallicity({"metallicities": 0.01}) == 0.01
    assert (
        resolve_gas_metallicity(
            {"metallicities": 0.0, "gas_metallicity": 1e-5}
        )
        == 1e-5
    )
    assert (
        resolve_gas_metallicity({"metallicities": 0.01, "gas_metallicity": 0})
        == 0
    )
    with pytest.raises(InconsistentParameter):
        resolve_gas_metallicity({"metallicities": 0.0, "gas_metallicity": -1})


def test_primordial_gas_has_no_metals(tmp_path):
    """Metal free gas has primordial helium and every metal turned off."""
    lines = _create_input(tmp_path, _parameters("c25.00-popIII-primordial"))

    helium = [line for line in lines if line.startswith("element abundance")]
    assert len(helium) == 1
    assert helium[0].split()[2] == "helium"
    # A primordial helium mass fraction of about 0.25
    number_ratio = 10 ** float(helium[0].split()[3])
    assert 4 * number_ratio / (1 + 4 * number_ratio) == pytest.approx(
        0.2485, abs=2e-3
    )

    # Each command is on its own line, with all the metals off and no grains
    off = [line for line in lines if line.startswith("elements")]
    assert len(off) == 28
    assert all(line.split()[2:] == ["off"] for line in off)
    assert not any("grains" in line for line in lines)

    # Cloudy stops on lines of elements that are off
    linelist = (tmp_path / "0" / "linelist.dat").read_text().splitlines()
    assert linelist and all(
        line.startswith(("H ", "He ")) for line in linelist
    )
    standard = (CLOUDY_DIR / "linelist-standard.dat").read_text().splitlines()
    assert set(linelist) <= set(standard)


def test_primordial_gas_rejects_grains(tmp_path):
    """Metal free gas can't have grains or depletion."""
    parameters = _parameters(
        "c25.00-popIII-primordial",
        grains="Orion",
        depletion_model="Jenkins2009_Gunasekera2021",
        depletion_scale=0.5,
    )
    with pytest.raises(InconsistentParameter):
        _create_input(tmp_path, parameters)


def test_enriched_gas_scales_with_gas_metallicity(tmp_path):
    """The gas abundances follow gas_metallicity, not the stars'."""

    def oxygen(gas_metallicity):
        parameters = _parameters(
            "c25.00-popIII-enriched", gas_metallicity=gas_metallicity
        )
        lines = _create_input(tmp_path, parameters)
        (line,) = [
            line
            for line in lines
            if line.startswith("element abundance oxygen")
        ]
        return float(line.split()[3])

    assert oxygen(1.42e-4) - oxygen(1.42e-6) == pytest.approx(2.0, abs=0.01)
    # 10^-2 Zsun with [alpha/Fe] = 0.4
    assert oxygen(1.42e-4) == pytest.approx(-5.18, abs=0.02)


@pytest.mark.parametrize("extra", ["carbon_enhanced", "high_z"])
def test_popIII_extra_parameter_files(tmp_path, extra):
    """The Pop III extras combine with the enriched parameter file."""
    fixed, variable = get_cloudy_params(
        f"popIII/{extra}.yaml", param_dir=PARAM_DIR
    )
    assert not variable
    parameters = _parameters("c25.00-popIII-enriched") | fixed
    lines = _create_input(tmp_path, parameters)
    if extra == "high_z":
        assert "CMB 10.0" in lines
    else:
        abundance = {
            line.split()[2]: float(line.split()[3])
            for line in lines
            if line.startswith("element abundance")
        }
        assert abundance["carbon"] - abundance["oxygen"] == pytest.approx(0.3)


def test_cosma7_job_script(tmp_path):
    """The COSMA7 script packs the models onto whole nodes."""
    filename = submission_scripts.cosma7(
        "grid",
        number_of_incident_grid_points=111,
        number_of_photoionisation_models=45,
        cloudy_output_dir=str(tmp_path),
        cloudy_executable_path="/path/to/cloudy/bin",
    )
    script = Path(filename).read_text()
    # 4995 models at 4 per core on 28 cores
    assert "#SBATCH --array=0-44\n" in script
    assert "N_INCIDENT=111\n" in script
    assert 'CLOUDY_EXE="/path/to/cloudy/bin/cloudy.exe"' in script
    assert (tmp_path / "grid" / "logs").is_dir()

    # Rerunning failed models reads the grid's shape from its parameters
    with open(tmp_path / "grid" / "grid_parameters.yaml", "w") as file:
        yaml.dump(
            {"incident_n_models": 3, "photoionisation_n_models": 2}, file
        )
    filename = submission_scripts.cosma7(
        "grid",
        number_of_jobs=1,
        cloudy_output_dir=str(tmp_path),
        cloudy_executable_path="/path/to/cloudy/bin",
        from_list="failed.dat",
    )
    assert "#SBATCH --array=0-0\n" in Path(filename).read_text()
