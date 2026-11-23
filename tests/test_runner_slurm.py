"""Tests for the Runner's Slurm jobs, with stand-ins for sbatch and sacct."""

import os
import subprocess
import sys

import pytest
from test_runner import ROOT, cloudy_run, write_run

from syncretize.runner import Runner

# sbatch prints job IDs 1, 2, ... and keeps each script as <id>.sh; sacct -j
# <id> prints the lines of $FAKE_SACCT for that job
SBATCH = f"""#!{sys.executable}
import os, shutil, sys
directory = os.environ["FAKE_SLURM_DIR"]
job = len([name for name in os.listdir(directory) if name.endswith(".sh")])
shutil.copy(sys.argv[-1], os.path.join(directory, f"{{job + 1}}.sh"))
print(f"{{job + 1}};cosma")
"""
SACCT = f"""#!{sys.executable}
import os, sys
for line in os.environ.get("FAKE_SACCT", "").splitlines():
    if line.split("|")[0].split("_")[0] == sys.argv[-1]:
        print(line)
"""

SLURM = {"partition": "cosma8-serial", "account": "dp004"}


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    """Put fake sbatch and sacct on PATH, keeping each job's script.

    Args:
        tmp_path (pathlib.Path)
            The directory to put the fakes and the submitted scripts in.
        monkeypatch (pytest.MonkeyPatch)
            Sets PATH and $FAKE_SLURM_DIR, and stops the finish job waiting
            for sacct.

    Returns:
        callable
            Takes a job ID and returns the lines of its submitted script.
    """
    bin_dir, jobs_dir = tmp_path / "bin", tmp_path / "submitted"
    bin_dir.mkdir()
    jobs_dir.mkdir()
    for name, text in (("sbatch", SBATCH), ("sacct", SACCT)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(jobs_dir))
    # The fake sacct doesn't lag, so the finish job needn't wait for it
    monkeypatch.setattr("syncretize.runner.section.SACCT_WAITS", (0, 0))

    def script(job):
        """The script submitted as a job.

        Args:
            job (int or str)
                The job's ID.

        Returns:
            list of str
                The script's lines.
        """
        return (jobs_dir / f"{job}.sh").read_text().splitlines()

    return script


def test_run_submits_the_start_job(tmp_path, jobs):
    """Run submits the start job with the run's Slurm options."""
    runner = Runner(
        write_run(
            tmp_path,
            Run={
                "name": "test",
                "work_dir": "work",
                "grid_dir": "grids",
                "environment": "source env.sh",
                "slurm": SLURM,
                "incident": {"cores": 2},
            },
        )
    )
    assert runner.run() == "1"

    script = jobs(1)
    assert script[0] == "#!/bin/bash -l"
    for option in (
        "--job-name=R0-test-Incident",
        f"--chdir={tmp_path / 'work'}",
        "--time=0-12:00:00",
        "--cpus-per-task=2",
        "--partition=cosma8-serial",
        "--account=dp004",
    ):
        assert f"#SBATCH {option}" in script
    assert not any("--mem" in line or "--array" in line for line in script)
    assert script[-2] == "source env.sh"
    assert script[-1].endswith(
        f"-m syncretize.runner start {tmp_path / 'run.yaml'}"
    )


def submitted_round(tmp_path, monkeypatch, jobs, **cloudy):
    """Run the start job's work (Cloudy setup, first round) for a run.

    Args:
        tmp_path (pathlib.Path)
            The directory to put the run in.
        monkeypatch (pytest.MonkeyPatch)
            Passed on to cloudy_run.
        jobs (callable)
            The jobs fixture, so its fake sbatch is on PATH.
        **cloudy
            The run's cloudy settings (Run.cloudy), e.g. workers=100.

    Returns:
        (Section, Queue)
            The run's Cloudy section, and its queue.
    """
    section, queue = cloudy_run(tmp_path, monkeypatch, {}, **cloudy)
    section.config.run["slurm"] = SLURM
    section.runner.start()
    return section, queue


def test_start_submits_workers_then_finish(tmp_path, monkeypatch, jobs):
    """Start submits the workers' array, then a finish job after it."""
    submitted_round(tmp_path, monkeypatch, jobs, workers=100)

    # 4 x 48 models need only 48 workers of 4
    workers, finish = jobs(1), jobs(2)
    assert "#SBATCH --job-name=R1-test-C1-Cloudy" in workers
    assert "#SBATCH --job-name=R1-test-C1-Runner" in finish
    assert "#SBATCH --array=0-47" in workers
    assert "#SBATCH --cpus-per-task=4" in workers
    assert "#SBATCH --time=0-12:00:00" in workers
    assert workers[-1].endswith(
        "work "
        + str(tmp_path / "run.yaml")
        + " --section 1 --walltime 0-12:00:00"
    )
    assert "#SBATCH --dependency=afterany:1" in finish
    assert "#SBATCH --time=0-01:00:00" in finish
    assert finish[-1].endswith(
        "finish " + str(tmp_path / "run.yaml") + " --section 1"
    )


def test_finish_requeues_and_submits_a_longer_round(
    tmp_path, monkeypatch, jobs
):
    """Finish requeues ended workers' claims and submits a longer round."""
    section, queue = submitted_round(tmp_path, monkeypatch, jobs, workers=10)
    queue.claim("1_3.0", n=2)
    queue.claim("1_4.1")

    # A claim of a running worker stops the finish job
    monkeypatch.setenv("FAKE_SACCT", "1_3|TIMEOUT\n1_4|RUNNING")
    with pytest.raises(RuntimeError, match="still claimed"):
        section.continue_run()

    monkeypatch.setenv("FAKE_SACCT", "1_3|TIMEOUT\n1_4|CANCELLED by 12")
    assert section.continue_run() == "4"
    assert queue.counts()["todo"] == 4 * 48
    assert "#SBATCH --time=0-18:00:00" in jobs(3)
    assert "#SBATCH --job-name=R2-test-C1-Cloudy" in jobs(3)
    assert "#SBATCH --array=0-9" in jobs(3)
    assert "#SBATCH --dependency=afterany:3" in jobs(4)


def test_finish_submits_assembly_when_done(tmp_path, monkeypatch, jobs):
    """Finish submits the assembly job once every model is done."""
    section, queue = submitted_round(tmp_path, monkeypatch, jobs)
    section.config.run["assemble"]["memory"] = "32G"
    for model in queue.claim("1_0.0", n=4 * 48):
        queue.done(model, "1_0.0")

    assert section.continue_run() == "3"
    assert "#SBATCH --job-name=R1-test-C1-Assemble" in jobs(3)
    assert "#SBATCH --mem=32G" in jobs(3)


def test_worker_names_its_claims_after_its_array_task(tmp_path, monkeypatch):
    """A worker names its claims after its Slurm array job and task."""
    section, queue = cloudy_run(tmp_path, monkeypatch, {})
    section.setup(section.config.incident["grid"])
    result = subprocess.run(
        [sys.executable, "-m", "syncretize.runner", "work"]
        + [section.config.path, "--walltime", "00:20:00"],
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "SLURM_ARRAY_JOB_ID": "7",
            "SLURM_ARRAY_TASK_ID": "2",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert queue.counts()["done"] == 4 * 48
    assert "7_2.0 " in result.stdout


def test_resume_resubmits_a_stopped_job(tmp_path, monkeypatch, jobs):
    """Resume waits for a queued start job and resubmits a stopped one."""
    runner = Runner(write_run(tmp_path))
    assert runner.run() == "1"
    with pytest.raises(ValueError, match="use resume"):
        runner.run()

    # Nothing while the start job is queued or running
    monkeypatch.setenv("FAKE_SACCT", "1|PENDING")
    assert runner.resume() is None

    # The start job records itself in the state file, and how it ended
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    runner.job_started("start")
    runner.job_ended("start", "failed: OSError()")
    (entry,) = runner._load_state()["jobs"]
    assert (
        entry["stage"] == "start" and entry["outcome"] == "failed: OSError()"
    )
    assert {"submitted", "started", "ended"} <= set(entry)

    # Once it has stopped, resume submits it again
    monkeypatch.setenv("FAKE_SACCT", "1|FAILED")
    assert runner.resume() == "2"
    assert jobs(2)[-1].endswith(f"start {tmp_path / 'run.yaml'}")
    assert runner._load_state()["next_job"] == "2"


def test_resume_finishes_a_round_after_its_workers(
    tmp_path, monkeypatch, jobs
):
    """Resume resubmits a lost finish job to run after its workers."""
    section, queue = submitted_round(tmp_path, monkeypatch, jobs)
    # The finish job (2) was cancelled while the workers (1) still run
    monkeypatch.setenv("FAKE_SACCT", "1_0|RUNNING\n2|CANCELLED by 5")
    assert section.runner.resume() == ["3"]
    assert "#SBATCH --dependency=afterany:1" in jobs(3)
    assert jobs(3)[-1].endswith(f"finish {tmp_path / 'run.yaml'} --section 1")


def test_resume_waits_for_a_longer_max_walltime(tmp_path, monkeypatch, jobs):
    """A section out of time at max_walltime resumes once it's raised."""
    section, queue = submitted_round(tmp_path, monkeypatch, jobs)
    monkeypatch.setenv("FAKE_SACCT", "1_0|TIMEOUT\n2|COMPLETED")
    section._save_state(walltime="3-00:00:00")
    queue.claim("1_0.0")
    assert section.continue_run() is None
    assert section._load_state()["next"] == "walltime"

    # Nothing to do until max_walltime is longer
    assert section.resume() is None
    section.config.run["cloudy"]["max_walltime"] = "4-00:00:00"
    assert section.resume() == "4"
    assert "#SBATCH --time=4-00:00:00" in jobs(3)
    state = section._load_state()
    assert (state["next"], state["walltime"]) == ("finish", "4-00:00:00")
    assert queue.counts()["failed"] == 0


def test_start_submits_a_round_for_each_section(tmp_path, monkeypatch, jobs):
    """Start submits a round for each section, and resume carries each on."""
    density = tmp_path / "density.yaml"
    density.write_text("hydrogen_density: 100.0\n")
    section, _ = cloudy_run(
        tmp_path,
        monkeypatch,
        {},
        more={
            "Cloudy2": {
                "config": "c23.01-sps-grid_limited",
                "extra_config": str(density),
            }
        },
    )
    runner = section.runner
    runner.config.run["slurm"] = SLURM
    runner.start()

    # Each section's workers and finish job, named and run for it
    assert "#SBATCH --job-name=R1-test-C1-Cloudy" in jobs(1)
    assert "#SBATCH --job-name=R1-test-C2-Cloudy" in jobs(3)
    assert "#SBATCH --job-name=R1-test-C2-Runner" in jobs(4)
    assert jobs(4)[-1].endswith("--section 2")
    assert runner._load_state()["next"] == "cloudy"

    # Resume carries on each section that has stopped
    monkeypatch.setenv("FAKE_SACCT", "1_0|RUNNING\n2|PENDING\n4|FAILED")
    assert runner.resume() == [None, "5"]
    assert jobs(5)[-1].endswith(
        "finish " + str(tmp_path / "run.yaml") + " --section 2"
    )
