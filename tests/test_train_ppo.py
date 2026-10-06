"""Small real PPO lifecycle test; no long smoke run is started here."""

import csv
import json
import subprocess
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO

from training.train_ppo import default_config

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args):
    return subprocess.run([sys.executable, "-m", "training.train_ppo", *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, check=True)


def test_default_smoke_config():
    config = default_config()
    assert config["target_total_steps"] == 50000
    assert config["effective_target_boundary"] == 50176
    assert config["first_stage_steps"] == 20480
    assert config["chunk_steps"] == 5120
    assert config["ppo"]["n_steps"] == 1024
    assert config["ppo"]["n_envs"] == 1


def test_pause_checkpoint_resume_and_metrics(tmp_path):
    run_dir = tmp_path / "lifecycle"
    run_dir.mkdir()
    (run_dir / "PAUSE_REQUESTED").touch()
    invoke("--run-dir", run_dir, "--target-steps", 2048,
           "--chunk-steps", 1024, "--first-stage-steps", 2048,
           "--eval-every", 1024, "--checkpoint-every", 1024)
    first_status = json.loads((run_dir / "status.json").read_text())
    assert first_status["status"] == "paused" and first_status["steps"] == 1024
    latest = run_dir / "checkpoints" / "latest.zip"
    assert latest.exists() and (run_dir / "checkpoints" / "step_001024.zip").exists()
    assert (run_dir / "checkpoints" / "best.zip").exists()
    model = MaskablePPO.load(str(latest), device="cpu")
    assert model.num_timesteps == 1024
    assert len(model.policy.optimizer.state) > 0
    assert (run_dir / "trainer_state.pkl").exists()
    assert list((run_dir / "tensorboard").glob("events.out.tfevents.*"))

    invoke("--run-dir", run_dir, "--resume")
    second_status = json.loads((run_dir / "status.json").read_text())
    assert second_status["status"] == "completed" and second_status["steps"] == 2048
    resumed_model = MaskablePPO.load(str(latest), device="cpu")
    assert resumed_model.num_timesteps == 2048
    assert len(resumed_model.policy.optimizer.state) > 0
    assert (run_dir / "checkpoints" / "final.zip").exists()
    assert (run_dir / "checkpoints" / "step_002048.zip").exists()
    with (run_dir / "training_metrics.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(row["timestep"]) for row in rows] == [1024, 2048]
    assert all(row["policy_loss"] and row["value_loss"] and row["approx_kl"] for row in rows)
    assert "RESUMED previous_steps=1024" in (run_dir / "events.log").read_text()
    assert len(list(csv.DictReader((run_dir / "evaluations.csv").open()))) == 2


def test_tmux_detaches_from_launching_shell():
    # A detached session remains alive after the shell command that created it exits.
    name = f"tetris-test-{__import__('os').getpid()}"
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", name, "sleep 5"], check=True)
        subprocess.run(["tmux", "has-session", "-t", name], check=True)
    finally:
        subprocess.run(["tmux", "kill-session", "-t", name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
