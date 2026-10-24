"""Record the source bytes used by a research evaluation process."""

import hashlib
import importlib.metadata
from pathlib import Path


SOURCE_PATHS = (
    "training/candidate_features.py",
    "training/env/candidate_env.py",
    "training/env/fast_candidate_env.py",
    "training/env/tetris_env.py",
    "training/evaluation/failure_curriculum_eval.py",
    "training/evaluation/fast_candidate_features.py",
    "training/evaluation/fast_counterfactual.py",
    "training/evaluation/fast_legal.py",
    "training/evaluation/hybrid_queue.py",
    "training/evaluation/ppo_clear_cadence_capture.py",
    "training/evaluation/ppo_clear_cadence_branches.py",
    "training/evaluation/source_provenance.py",
    "training/tetris_core/core.py",
    "training/tetris_core/actions.py",
    "training/tetris_core/legal_placements.py",
    "training/tetris_core/pieces.py",
    "training/tetris_core/rng.py",
    "training/traditional/v2.py",
    "training/train_ppo.py",
)
PACKAGES = ("torch", "stable-baselines3", "sb3-contrib", "numpy", "gymnasium")


def snapshot(root: Path) -> dict:
    """Capture source and package identities before a new evaluation starts."""
    return {
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in SOURCE_PATHS
        },
        "package_versions": {name: importlib.metadata.version(name) for name in PACKAGES},
    }


def verify(root: Path, record: dict) -> None:
    """Reject a changed execution environment when resuming a new record."""
    if record != snapshot(root):
        raise RuntimeError("Evaluation source or package versions changed")
