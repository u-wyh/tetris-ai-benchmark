"""Publish the final scientific outcome, including a justified early stop."""
import json
import subprocess
import time
import traceback

from training.evaluation.hybrid_publish import record
from training.local_correction_data import RUN
from training.train_ppo import ROOT, atomic_json, now

TERMINAL = {"completed", "insufficient_verified_corrections",
            "v1_failed_short_validation", "v1_failed_long_validation",
            "no_promising_candidate", "failed"}
REPORTS = ["reports/experiments/ppo30m_local_correction_dataset.md",
           "reports/experiments/ppo30m_local_correction_training.md",
           "reports/experiments/ppo30m_local_correction_comparison.md"]


def main():
    marker = RUN / "publish_status.json"
    if marker.exists() and json.loads(marker.read_text()).get("state") == "completed":
        return
    while True:
        status = json.loads((RUN / "status.json").read_text())
        if status["state"] in TERMINAL:
            break
        time.sleep(30)
    try:
        state = status["state"]
        if state == "failed":
            record("PPO 30M 局部纠错后台队列失败；保留已提交的局与 checkpoint，未把不完整结果作为结论。依据：`runs/ppo30m_local_correction/status.json`、`queue.log`。")
            paths = ["OPERATIONS.md"]
        else:
            for name in REPORTS:
                path = ROOT / name
                if not path.exists() or f"状态：`{state}`" not in path.read_text():
                    raise RuntimeError(f"Terminal report missing or stale: {name}")
            record(f"PPO 30M 局部纠错队列结束，结果状态为 `{state}`；数据、训练及比较报告已生成。依据：`runs/ppo30m_local_correction/status.json`、`reports/experiments/ppo30m_local_correction_comparison.md`。")
            paths = [*REPORTS, "OPERATIONS.md"]
        result = subprocess.run(["git", "commit", "--only", "-m",
                                 f"Report PPO30M local correction: {state}", "--", *paths],
                                cwd=ROOT, text=True, capture_output=True, check=True)
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT,
                       text=True, capture_output=True, check=True)
        atomic_json(marker, {"state": "completed", "outcome": state,
                             "updated_at": now(), "commit": result.stdout.strip()})
    except Exception:
        atomic_json(marker, {"state": "failed", "updated_at": now(),
                             "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
