"""Finalize and push the active-V2 diagnostic reports after the tmux queue ends."""
import json
import subprocess
import time
import traceback

from training.evaluation.hybrid_publish import record
from training.evaluation.local_correction_active_v2 import V2
from training.evaluation.local_correction_v2_materialize import main as materialize
from training.evaluation.local_correction_v2_report import generate
from training.train_ppo import ROOT, atomic_json, now

PATHS = ["reports/experiments/local_correction_failure_diagnosis.md",
         "reports/experiments/local_correction_counterfactual_v2.md",
         "reports/experiments/local_correction_dataset_v2.md", "OPERATIONS.md"]


def main():
    marker = V2 / "publish_status.json"
    if marker.exists() and json.loads(marker.read_text()).get("state") == "completed":
        return
    while True:
        status_path = V2 / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status["state"] in ("completed", "failed"):
                break
        time.sleep(20)
    try:
        if status["state"] == "completed":
            materialize()
            generate()
            decision = json.loads((V2 / "readiness.json").read_text())
            record(f"局部纠错主动反事实 V2 队列完成：新分支 {decision['verified_new']}，"
                   f"新增训练/验证 A {decision['train_A_new']}/{decision['validation_A_new']}，"
                   f"就绪判断 `{decision['state']}`；未自动训练。"
                   "依据：`runs/ppo30m_local_correction/active_v2/status.json`、"
                   "`dataset_manifest.json`、`readiness.json` 及三份 V2 报告。")
            paths = PATHS
        else:
            record("局部纠错主动反事实 V2 队列失败；已完成分支保持原子保存，未把不完整样本当作最终数据。"
                   "依据：`runs/ppo30m_local_correction/active_v2/status.json`、`queue.log`。")
            paths = ["OPERATIONS.md"]
        result = subprocess.run(["git", "commit", "--only", "-m",
                                 f"Report local correction active V2: {status['state']}",
                                 "--", *paths], cwd=ROOT, check=True, text=True,
                                capture_output=True)
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT, check=True,
                       text=True, capture_output=True)
        atomic_json(marker, {"state": "completed", "outcome": status["state"],
                             "updated_at": now(), "commit": result.stdout.strip()})
    except Exception:
        atomic_json(marker, {"state": "failed", "error": traceback.format_exc(),
                             "updated_at": now()})
        raise


if __name__ == "__main__":
    main()
