"""Commit and push the completed Hybrid V2 comparison without touching other work."""
import json
import subprocess
import time
import traceback

from training.evaluation.hybrid_v2_queue import RUN
from training.evaluation.hybrid_publish import record
from training.train_ppo import ROOT, atomic_json, now


def main():
    marker = RUN / "publish_status.json"
    if marker.exists() and json.loads(marker.read_text()).get("state") == "completed":
        return
    while True:
        status = json.loads((RUN / "status.json").read_text())
        if status["state"] in ("completed", "failed"):
            break
        time.sleep(30)
    if status["state"] == "failed":
        record("Hybrid V2 后台队列失败或中断；未把不完整结果发布为确认结论。依据：`runs/hybrid_v2/status.json`、`queue.log`。")
        atomic_json(marker, {"state": "evaluation_failed", "updated_at": now()})
        return
    try:
        report = ROOT / "reports/experiments/hybrid_v2_comparison.md"
        if not report.exists() or "confirmation_100k" not in report.read_text():
            raise RuntimeError("Hybrid V2 final comparison report missing")
        record("Hybrid V2 独立 32-seed 确认评测完成，最终比较报告已生成并提交推送。依据：`runs/hybrid_v2/status.json`、`reports/experiments/hybrid_v2_comparison.md`。")
        paths = ["reports/experiments/hybrid_v2_comparison.md", "OPERATIONS.md"]
        result = subprocess.run(["git", "commit", "--only", "-m",
                                 "Report Hybrid V2 independent confirmation", "--", *paths],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT,
                       capture_output=True, text=True, check=True)
        atomic_json(marker, {"state": "completed", "updated_at": now(),
                             "commit_output": result.stdout.strip()})
    except Exception:
        atomic_json(marker, {"state": "failed", "updated_at": now(),
                             "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
