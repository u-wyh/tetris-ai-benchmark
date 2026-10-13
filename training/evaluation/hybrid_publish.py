"""Publish completed hybrid report after the long-running evaluation queue exits."""
import subprocess
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import json

from training.evaluation.hybrid_queue import RUN
from training.train_ppo import ROOT, atomic_json, now


def record(message):
    current = datetime.now(ZoneInfo("Asia/Shanghai"))
    stamp = current.strftime("%H:%M")
    path = ROOT / "OPERATIONS.md"
    content = path.read_text()
    heading = f"## {current:%Y-%m-%d}\n\n"
    if heading not in content:
        anchor = "## 2026-10-13\n\n"
        if anchor not in content:
            raise RuntimeError("Operations date section missing")
        content = content.replace(anchor, heading + anchor, 1)
    path.write_text(content.replace(heading, heading + f"- {stamp}｜{message}\n", 1))


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
        record("Hybrid 后台队列失败；未生成完整确认报告。依据：`runs/candidate_ppo30m_bc_hybrid/status.json`、`queue.log`。")
        atomic_json(marker, {"state": "evaluation_failed", "updated_at": now()})
        return
    try:
        report = ROOT / "reports/experiments/candidate_ppo30m_bc_hybrid.md"
        if not report.exists() or "### confirm_100k" not in report.read_text():
            raise RuntimeError("Final hybrid confirmation report is missing")
        record("Hybrid 调参及独立确认评测完成，最终报告已生成并准备提交推送。依据：`runs/candidate_ppo30m_bc_hybrid/status.json`、`reports/experiments/candidate_ppo30m_bc_hybrid.md`。")
        paths = ["reports/experiments/candidate_ppo30m_bc_hybrid.md", "OPERATIONS.md"]
        result = subprocess.run(["git", "commit", "--only", "-m",
                                 "Report PPO30M BC hybrid confirmation results", "--", *paths],
                                cwd=ROOT, check=True, capture_output=True, text=True)
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT,
                       check=True, capture_output=True, text=True)
        atomic_json(marker, {"state": "completed", "updated_at": now(),
                             "commit_output": result.stdout.strip()})
    except Exception:
        atomic_json(marker, {"state": "failed", "updated_at": now(),
                             "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
