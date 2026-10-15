"""Independent tmux runner: wait for capture, validate, analyze, branch, report, test, publish."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from training.evaluation.hybrid_queue import verify_models
from training.evaluation.ppo_death_capture import RUN
from training.evaluation.ppo_death_replay import validate
from training.train_ppo import ROOT, atomic_json, now


def status(stage, **extra):
    atomic_json(RUN / "pipeline_status.json", {"stage": stage, "pid": os.getpid(),
                                               "updated_at": now(), **extra})
    print(f"{now()} {stage} {extra}", flush=True)


def operation(message, evidence):
    with (ROOT / "OPERATIONS.md").open("a") as file:
        file.write(f"\n- {now()}｜{message}。依据：{evidence}。\n")


def command(args, log_name):
    with (RUN / log_name).open("w") as log:
        subprocess.run(args, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def wait_capture():
    status("waiting_for_capture")
    while True:
        if (RUN / "capture_completed.json").exists():
            return json.loads((RUN / "capture_completed.json").read_text())
        capture_status = RUN / "status.json"
        if capture_status.exists():
            row = json.loads(capture_status.read_text())
            if row.get("stage") == "capture_failed":
                raise RuntimeError(f"Capture failed: {row.get('error')}")
        if subprocess.run(["tmux","has-session","-t","tetris-ppo-death-analysis"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            raise RuntimeError("Capture tmux exited without capture_completed.json")
        time.sleep(60)


def _replay_worker(seed):
    return validate(seed)


def replay_all(seeds):
    status("replay_validation", completed=0, total=len(seeds))
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(_replay_worker, seed): seed for seed in seeds}
        for completed, future in enumerate(as_completed(futures),1):
            result = future.result()
            status("replay_validation", completed=completed, total=len(seeds),
                   last_seed=result["seed"])


def main():
    try:
        verify_models()
        capture = wait_capture()
        seeds = capture["initial_seeds"] + capture["extra_seeds"]
        replay_all(seeds)
        operation(f"死亡分析 {len(seeds)} 局完成逐 seed 确定性重放校验", "runs/ppo30m_death_analysis/replay_validation/")

        from training.evaluation.ppo_death_analysis import analyze, figures
        status("death_window_analysis")
        analysis = analyze()
        figures(analysis)
        operation(f"死亡窗口、对照和探索性分类完成：真实死亡 {analysis['death_count']} 局，100K 右截断 {analysis['right_censored_count']} 局",
                  "runs/ppo30m_death_analysis/death_analysis.json")

        from training.evaluation.ppo_death_counterfactual import prepare, evaluate
        status("counterfactual_preparation")
        prepared = prepare()
        operation(f"预先确定反事实状态 {prepared[0]} 个、合法动作分支 {prepared[1]} 个",
                  "runs/ppo30m_death_analysis/counterfactual_prepared.json")
        status("counterfactual_1000")
        evaluate(1000)
        status("conditional_counterfactual_5000")
        evaluate(5000)
        operation("1000 块与有条件 5000 块反事实原子分支完成",
                  "runs/ppo30m_death_analysis/counterfactual_{1000,5000}_completed.json")

        from training.evaluation.ppo_death_report import write_reports
        status("reports")
        write_reports()
        status("python_tests")
        command([sys.executable,"-m","pytest","-q"], "python_tests.log")
        status("parity_tests")
        command([sys.executable,"-m","unittest","discover","-s","tests/parity","-v"], "parity_tests.log")
        status("js_tests")
        command(["node","--test",*[str(path) for path in sorted((ROOT/"tests").glob("*.test.cjs"))]],
                "js_tests.log")
        operation("死亡机制报告与 Python、Parity、JS 回归测试完成",
                  "reports/experiments/ppo30m_death_*；runs/ppo30m_death_analysis/*_tests.log")
        status("publishing")
        # Explicit path list protects unrelated dirty workspace files.
        subprocess.run(["git","add","OPERATIONS.md","CHANGELOG.md",
                        "reports/experiments/ppo30m_death_mechanism_results.md",
                        "reports/experiments/ppo30m_death_counterfactual_analysis.md"], cwd=ROOT, check=True)
        if subprocess.run(["git","diff","--cached","--quiet"], cwd=ROOT).returncode == 1:
            subprocess.run(["git","commit","-m","Report PPO30M death mechanisms and counterfactuals"],
                           cwd=ROOT, check=True)
        subprocess.run(["git","push","origin","main"], cwd=ROOT, check=True)
        commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
        status("completed", commit=commit, deaths=analysis["death_count"], states=prepared[0])
    except BaseException as exc:
        status("failed", error=repr(exc), traceback=traceback.format_exc())
        operation(f"死亡分析后台队列停止：{repr(exc)}", "runs/ppo30m_death_analysis/pipeline_status.json")
        raise


if __name__ == "__main__":
    main()
