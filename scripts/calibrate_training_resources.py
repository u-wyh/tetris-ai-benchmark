"""Bounded full-training CPU/CUDA comparisons under the resource guard."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from resource_guard import ROOT, now, operation, read_telemetry, write_json


def run(output):
    # Import from the repository even when invoked as a script.
    sys.path.insert(0, str(ROOT))
    from training.evaluation.hybrid_queue import PPO, digest, verify_models
    from training.train_vector_transaction import verify_task
    verify_models()
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(at=now(),benchmark_only=True,steps_per_trial=8192,
                    seed=1493000,seed_usage="Reuse performance-benchmark seeds; no quality evaluation",
                    model=str(PPO),model_sha256=digest(PPO),
                    variants=[("cpu4", "cpu", 4), ("cuda4", "cuda", 4), ("cuda8", "cuda", 8)],
                    resume_steps=4096,trial_time_budget_seconds=600,
                    guard_profile=json.loads((ROOT / "configs/training_resources.json").read_text()))
    write_json(output / "protocol.json", protocol)
    operation("开始CPU/GPU资源校准，三组各8192步，优胜组恢复4096步；仅性能短测", output)
    results = []

    def trial(label, device, workers, resume=False):
        preset = output / (label + "_preset.json")
        directory = output / label
        if not resume:
            config = dict(target_total_steps=12288,task_steps=4096,device=device,
                          n_envs=workers,n_steps=4096//workers,base_seed=1493000,
                          candidate_policy=True,fast_candidate_env=True,max_pieces=100000,
                          overnight_monitor=True,step0_baseline=False,transaction_retention=3,
                          validation_interval=100000000,milestone_interval=100000000,
                          final_test_seeds=100,final_test_max_pieces=50000,
                          initial_model=dict(path=str(PPO),sha256=digest(PPO)),
                          ppo_overrides=dict(gamma=0.999,gae_lambda=0.98))
            write_json(preset,config)
        guard_output = output / (label + ("_resume_guard" if resume else "_guard"))
        command = [sys.executable,str(ROOT / "scripts/resource_guard.py"),"--output",str(guard_output),
                   "--seconds","600","--",sys.executable,"-u","-m","training.train_vector_transaction",
                   "--run-dir",str(directory),"--config-file",str(preset),
                   "--max-tasks","1" if resume else "2"]
        if resume:
            command.append("--resume")
        write_json(output / "status.json",dict(at=now(),stage="training",variant=label,resume=resume))
        completed = subprocess.run(command,cwd=ROOT)
        if completed.returncode:
            raise RuntimeError(f"{label} guard failed; inspect {guard_output}")
        config = json.loads((directory / "config.json").read_text())
        metrics = []
        numbers = [3] if resume else [1,2]
        for number in numbers:
            task = directory / "committed" / f"task_{number:06d}"
            verify_task(task,number,config)
            metrics.append(json.loads((task / "metrics.json").read_text()))
        samples = [json.loads(line) for line in (guard_output / "samples.jsonl").read_text().splitlines()]
        timed = [s for s in samples if s["elapsed_seconds"] >= 5] or samples
        seconds = sum(m["wall_seconds"] for m in metrics)
        result = dict(variant=label,device=device,workers=workers,resume=resume,
                      committed_steps=len(numbers)*4096,training_seconds=seconds,
                      steps_per_second=len(numbers)*4096/seconds,
                      task_training_seconds=[m["wall_seconds"] for m in metrics],
                      cpu_peak_c=max(s["cpu_c"] for s in samples),
                      gpu_peak_c=max(s["gpu_c"] for s in samples),
                      gpu_mean_util_percent=statistics.mean(s["gpu_util_percent"] for s in timed),
                      gpu_peak_memory_mib=max(s["gpu_memory_mib"] for s in samples),
                      gpu_mean_power_w=statistics.mean(s["gpu_power_w"] for s in timed if s["gpu_power_w"] is not None),
                      peak_memory_mib=max(s["memory_mib"] for s in samples),
                      cpu_mean_used_cores=statistics.mean(s["cpu_used_cores"] for s in timed if "cpu_used_cores" in s),
                      quota_changes=sum(s["decision"] in ("reduce","increase") for s in samples),
                      guard_result=json.loads((guard_output / "result.json").read_text()))
        write_json(output / (label + ("_resume_result.json" if resume else "_result.json")),result)
        print(json.dumps(result),flush=True)
        return result

    try:
        for label,device,workers in protocol["variants"]:
            # Begin each comparison after cooling to comparable startup conditions.
            deadline=time.monotonic()+180
            while True:
                sample=read_telemetry(True)
                if sample["cpu_c"]<65 and sample["gpu_c"]<55:
                    break
                if time.monotonic()>deadline:
                    raise RuntimeError("Cooldown budget exhausted")
                time.sleep(5)
            results.append(trial(label,device,workers))
            write_json(output / "partial_results.json",results)
        winner=max(results,key=lambda r:r["steps_per_second"])
        resumed=trial(winner["variant"],winner["device"],winner["workers"],resume=True)
        recommendation=dict(at=now(),device=winner["device"],n_envs=winner["workers"],
                            guard_profile="configs/training_resources.json",
                            benchmark_path=str(output),
                            reason="Highest observed training throughput within temperature policy; short benchmark only")
        write_json(ROOT / "configs/training_resources_recommended.json",recommendation)
        summary=dict(at=now(),results=results,resume_verification=resumed,
                     recommendation=recommendation,
                     model_unchanged=digest(PPO)==protocol["model_sha256"])
        write_json(output / "summary.json",summary)
        write_json(output / "status.json",dict(at=now(),stage="completed"))
        operation(f"资源校准完成；推荐{winner['variant']}，吞吐{winner['steps_per_second']:.1f}步/秒，恢复事务校验通过",output / "summary.json")
    except BaseException as exc:
        write_json(output / "status.json",dict(at=now(),stage="failed",error=repr(exc)))
        operation(f"资源校准中断：{exc!r}，保留已有结果",output)
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.output.resolve())
