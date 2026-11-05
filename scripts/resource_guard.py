"""Run a command in an isolated user service with bounded resources and telemetry.

Example: .venv/bin/python scripts/resource_guard.py --output runs/guarded_trial \
    --seconds 600 -- .venv/bin/python -m training.train_vector_transaction ...
Temperature thresholds are conservative operating targets, not chip ratings.
"""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def operation(message, output):
    with (ROOT / "OPERATIONS.md").open("a") as f:
        f.write(f"\n- {now()}：资源保护：{message}。证据：{output}。\n")


def physical_cpus(reserve, limit):
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((base / "physical_package_id").read_text().strip(),
               (base / "core_id").read_text().strip())
        cores.setdefault(key, cpu)
    choices = list(cores.values())
    if reserve < 0 or limit < 1 or len(choices) <= reserve:
        raise ValueError("Not enough physical cores after reservation")
    return choices[reserve:reserve + limit]


def read_telemetry(require_gpu):
    result = {"at": now()}
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        name = (hwmon / "name").read_text().strip()
        if name == "k10temp":
            result["cpu_c"] = int((hwmon / "temp1_input").read_text()) / 1000
        elif name == "asus" and (hwmon / "fan1_input").exists():
            result["cpu_fan_rpm"] = int((hwmon / "fan1_input").read_text())
    if "cpu_c" not in result or not math.isfinite(result["cpu_c"]) or result["cpu_c"] <= 0:
        raise RuntimeError("CPU temperature sensor unavailable")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
    if gpu.returncode == 0:
        parts = gpu.stdout.splitlines()[0].split(",")
        result["gpu_c"] = float(parts[0])
        result["gpu_util_percent"] = float(parts[1])
        result["gpu_memory_mib"] = float(parts[2])
        result["gpu_power_w"] = None if parts[3].strip() in ("[N/A]", "N/A") else float(parts[3])
    elif require_gpu:
        raise RuntimeError("GPU telemetry unavailable: " + gpu.stderr.strip())
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            result["available_mib"] = int(line.split()[1]) / 1024
    return result


def control(sample, quota, cool_since, clock, profile):
    """Return a quota and an action; missing/invalid required readings fail closed."""
    required = ["cpu_c", "available_mib"]
    if profile["require_gpu_telemetry"]:
        required.append("gpu_c")
    if any(k not in sample or not math.isfinite(sample[k]) for k in required):
        return quota, None, "stop:invalid_telemetry"
    cpu, gpu = sample["cpu_c"], sample.get("gpu_c", 0)
    if cpu >= profile["cpu_stop_c"] or gpu >= profile["gpu_stop_c"]:
        return quota, None, "stop:temperature"
    if sample["available_mib"] < profile["minimum_available_mib"]:
        return quota, None, "stop:memory_pressure"
    if cpu >= profile["cpu_soft_c"] or gpu >= profile["gpu_soft_c"]:
        return max(profile["min_cpu_quota"], quota - profile["quota_step"]), None, "reduce"
    if cpu < profile["cpu_soft_c"] - 7 and gpu < profile["gpu_soft_c"] - 7:
        cool_since = clock if cool_since is None else cool_since
        if clock - cool_since >= profile["cooldown_seconds"]:
            return min(profile["max_cpu_quota"], quota + profile["quota_step"]), clock, "increase"
        return quota, cool_since, "hold"
    return quota, None, "hold"


def systemctl(*args, check=True):
    return subprocess.run(["systemctl", "--user", *args], capture_output=True,
                          text=True, check=check, timeout=10)


def unit_state(unit):
    values = systemctl("show", unit, "-p", "ActiveState", "-p", "SubState", "-p", "Result",
                       "-p", "ExecMainStatus", "-p", "ControlGroup").stdout
    return dict(line.split("=", 1) for line in values.splitlines() if "=" in line)


def terminal_result(state):
    if state.get("SubState") == "exited":
        code = int(state.get("ExecMainStatus", "1"))
        return ("completed" if code == 0 else "command_failed"), code
    if state.get("ActiveState") in ("failed", "inactive"):
        return "command_failed:" + state.get("Result", "unknown"), 1
    return None


def run(args):
    profile = json.loads(args.profile.read_text())
    if not (0 < profile["min_cpu_quota"] <= profile["initial_cpu_quota"] <= profile["max_cpu_quota"]
            and profile["quota_step"] > 0 and profile["sample_seconds"] > 0
            and profile["cooldown_seconds"] > 0 and args.seconds > 0
            and profile["cpu_soft_c"] < profile["cpu_stop_c"]
            and profile["gpu_soft_c"] < profile["gpu_stop_c"]):
        raise ValueError("Invalid resource profile")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A child command is required")
    if not os.path.isabs(command[0]):
        # Preserve a virtualenv's executable path: resolving its symlink selects
        # the system interpreter and silently loses installed training packages.
        command[0] = os.path.abspath(ROOT / command[0])
    cpus = physical_cpus(profile["reserve_physical_cores"], profile["max_physical_cores"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # No workload exists during preflight. A cold NVIDIA query may time out;
    # retry once with fresh telemetry, never reuse a failed or stale reading.
    try:
        for attempt in range(2):
            try:
                baseline = read_telemetry(profile["require_gpu_telemetry"])
                break
            except subprocess.TimeoutExpired as exc:
                write_json(output / "preflight.json", dict(at=now(), attempt=attempt + 1,
                                                          error=repr(exc)))
                if attempt:
                    raise
                time.sleep(2)
        quota = profile["initial_cpu_quota"]
        quota, _, decision = control(baseline, quota, None, 0, profile)
        if decision.startswith("stop:"):
            raise RuntimeError("Unsafe startup conditions: " + decision)
    except Exception as exc:
        result = dict(at=now(), outcome="startup_failed", exit_code=1, error=repr(exc))
        write_json(output / "result.json", result)
        operation(f"启动前检查失败，未启动计算：{exc!r}", output)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1
    unit = f"tetris-guard-{os.getpid()}-{time.time_ns()}"
    metadata = dict(at=now(),unit=unit,profile=profile,physical_cpus=cpus,
                    command=command,seconds=args.seconds,baseline=baseline)
    write_json(output / "launch.json", metadata)
    write_json(output / "heartbeat.json", dict(pid=os.getpid(),monotonic=time.monotonic()))
    # systemd owns the whole process tree. Stopping the service also kills forkserver workers.
    launch = ["systemd-run", "--user", "--quiet", "--unit=" + unit,
              "--service-type=exec", "--remain-after-exit", "--working-directory=" + str(ROOT),
              "-p", f"CPUQuota={quota}%", "-p", "CPUQuotaPeriodSec=20ms",
              "-p", f"MemoryMax={profile['memory_max_gib']}G", "-p", "MemorySwapMax=0",
              "-p", f"RuntimeMaxSec={args.seconds}s", "-p", "TimeoutStopSec=10s",
              "-p", "KillMode=control-group", "-p", "OOMPolicy=stop",
              "-p", "StandardOutput=append:" + str(output / "command.log"),
              "-p", "StandardError=append:" + str(output / "command.log")]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        launch.append(f"--setenv={key}=1")
    # User services do not inherit the shell's Node/Python toolchain paths.
    launch.append("--setenv=PATH=" + os.environ["PATH"])
    launch += ["/usr/bin/taskset", "--cpu-list", ",".join(map(str, cpus)),
               sys.executable,str(ROOT / "scripts/guarded_process.py"),
               "--guardian-pid",str(os.getpid()),"--heartbeat",str(output / "heartbeat.json"),
               "--",*command]
    started = time.monotonic()
    cool_since = None
    outcome, code, previous_usage = "technical_failure", 1, None
    stopping = False

    def interrupted(signum, frame):
        nonlocal stopping
        stopping = True

    old_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        subprocess.run(launch, check=True, timeout=15)
        operation(f"启动受保护任务 {unit}，CPU亲和性{cpus}，初始配额{quota}%，内存上限{profile['memory_max_gib']}GiB", output)
        while True:
            write_json(output / "heartbeat.json", dict(pid=os.getpid(),monotonic=time.monotonic()))
            state = unit_state(unit)
            terminal = terminal_result(state)
            if terminal is not None:
                outcome, code = terminal
                break
            if stopping or time.monotonic() - started >= args.seconds:
                outcome = "interrupted" if stopping else "time_budget"
                break
            sample = read_telemetry(profile["require_gpu_telemetry"])
            clock = time.monotonic()
            new_quota, cool_since, decision = control(sample, quota, cool_since, clock, profile)
            cgroup = Path("/sys/fs/cgroup") / state["ControlGroup"].lstrip("/")
            try:
                stats = dict(line.split() for line in (cgroup / "cpu.stat").read_text().splitlines())
                memory_mib = int((cgroup / "memory.current").read_text()) / 1024**2
            except FileNotFoundError:
                # systemd can remove an empty cgroup after the state query but
                # before sampling. Only a confirmed terminal unit resolves it.
                terminal = terminal_result(unit_state(unit))
                if terminal is None:
                    raise
                outcome, code = terminal
                break
            usage = int(stats["usage_usec"])
            sample.update(elapsed_seconds=round(clock-started, 3),cpu_quota_percent=quota,
                          decision=decision,cgroup_cpu_usage_usec=usage,
                          memory_mib=memory_mib)
            if previous_usage is not None:
                sample["cpu_used_cores"] = (usage-previous_usage[0]) / 1e6 / (clock-previous_usage[1])
            previous_usage = (usage,clock)
            with (output / "samples.jsonl").open("a") as f:
                f.write(json.dumps(sample) + "\n")
            if decision.startswith("stop:"):
                outcome = decision
                break
            if new_quota != quota:
                systemctl("set-property", "--runtime", unit, f"CPUQuota={new_quota}%")
                operation(f"{unit} 自动调整CPU配额{quota}%→{new_quota}%，CPU{sample['cpu_c']}°C/GPU{sample.get('gpu_c')}°C", output)
                quota = new_quota
            time.sleep(profile["sample_seconds"])
    except Exception as exc:
        outcome = "technical_failure:" + repr(exc)
    finally:
        try:
            systemctl("stop", unit, check=False)
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            result = dict(at=now(),outcome=outcome,exit_code=code,
                          wall_seconds=round(time.monotonic()-started,3),final_cpu_quota=quota)
            write_json(output / "result.json", result)
            operation(f"{unit} 结束：{outcome}，已释放进程组；已提交训练事务保留", output)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/training_resources.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
