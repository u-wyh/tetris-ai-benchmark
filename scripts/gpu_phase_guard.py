"""Require a live resource_guard, watchdog and kernel limits before CUDA capture."""
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess

from own_policy_capture import sha
from guarded_process import heartbeat_ok

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT/'runs/autonomous_development_20261105/control.json'
MACHINE = '6b1b552131804ebc918f30005b4d0204'


def require_authorization(job, attempt):
    control = json.loads(CONTROL.read_text())
    if (control.get('enabled') is not True or control.get('execution_machine_id') != MACHINE or
            Path('/etc/machine-id').read_text().strip() != MACHINE or
            job.get('thread_id') != control['thread_id'] or job.get('control') != str(CONTROL) or
            any((p/'STOP').exists() for p in [CONTROL.parent, Path(attempt), Path(job['job_folder'])])):
        raise RuntimeError('Lightbook authorization, host pin or STOP blocks capture')
    with sqlite3.connect(f'file:{Path.home()/".codex/goals_1.sqlite"}?mode=ro', uri=True) as db:
        row = db.execute('SELECT status FROM thread_goals WHERE thread_id=?', (control['thread_id'],)).fetchone()
    if row is None or row[0] != 'active':
        raise RuntimeError('Goal paused, cancelled or usage limited; preserve progress')


def process_args(pid):
    return (Path('/proc')/str(pid)/'cmdline').read_bytes().decode().rstrip('\0').split('\0')


def argument(args, name):
    if args.count(name) != 1:
        raise ValueError('Missing or ambiguous guardian argument: '+name)
    return args[args.index(name)+1]


def duration_seconds(value):
    parts = re.findall(r'([0-9]+(?:\.[0-9]+)?)\s*(us|ms|min|s|h)', value)
    if not parts or re.sub(r'([0-9]+(?:\.[0-9]+)?)\s*(us|ms|min|s|h)\s*', '', value):
        raise ValueError('Unbounded or unknown systemd runtime limit')
    factors = dict(us=1e-6, ms=.001, s=1, min=60, h=3600)
    return sum(float(number)*factors[unit] for number, unit in parts)


def verify_live_guard(request, attempt, job):
    require_authorization(job, attempt)
    attempt = Path(attempt).resolve()
    controller_path = Path(request['controller_design']).resolve()
    controller = json.loads(controller_path.read_text())
    guard = Path(request['guard_dir']).resolve()
    if guard != controller_path.parent/'guard' or (guard/'result.json').exists():
        raise RuntimeError('A live controller guard is required')
    launch = json.loads((guard/'launch.json').read_text())
    expected_command = [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/run_gpu_evaluation_phase.py'),
                        '--attempt-dir', str(attempt)]
    if (launch['command'] != expected_command or controller['validation_command'] != expected_command or
            controller.get('thread_id') != job['thread_id'] or controller.get('control') != str(CONTROL) or
            not 0 < launch['seconds'] <= controller['command_seconds'] <= 180):
        raise RuntimeError('Guard command, authorization or command time budget differs')
    required = [attempt/'request.json', controller_path.parent/'profile.json',
        ROOT/'configs/training_resources.json', ROOT/'scripts/run_gpu_evaluation_phase.py',
        ROOT/'scripts/gpu_phase_guard.py', ROOT/'scripts/inspect_gpu_evaluation.py',
        ROOT/'scripts/resource_guard.py', ROOT/'scripts/guarded_process.py', Path(job['job_folder'])/'design.json']
    if any(str(p) not in controller['frozen_files'] for p in required):
        raise RuntimeError('Controller must freeze the request, job, profile and guard/launcher sources')
    for path, digest in controller['frozen_files'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen phase/controller input changed: '+path)
    profile = json.loads((ROOT/'configs/training_resources.json').read_text())
    if (launch['profile'] != profile or
            json.loads((controller_path.parent/'profile.json').read_text()) != profile):
        raise RuntimeError('Phase requires the original CPU250%/6GiB/thermal GPU profile')
    unit = launch['unit']+'.service'
    raw = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'ControlGroup',
        '-p', 'MainPID', '-p', 'KillMode', '-p', 'OOMPolicy', '-p', 'RuntimeMaxUSec'],
        text=True, capture_output=True, check=True, timeout=5).stdout
    state = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    group = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    if (state['ActiveState'] != 'active' or state['ControlGroup'] != group or
            not group.endswith('/'+unit) or state['KillMode'] != 'control-group' or state['OOMPolicy'] != 'stop' or
            not 0 < duration_seconds(state['RuntimeMaxUSec']) <= launch['seconds']):
        raise RuntimeError('Current process is not in the matching live bounded guard service')
    cgroup = Path('/sys/fs/cgroup')/group.lstrip('/')
    quota, period = (cgroup/'cpu.max').read_text().split()
    memory, swap = (cgroup/'memory.max').read_text().strip(), (cgroup/'memory.swap.max').read_text().strip()
    if (quota == 'max' or not 0 < int(quota)/int(period) <= 2.5 or memory == 'max' or
            not 0 < int(memory) <= 6*1024**3 or swap != '0' or
            set(os.sched_getaffinity(0)) != set(launch['physical_cpus']) or len(launch['physical_cpus']) != 6):
        raise RuntimeError('Kernel CPU/memory/swap/affinity limits do not match protected execution')
    watchdog_pid = int(state['MainPID'])
    ancestor = os.getpid()
    for _ in range(16):
        if ancestor == watchdog_pid:
            break
        stat = (Path('/proc')/str(ancestor)/'stat').read_text().rsplit(')', 1)[1].split()
        ancestor = int(stat[1])
        if ancestor <= 1:
            raise RuntimeError('Current process is not a watchdog descendant')
    else:
        raise RuntimeError('Cannot establish watchdog ancestry')
    args = process_args(watchdog_pid)
    if (str(ROOT/'scripts/guarded_process.py') not in args or
            Path(argument(args, '--heartbeat')).resolve() != guard/'heartbeat.json'):
        raise RuntimeError('Kernel service does not own the expected heartbeat watchdog')
    guardian_pid = int(argument(args, '--guardian-pid'))
    guardian = process_args(guardian_pid)
    if (str(ROOT/'scripts/resource_guard.py') not in guardian or
            Path(argument(guardian, '--output')).resolve() != guard or
            Path(argument(guardian, '--profile')).resolve() != controller_path.parent/'profile.json' or
            not heartbeat_ok(guard/'heartbeat.json', guardian_pid, 10)):
        raise RuntimeError('Temperature guardian missing or heartbeat stale')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; preserve progress without CPU fallback')
    return dict(unit=unit, cgroup=group, watchdog_pid=watchdog_pid, guardian_pid=guardian_pid,
                cpu_cores_limit=int(quota)/int(period), memory_max_bytes=int(memory), memory_swap_bytes=0,
                physical_cpus=launch['physical_cpus'], command_seconds=launch['seconds'], cuda_available=True)
