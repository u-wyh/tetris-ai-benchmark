"""Read actual local development activity; never start or restart a job."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sqlite3
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def systemctl(*args):
    result = subprocess.run(['systemctl', '--user', *args], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'systemctl query failed')
    return result.stdout


def service_state(name):
    text = systemctl('show', name, '-p', 'Id,MainPID,ActiveState,SubState,Result,ControlGroup,NextElapseUSecRealtime,NextElapseUSecMonotonic,LastTriggerUSec')
    return dict(line.split('=', 1) for line in text.splitlines() if '=' in line)


def active_units():
    text = systemctl('list-units', '--type=service', '--state=active,activating,deactivating',
                     '--plain', '--no-legend', '--no-pager', 'tetris-*.service')
    return [line.split()[0] for line in text.splitlines() if line.strip()]


def last_sample(path):
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(0, 2)
        offset = max(0, stream.tell()-16384)
        stream.seek(offset)
        lines = stream.read().splitlines()
    if offset:
        lines = lines[1:]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # A writer can currently be appending the last line.
    return {}


def argument(args, flag):
    try:
        return args[args.index(flag)+1]
    except (ValueError, IndexError):
        return None


def artifact_path(root, value):
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return None
    path = path.resolve()
    return path if path.is_relative_to((root/'runs').resolve()) else None


def inspect_service(root, name, state, proc_root=Path('/proc'), current_time=None):
    current_time = current_time or datetime.now().astimezone()
    pid = int(state.get('MainPID') or 0)
    result = dict(unit=name, main_pid=pid, active_state=state.get('ActiveState'),
                  process_live=False, recorded_stage=None)
    if pid and state.get('ActiveState') in ('active','activating','deactivating'):
        try:
            args = (proc_root/str(pid)/'cmdline').read_bytes().decode(errors='replace').split('\0')
            result['process_live'] = bool(args[0])
        except (FileNotFoundError, ProcessLookupError):
            args = []
    else:
        args = []
    # Do not expose full argv: future commands might contain private arguments.
    module = argument(args, '-m') or ''
    training = module == 'training.train_vector_transaction'
    evaluation = module.startswith('training.evaluation.')
    heartbeat = artifact_path(root, argument(args, '--heartbeat'))
    directory = artifact_path(root, argument(args, '--run-dir'))
    result['activity'] = ('training' if training else 'evaluation' if evaluation
                          else 'guarded_work' if heartbeat else 'controller')
    if directory:
        result['run_dir'] = str(directory)
        status = read_json(directory/('transaction_state.json' if training else 'status.json'))
        result['recorded_stage'] = status.get('stage', status.get('status'))
        if training:
            config = read_json(directory/'config.json')
            result.update(committed_steps=status.get('committed_steps'), device=config.get('device'),
                          n_envs=config.get('ppo',{}).get('n_envs'))
    if heartbeat:
        folder = heartbeat.parent
        launch = read_json(folder/'launch.json')
        sample = last_sample(folder/'samples.jsonl')
        result['guard_dir'] = str(folder)
        result['latest_sample'] = sample
        try:
            age = (current_time-datetime.fromisoformat(sample['at'])).total_seconds()
            result['sample_age_seconds'] = round(age, 3)
            result['sample_fresh'] = 0 <= age <= max(15, launch.get('profile',{}).get('sample_seconds',2)*4)
        except (KeyError, TypeError, ValueError):
            result['sample_fresh'] = False
    return result


def kernel_budget(group, cgroup_root=Path('/sys/fs/cgroup')):
    if not group or not group.startswith('/') or '..' in Path(group).parts:
        return dict(available=False)
    path = cgroup_root/group.lstrip('/')
    try:
        quota, period = (path/'cpu.max').read_text().split()
        result = dict(available=True, cpu_quota_percent=None if quota=='max' else 100*int(quota)/int(period))
        for filename, field in [('memory.current','memory_current_mib'),('memory.max','memory_max_mib'),
                                ('memory.swap.max','memory_swap_max_mib')]:
            value = (path/filename).read_text().strip()
            result[field] = None if value=='max' else int(value)/1024**2
        return result
    except (OSError, ValueError):
        return dict(available=False)


def goal_state(codex_dir, thread):
    try:
        with sqlite3.connect(f'file:{codex_dir/"goals_1.sqlite"}?mode=ro', uri=True) as db:
            row = db.execute('SELECT status FROM thread_goals WHERE thread_id=?',(thread,)).fetchone()
        return row[0] if row else 'unknown'
    except sqlite3.Error:
        return 'unknown'


def observe(root=ROOT, codex_dir=Path.home()/'.codex', show=None, units=None, proc_root=Path('/proc'),
            cgroup_root=Path('/sys/fs/cgroup'), current_time=None):
    show, units = show or service_state, units or active_units
    current_time = current_time or datetime.now().astimezone()
    control = read_json(root/'runs/autonomous_development_20261105/control.json')
    recorded = read_json(root/'runs/autonomous_development_20261105/development_status.json')
    services = []
    for name in units():
        try:
            services.append(inspect_service(root, name, show(name), proc_root, current_time))
        except (OSError, ValueError, RuntimeError) as exc:
            services.append(dict(unit=name, process_live=False, observation_error=str(exc)))
    live = [r for r in services if r.get('process_live')]
    activity = ('training' if any(r['activity']=='training' for r in live) else
                'evaluation' if any(r['activity']=='evaluation' for r in live) else
                'waiting_or_starting' if live else 'no_active_compute')
    result = dict(at=current_time.isoformat(timespec='seconds'), hostname=socket.gethostname(),
                  configured_execution_host=recorded.get('execution_host'), observed_activity=activity,
                  recorded_global_stage=recorded.get('stage'), services=services,
                  goal_status=goal_state(codex_dir,control.get('thread_id')),
                  account_quota='not_queried', automatic_actions='none')
    for name, key in [('tetris-compute.slice','combined_compute_budget'),
                      ('tetris-development-hourly.timer','hourly_timer')]:
        try:
            state=show(name)
            result[key] = kernel_budget(state.get('ControlGroup'),cgroup_root) if name.endswith('.slice') else state
        except RuntimeError as exc:
            result[key] = dict(observation_error=str(exc))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    print(json.dumps(observe(args.root.resolve()),ensure_ascii=False,indent=2))
