"""Maintain this machine's authorized backend; never start AI turns.

The hourly dispatcher remains solely responsible for quota reads and delivery.
No daemon restart is used: the native start command preserves a running backend.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import subprocess

from codex_local_rpc import LocalRPC
from development_allowed import allowed


def authorized(control, folder, machine_id):
    return (control.get('enabled') is True
            and control.get('execution_host') == 'remote-ssh-discovered:lightbook'
            and control.get('execution_machine_id') == machine_id
            and allowed(control['thread_id'], folder))


def run(control_path):
    control = json.loads(control_path.read_text())
    folder = control_path.parent
    event = dict(at=datetime.now().astimezone().isoformat(timespec='seconds'),
                 hostname=socket.gethostname(), action='disabled', ai_turn_started=False)
    code = 0
    try:
        machine_id = Path('/etc/machine-id').read_text().strip()
        if authorized(control, folder, machine_id):
            # Native start is idempotent and detaches from SSH/desktop lifetime.
            subprocess.run([control['codex_cli'], 'app-server', 'daemon', 'start'],
                           check=True, timeout=45, capture_output=True)
            with LocalRPC(timeout=10) as rpc:
                goal = rpc.request('thread/goal/get', dict(threadId=control['thread_id']))['goal']
            # Recheck a pause/control change arriving during the local health read.
            current = json.loads(control_path.read_text())
            permitted = (authorized(current, folder, machine_id)
                         and goal and goal['status'] in ('active', 'usageLimited', 'usage_limited'))
            power = subprocess.run(['systemctl', 'is-enabled', 'sleep.target', 'suspend.target',
                                    'hibernate.target', 'hybrid-sleep.target'],
                                   timeout=10, capture_output=True, text=True)
            sleep_blocked = power.stdout.splitlines() == ['masked'] * 4
            event.update(action='healthy' if permitted else 'stopped',
                         backend_rpc_verified=True, goal_status=goal['status'] if goal else None,
                         sleep_targets_masked=sleep_blocked)
            if permitted and not sleep_blocked:
                event.update(action='sleep_protection_missing')
                code = 1
    except Exception as exc:
        event.update(action='health_check_failed', error=type(exc).__name__)
        code = 1
    temporary = folder/'unattended_health.tmp'
    temporary.write_text(json.dumps(event, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(folder/'unattended_health.json')
    with (folder/'unattended_health.jsonl').open('a') as stream:
        stream.write(json.dumps(event, ensure_ascii=False)+'\n')
    print(json.dumps(event, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().control.resolve()))
