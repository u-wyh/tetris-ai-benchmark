"""Maintain this machine's authorized backend; never start AI turns.

The hourly dispatcher remains solely responsible for quota reads and delivery.
No daemon restart is used: the native start command preserves a running backend.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess

from codex_local_rpc import LocalRPC
from development_allowed import allowed

REQUIRED_UNITS = ('tetris-codex-backend.service', 'tetris-development-hourly.timer',
                  'tetris-unattended-watchdog.timer')


def desktop_proxies(proc_root=Path('/proc')):
    pids = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry/'cmdline').read_bytes().split(b'\0')
            if (len(args) >= 3 and Path(os.fsdecode(args[0])).name == 'codex'
                    and args[1:3] == [b'app-server', b'proxy']):
                pids.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(pids)


def startup_readiness():
    result = subprocess.run(['systemctl', '--user', 'show', *REQUIRED_UNITS,
                             '-p', 'Id', '-p', 'ActiveState', '-p', 'UnitFileState'],
                            check=True, timeout=10, capture_output=True, text=True)
    services = {}
    for block in result.stdout.strip().split('\n\n'):
        fields = dict(line.split('=', 1) for line in block.splitlines() if '=' in line)
        if fields.get('Id') in REQUIRED_UNITS:
            services[fields.pop('Id')] = fields
    linger = subprocess.run(['loginctl', 'show-user', str(os.getuid()), '-p', 'Linger', '--value'],
                            check=True, timeout=10, capture_output=True, text=True).stdout.strip()
    ready = (linger == 'yes' and len(services) == len(REQUIRED_UNITS)
             and all(row.get('ActiveState') == 'active' and row.get('UnitFileState') == 'enabled'
                     for row in services.values()))
    return dict(ready=ready, linger=linger, units=services)


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
            with LocalRPC(timeout=35) as rpc:
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
            if permitted:
                startup = startup_readiness()
                event.update(startup=startup, desktop_proxy_pids=desktop_proxies())
                # A reachable backend alone does not prove unattended startup.
                if not startup['ready']:
                    event.update(action='unattended_setup_incomplete')
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
