"""Bounded desktop-proxy pause test; systemd ExecStopPost always restores it.

This tests execution while desktop communication is paused. It does not test
powering off the other computer or rebooting this machine.
"""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from development_allowed import allowed
from unattended_watchdog import desktop_proxies


def stamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def identity(pid, proc=Path('/proc')):
    entry = proc/str(pid)
    fields = (entry/'stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19], parent=int(fields[1]), state=fields[0])


def send_proxy(record, sig):
    pid = record['pid']
    fd = os.pidfd_open(pid)
    try:
        current = identity(pid)
        if (current['start_ticks'] != record['start_ticks']
                or pid not in desktop_proxies()
                or Path(f'/proc/{pid}').stat().st_uid != os.getuid()):
            raise RuntimeError('Proxy identity changed; refusing signal')
        signal.pidfd_send_signal(fd, sig)
    finally:
        os.close(fd)


def restore(folder):
    path = folder/'state.json'
    if not path.exists():
        return
    state = json.loads(path.read_text())
    if state.get('restored_at'):
        return
    errors = []
    for record in state['proxies']:
        try:
            send_proxy(record, signal.SIGCONT)
        except (ProcessLookupError, FileNotFoundError):
            pass  # An exited proxy cannot remain suspended.
        except Exception as exc:
            errors.append(dict(pid=record['pid'], error=repr(exc)))
    if errors:
        write(folder/'restore_failure.json', dict(at=stamp(), errors=errors))
        raise RuntimeError('Proxy restoration needs inspection')
    state['restored_at'] = stamp()
    write(path, state)


def observe(folder):
    state = json.loads((folder/'state.json').read_text())
    if state.get('restored_at') or state['phase'] != 'proxy_paused':
        raise RuntimeError('No live paused-proxy test')
    proxies = [identity(row['pid']) for row in state['proxies']]
    if (desktop_proxies() != sorted(row['pid'] for row in proxies)
            or any(row['state'] not in ('T', 't') or row['start_ticks'] != saved['start_ticks']
                   for row, saved in zip(proxies, state['proxies']))):
        raise RuntimeError('Desktop transport was not fully paused')
    ancestors, pid = [], os.getpid()
    while pid > 0:
        row = identity(pid)
        ancestors.append(row)
        pid = row['parent']
    backend = state['backend']
    if not any(row['pid'] == backend['pid'] and row['start_ticks'] == backend['start_ticks']
               for row in ancestors):
        raise RuntimeError('Observation did not execute under the pinned native backend')
    write(folder/'observation.json', dict(at=stamp(), executed_while_all_proxies_paused=True,
          proxies=proxies, tool_process_ancestry=ancestors,
          scope='New AI tool invocation executed under native backend while desktop proxies stopped'))


def run(folder):
    lock = (folder/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder/'state.json').exists():
        raise RuntimeError('Preserve prior test; do not repeat')
    design = json.loads((folder/'design.json').read_text())
    control_path = Path(design['control'])
    control = json.loads(control_path.read_text())
    if (control.get('enabled') is not True
            or control.get('execution_host') != 'remote-ssh-discovered:lightbook'
            or control.get('execution_machine_id') != Path('/etc/machine-id').read_text().strip()
            or not allowed(control['thread_id'], control_path.parent)
            or not 1 <= design['seconds'] <= 30):
        raise RuntimeError('Unattended probe is not authorized on this host')
    backend = identity(design['backend']['pid'])
    if backend['start_ticks'] != design['backend']['start_ticks'] or backend['parent'] != 1:
        raise RuntimeError('Native backend identity or detached parent changed')
    pids = desktop_proxies()
    if not pids or os.getpid() in pids or backend['pid'] in pids:
        raise RuntimeError('No distinct desktop proxy to test')
    state = dict(at=stamp(), phase='prepared', backend=backend,
                 proxies=[identity(pid) for pid in pids])
    if any(row['state'] in ('T', 't') for row in state['proxies']):
        raise RuntimeError('A desktop proxy was already suspended before the test')
    write(folder/'state.json', state)  # Persist recovery identities before SIGSTOP.
    def stop(*args):
        raise InterruptedError('Probe interrupted')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    outcome = 'interrupted'
    try:
        for row in state['proxies']:
            send_proxy(row, signal.SIGSTOP)
        state.update(at=stamp(), phase='proxy_paused')
        write(folder/'state.json', state)
        deadline = time.monotonic() + design['seconds']
        while time.monotonic() < deadline:
            if not allowed(control['thread_id'], control_path.parent):
                raise InterruptedError('Goal stopped')
            time.sleep(.2)
        outcome = 'observed' if (folder/'observation.json').exists() else 'no_observation'
    finally:
        restore(folder)
        write(folder/'summary.json', dict(at=stamp(), outcome=outcome,
              desktop_proxy_restored=True, desktop_disconnect_tested=False, reboot_tested=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'observe', 'restore'])
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.run_dir.resolve())
