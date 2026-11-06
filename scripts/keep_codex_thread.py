"""Keep the authorized local Codex thread subscribed without a desktop client.

Only restores the saved runtime and holds a connection. It never starts turns,
changes a goal, reads account quota or changes model/approval settings.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import threading
import time

from codex_local_rpc import LocalRPC

ROOT = Path(__file__).resolve().parents[1]
LIVE_GOALS = ('active', 'usageLimited', 'usage_limited')


def permitted(control_path, expected_thread):
    control = json.loads(control_path.read_text())
    return (control.get('enabled') is True and control['thread_id'] == expected_thread
            and not (control_path.parent/'STOP').exists()
            and control.get('execution_host') == 'remote-ssh-discovered:lightbook'
            and control.get('execution_machine_id') == Path('/etc/machine-id').read_text().strip())


def validate_thread(thread, thread_id, root):
    if thread['id'] != thread_id or Path(thread['cwd']).resolve() != root.resolve():
        raise RuntimeError('Thread identity or local workspace changed')
    if any(env.get('environmentId') != 'local' for env in thread.get('environments') or []):
        raise RuntimeError('Refusing a nonlocal execution environment')


def attach(rpc, thread_id, root, allowed):
    params = dict(threadId=thread_id)
    if not allowed():
        return None
    goal = rpc.request('thread/goal/get', params)['goal']
    if not goal or goal['status'] not in LIVE_GOALS:
        return None
    thread = rpc.request('thread/read', dict(params, includeTurns=False))['thread']
    validate_thread(thread, thread_id, root)
    # Recheck a pause or local STOP arriving while the metadata is read.
    goal = rpc.request('thread/goal/get', params)['goal']
    if not allowed() or not goal or goal['status'] not in LIVE_GOALS:
        return None
    thread = rpc.request('thread/resume', dict(params, excludeTurns=True))['thread']
    validate_thread(thread, thread_id, root)
    return thread


def run(control_path):
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    thread_id = json.loads(control_path.read_text())['thread_id']
    output = control_path.parent/'thread_connection.json'

    def write(action, **fields):
        value = dict(at=datetime.now().astimezone().isoformat(timespec='seconds'),
                     action=action, thread_id=thread_id, starts_ai_turns=False, **fields)
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(output)

    check = lambda: permitted(control_path, thread_id) and not stop.is_set()
    try:
        while check():
            with LocalRPC(timeout=35) as rpc:
                thread = attach(rpc, thread_id, ROOT, check)
                if thread is None:
                    write('goal_stopped', subscribed=False)
                else:
                    while check():
                        rpc.deadline = time.monotonic()+35
                        goal = rpc.request('thread/goal/get', dict(threadId=thread_id))['goal']
                        if not goal or goal['status'] not in LIVE_GOALS:
                            break
                        thread = rpc.request('thread/read', dict(threadId=thread_id, includeTurns=False))['thread']
                        validate_thread(thread, thread_id, ROOT)
                        if not check():
                            break
                        write('subscribed', subscribed=True, goal_status=goal['status'],
                              thread_status=thread['status'])
                        if stop.wait(30):
                            break
            write('disconnected', subscribed=False)
            if stop.wait(30):
                break
        write('stopped', subscribed=False)
    except Exception as exc:
        write('connection_failed', subscribed=False, error=type(exc).__name__)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    run(parser.parse_args().control.resolve())
