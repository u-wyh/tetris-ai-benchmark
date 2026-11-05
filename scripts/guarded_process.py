"""Stop a resource-guarded command if its external temperature monitor disappears."""

import argparse
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import time


def heartbeat_ok(path, guardian_pid, timeout):
    try:
        value=json.loads(path.read_text())
        age=time.monotonic()-value['monotonic']
        return value['pid']==guardian_pid and math.isfinite(age) and 0<=age<=timeout
    except (OSError,ValueError,KeyError,TypeError):
        return False


def run(args):
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or args.timeout<=0:
        raise ValueError('A command and positive watchdog timeout are required')
    # pidfd remains attached to the original process even if its numeric PID is reused.
    descriptor=os.pidfd_open(args.guardian_pid)
    child=None
    stopped=False
    reason='command_completed'
    code=1

    def interrupted(signum, frame):
        nonlocal stopped
        stopped=True

    for signum in (signal.SIGTERM,signal.SIGINT):
        signal.signal(signum,interrupted)
    try:
        if not heartbeat_ok(args.heartbeat,args.guardian_pid,args.timeout):
            reason='missing_or_stale_heartbeat'
            return 1
        child=subprocess.Popen(command,start_new_session=True)
        while child.poll() is None:
            if stopped:
                reason='service_stopping'
                break
            if select.select([descriptor],[],[],0.5)[0]:
                reason='guardian_exited'
                break
            if not heartbeat_ok(args.heartbeat,args.guardian_pid,args.timeout):
                reason='missing_or_stale_heartbeat'
                break
        else:
            code=child.returncode
        return code
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        os.close(descriptor)
        result=dict(at_epoch=time.time(),reason=reason,exit_code=code,
                    child_pid=child.pid if child is not None else None)
        destination=args.heartbeat.parent/'watchdog_result.json'
        temporary=destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guardian-pid',type=int,required=True)
    parser.add_argument('--heartbeat',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=30)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    raise SystemExit(run(parser.parse_args()))
