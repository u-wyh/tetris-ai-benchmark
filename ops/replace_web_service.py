"""Warm a replacement before a brief port handoff; retain/restore the old command."""
import argparse
import functools
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=('web', 'api'), required=True)
    p.add_argument('--old-pid', type=int, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    if args.host in ('0.0.0.0', '::'):
        p.error('Use the trusted service address')
    port = 8080 if args.kind == 'web' else 8081
    cmd = Path(f'/proc/{args.old_pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
    cwd = os.readlink(f'/proc/{args.old_pid}/cwd')
    expected = 'http.server' if args.kind == 'web' else 'training.web_inference'
    if expected not in cmd or str(port) not in cmd:
        raise RuntimeError('Old PID is not the expected existing service; no process stopped')
    args.run.mkdir(parents=True, exist_ok=True)
    from training.evaluation.web_longrun_verify import atomic
    record = dict(kind=args.kind, host=args.host, port=port, old_pid=args.old_pid,
                  old_command=cmd, old_cwd=cwd, new_pid=os.getpid())
    target = args.run / f'{args.kind}-handoff.json'
    atomic(target, json.dumps(record).encode())
    if args.kind == 'api':
        from training.web_inference import Inference, make_handler
        handler = make_handler(Inference(), f'http://{args.host}:8080')
    else:
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(ROOT/'tetris-game'))
    server = ThreadingHTTPServer((args.host, port), handler, bind_and_activate=False)
    record['ready_to_replace_at'] = time.time()
    atomic(target, json.dumps(record).encode())
    #All imports, model identity checks and model loads finish before stopping the old PID.
    os.kill(args.old_pid, signal.SIGTERM)
    started = time.monotonic()
    try:
        while True:
            try:
                server.server_bind()
                server.server_activate()
                break
            except OSError:
                if time.monotonic()-started > 5:
                    raise
                time.sleep(.025)
    except Exception:
        server.server_close()
        restored = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
        record.update(rollback_pid=restored.pid, result='rollback')
        atomic(target, json.dumps(record).encode())
        raise
    record.update(bind_gap_ms=1000*(time.monotonic()-started), result='serving', ready_at=time.time())
    atomic(target, json.dumps(record).encode())
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(record), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
