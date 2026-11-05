"""Fail-closed systemd ExecCondition for this user's authorized local development."""

import argparse
from pathlib import Path
import sqlite3


def allowed(thread_id, run_dir=None, codex_dir=None):
    if run_dir is not None and (Path(run_dir)/'STOP').exists():
        return False
    directory = Path(codex_dir) if codex_dir else Path.home()/'.codex'
    try:
        with sqlite3.connect(f'file:{directory/"goals_1.sqlite"}?mode=ro',uri=True) as db:
            row = db.execute('SELECT status FROM thread_goals WHERE thread_id=?',(thread_id,)).fetchone()
    except (sqlite3.Error, OSError):
        return False
    return bool(row and row[0] in ('active','usageLimited','usage_limited'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thread-id',required=True)
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    raise SystemExit(0 if allowed(args.thread_id,args.run_dir) else 1)
