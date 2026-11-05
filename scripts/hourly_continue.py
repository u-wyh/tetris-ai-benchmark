"""Queue one authorized continuation through the local Codex CLI, at most hourly.

The native automation tool is unavailable in this chat. A user systemd timer runs
this helper; no API keys, model overrides, or account-limit bypasses are used.
Read-only queue/goal inspection matches codex-cli 0.160.1's local SQLite schema.
"""

import argparse
import fcntl
import json
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def decision(enabled, goal_status, pending):
    if not enabled:
        return "disabled"
    if goal_status in ("paused", "complete", "completed", "blocked", "budgetLimited", "budget_limited"):
        return "goal_" + goal_status
    if goal_status is None:
        return "no_active_goal"
    if goal_status not in ("active", "usageLimited", "usage_limited"):
        return "unsupported_goal_status"
    if pending:
        return "already_queued"
    return "queue"


def inspect(codex_dir, thread):
    with sqlite3.connect(f"file:{codex_dir/'goals_1.sqlite'}?mode=ro",uri=True) as db:
        row=db.execute("SELECT status FROM thread_goals WHERE thread_id=?",(thread,)).fetchone()
    with sqlite3.connect(f"file:{codex_dir/'queue_1.sqlite'}?mode=ro",uri=True) as db:
        pending=db.execute("SELECT count(*) FROM queued_items WHERE thread_id=?",(thread,)).fetchone()[0]
    return row[0] if row else None,pending


def recover_goal(thread):
    from codex_local_rpc import recover_available_goal
    return recover_available_goal(thread)


def run(control_path, dry_run=False, clock=time.time):
    control=json.loads(control_path.read_text())
    folder=control_path.parent
    with (folder/'hourly.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        event=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),dry_run=dry_run)
        code=0
        try:
            if control.get('execution_host') and (
                    control['execution_host'] != 'remote-ssh-discovered:lightbook'
                    or control.get('execution_machine_id') != Path('/etc/machine-id').read_text().strip()):
                raise RuntimeError('Execution host identity does not match authorized lightbook')
            status,pending=inspect(Path.home()/'.codex',control['thread_id'])
            action=decision(control['enabled'] and not (folder/'STOP').exists(),status,pending)
            event.update(goal_status=status,pending_messages=pending,action=action)
            interval=control.get('interval_seconds',3600)
            if interval<3600:
                raise ValueError('Hourly continuation requires at least 3600 seconds between attempts')
            attempt_file=folder/'hourly_last_attempt.json'
            last=(json.loads(attempt_file.read_text())['epoch'] if attempt_file.exists()
                  else datetime.fromisoformat(control['created_at']).timestamp())
            # A pending message must not prevent recovery of a usage-limited
            # goal: otherwise that message could remain blocked indefinitely.
            check_needed=(action=='queue' or (action=='already_queued' and
                          status in ('usageLimited','usage_limited')))
            if check_needed and clock()-last<interval:
                action='interval_not_elapsed'
                event['action']=action
            if check_needed and action!='interval_not_elapsed' and not dry_run:
                # Record before delivery: even a failed or interrupted attempt waits an hour.
                temporary=attempt_file.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(epoch=clock()))+'\n')
                temporary.replace(attempt_file)
                if status in ('usageLimited','usage_limited'):
                    recovery=recover_goal(control['thread_id'])
                    event['quota_recovery']=recovery
                    if not recovery['ready']:
                        action=recovery['action']
                        event['action']=action
                    else:
                        status,pending=inspect(Path.home()/'.codex',control['thread_id'])
                        action=decision(control['enabled'],status,pending)
                        event.update(goal_status=status,pending_messages=pending,action=action)
                        if recovery['action']=='quota_available_goal_resumed':
                            with (ROOT/'OPERATIONS.md').open('a') as f:
                                f.write(f"\n- {event['at']}：实时账户允许正常使用，恢复原目标的旧usageLimited状态，保留目标、预算和用量；未消耗额度重置券。证据：{folder/'hourly.jsonl'}。\n")
            if action=='queue' and not dry_run:
                result=subprocess.run([control['codex_cli'],'queue','--remote','unix://','--thread',control['thread_id'],
                                       '--message',control['prompt']],cwd=ROOT,
                                      capture_output=True,text=True,timeout=30)
                event.update(returncode=result.returncode,acknowledgement=result.stdout.strip()[:300])
                if result.returncode:
                    event['action']='queue_failed'
                    code=1
                else:
                    with (ROOT/'OPERATIONS.md').open('a') as f:
                        f.write(f"\n- {event['at']}：按用户授权每小时续办检查，已向原聊天排队发送继续开发请求。只确认投递，不代表额度已恢复或工作已完成。证据：{folder/'hourly.jsonl'}。\n")
        except Exception as exc:
            event.update(action='check_failed',error=repr(exc))
            code=1
        with (folder/'hourly.jsonl').open('a') as f:
            f.write(json.dumps(event,ensure_ascii=False)+'\n')
        print(json.dumps(event,ensure_ascii=False),flush=True)
        return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    raise SystemExit(run(args.control.resolve(),args.dry_run))
