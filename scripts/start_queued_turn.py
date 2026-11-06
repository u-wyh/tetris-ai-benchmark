"""Start existing queued input on an idle local thread, without a desktop client."""

import json
from pathlib import Path
import time

from codex_local_rpc import LocalRPC


def start_queued_turn(thread_id, folder, expected_cwd, permitted, client_factory=LocalRPC,
                      clock=time.time):
    """Busy threads are untouched; failed/quota-limited attempts wait one hour."""
    if not permitted():
        return dict(action='stopped')
    attempt = folder/'queued_start_last_attempt.json'
    if attempt.exists() and clock()-json.loads(attempt.read_text())['epoch'] < 3600:
        return dict(action='start_interval_not_elapsed')
    with client_factory() as rpc:
        params = dict(threadId=thread_id)
        goal = rpc.request('thread/goal/get', params)['goal']
        if not goal or goal['status'] != 'active':
            return dict(action='goal_not_active')
        queued = rpc.request('thread/queue/list', params)['data']
        if not queued:
            return dict(action='queue_empty')
        thread = rpc.request('thread/read', dict(params, includeTurns=False))['thread']

        def validate_thread(value):
            if value['id'] != thread_id or Path(value['cwd']).resolve() != expected_cwd.resolve():
                raise RuntimeError('Queued thread identity or workspace changed')
            if any(env.get('environmentId') != 'local' for env in value.get('environments') or []):
                raise RuntimeError('Refusing nonlocal execution environment')

        validate_thread(thread)
        state = thread['status']['type']
        if state == 'active':
            return dict(action='thread_busy', queued_submission_id=queued[0]['id'])
        if state not in ('idle', 'notLoaded'):
            return dict(action='thread_not_ready', thread_status=state)
        # Persist before a quota/network/start attempt, including failed attempts.
        temporary = attempt.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(epoch=clock(), queued_submission_id=queued[0]['id']))+'\n')
        temporary.replace(attempt)
        rates = rpc.request('account/rateLimits/read', dict(excludeResetCreditDetails=True))
        if rates.get('ordinaryUsageAllowed') is not True:
            return dict(action='quota_unavailable')
        goal = rpc.request('thread/goal/get', params)['goal']
        if not permitted() or not goal or goal['status'] != 'active':
            return dict(action='stopped')
        if state == 'notLoaded':
            thread = rpc.request('thread/resume', dict(params, excludeTurns=True))['thread']
            validate_thread(thread)
        # Resume may already have dispatched work. Never interrupt an active turn.
        thread = rpc.request('thread/read', dict(params, includeTurns=False))['thread']
        validate_thread(thread)
        if thread['status']['type'] != 'idle':
            return dict(action='thread_no_longer_idle')
        goal = rpc.request('thread/goal/get', params)['goal']
        if not permitted() or not goal or goal['status'] != 'active':
            return dict(action='stopped')
        result = rpc.request('thread/queue/start', dict(params, queuedSubmissionId=queued[0]['id']))
        turn = result['turn']
        return dict(action='queued_turn_start_accepted', queued_submission_id=queued[0]['id'],
                    turn_id=turn['id'], turn_status=turn['status'],
                    actual_development_execution_verified=False)
