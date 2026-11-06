"""Headless queue dispatch: preserve pause, quota, identity and active turns."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from start_queued_turn import start_queued_turn


class RPC:
    def __init__(self, cwd, state='idle', quota=True, pause_at=None):
        self.cwd, self.state, self.quota, self.pause_at = cwd, state, quota, pause_at
        self.calls = []
        self.goal = 'active'
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def request(self, method, params):
        self.calls.append((method, params))
        if method == self.pause_at:
            self.goal = 'paused'
        if method == 'thread/goal/get': return dict(goal=dict(status=self.goal))
        if method == 'thread/queue/list': return dict(data=[dict(id='existing')])
        if method == 'account/rateLimits/read': return dict(ordinaryUsageAllowed=self.quota)
        if method == 'thread/resume': self.state = 'idle'
        if method in ('thread/read', 'thread/resume'):
            return dict(thread=dict(id='target', cwd=str(self.cwd), status=dict(type=self.state),
                                    environments=[dict(environmentId='local')]))
        if method == 'thread/queue/start':
            assert params == dict(threadId='target', queuedSubmissionId='existing')
            return dict(turn=dict(id='new-turn', status='inProgress'))
        raise AssertionError(method)


def start(folder, rpc, clock=4000, permitted=lambda: True):
    return start_queued_turn('target', folder, folder, permitted, lambda: rpc, lambda: clock)


@pytest.mark.parametrize('state', ['idle', 'notLoaded'])
def test_existing_queue_starts_without_desktop_and_without_new_message(tmp_path, state):
    rpc = RPC(tmp_path, state)
    result = start(tmp_path, rpc)
    assert result['action'] == 'queued_turn_start_accepted'
    assert result['actual_development_execution_verified'] is False
    assert ('thread/resume' in [m for m, _ in rpc.calls]) == (state == 'notLoaded')
    assert not any(m in ('turn/start','thread/queue/add','thread/goal/set') for m, _ in rpc.calls)
    before = copy.deepcopy(rpc.calls)
    assert start(tmp_path, rpc, clock=4001)['action'] == 'start_interval_not_elapsed'
    assert rpc.calls == before


def test_active_thread_is_not_interrupted_and_does_not_consume_retry(tmp_path):
    rpc = RPC(tmp_path, 'active')
    assert start(tmp_path, rpc)['action'] == 'thread_busy'
    assert not (tmp_path/'queued_start_last_attempt.json').exists()
    assert not any(m in ('account/rateLimits/read','thread/queue/start') for m, _ in rpc.calls)


@pytest.mark.parametrize('quota', [False, None])
def test_quota_denial_never_resumes_or_starts_and_waits_hour(tmp_path, quota):
    rpc = RPC(tmp_path, 'notLoaded', quota)
    assert start(tmp_path, rpc)['action'] == 'quota_unavailable'
    assert start(tmp_path, rpc, clock=7599)['action'] == 'start_interval_not_elapsed'
    assert start(tmp_path, rpc, clock=7600)['action'] == 'quota_unavailable'
    assert sum(m == 'account/rateLimits/read' for m, _ in rpc.calls) == 2
    assert not any(m in ('thread/resume','thread/queue/start') for m, _ in rpc.calls)


@pytest.mark.parametrize('pause_at', ['account/rateLimits/read','thread/resume'])
def test_user_pause_during_quota_read_or_resume_wins(tmp_path, pause_at):
    rpc = RPC(tmp_path, 'notLoaded', pause_at=pause_at)
    assert start(tmp_path, rpc)['action'] == 'stopped'
    assert not any(m == 'thread/queue/start' for m, _ in rpc.calls)


def test_control_stop_during_quota_read_wins(tmp_path):
    rpc = RPC(tmp_path)
    assert start(tmp_path, rpc, permitted=lambda: not any(m=='account/rateLimits/read'
        for m, _ in rpc.calls))['action'] == 'stopped'
    assert not any(m == 'thread/queue/start' for m, _ in rpc.calls)


def test_wrong_workspace_refuses_before_quota_or_resume(tmp_path):
    rpc = RPC(tmp_path/'other', 'notLoaded')
    with pytest.raises(RuntimeError, match='workspace'):
        start(tmp_path, rpc)
    assert not any(m in ('account/rateLimits/read','thread/resume') for m, _ in rpc.calls)


def test_transport_failure_is_not_retried_within_hour(tmp_path):
    rpc = RPC(tmp_path)
    original = rpc.request
    def failing(method, params):
        if method == 'thread/queue/start': raise TimeoutError('transport')
        return original(method, params)
    rpc.request = failing
    with pytest.raises(TimeoutError): start(tmp_path, rpc)
    assert start(tmp_path, rpc, clock=4001)['action'] == 'start_interval_not_elapsed'


def test_resume_racing_with_automatic_turn_does_not_start_twice(tmp_path):
    rpc = RPC(tmp_path, 'notLoaded')
    original = rpc.request
    def racing(method, params):
        result = original(method, params)
        if method == 'thread/resume': rpc.state = 'active'
        return result
    rpc.request = racing
    assert start(tmp_path, rpc)['action'] == 'thread_no_longer_idle'
    assert not any(m == 'thread/queue/start' for m, _ in rpc.calls)
