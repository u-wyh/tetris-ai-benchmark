"""Only live ordinary quota can resume a stale usage-limited goal."""

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from codex_local_rpc import recover_available_goal


GOAL = dict(status='usageLimited', objective='continue approved work', tokenBudget=5000, tokensUsed=123)


class FakeRPC:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def request(self, method, params):
        self.calls.append((method, params))
        return next(self.responses)


@pytest.mark.parametrize('status', ['paused', 'blocked', 'budgetLimited', 'complete'])
def test_user_pause_and_budget_states_never_query_or_resume(status):
    rpc = FakeRPC([dict(goal=dict(GOAL, status=status))])
    assert not recover_available_goal('thread', lambda: rpc)['ready']
    assert [call[0] for call in rpc.calls] == ['thread/goal/get']


@pytest.mark.parametrize('rates', [{'ordinaryUsageAllowed': False}, {}])
def test_missing_or_unavailable_live_quota_never_resumes(rates):
    rpc = FakeRPC([dict(goal=GOAL), rates])
    assert recover_available_goal('thread', lambda: rpc)['action'] == 'quota_unavailable'
    assert not any(method == 'thread/goal/set' for method, _ in rpc.calls)


def test_pause_arriving_during_the_quota_query_wins():
    rpc = FakeRPC([dict(goal=GOAL), dict(ordinaryUsageAllowed=True), dict(goal=dict(GOAL, status='paused'))])
    assert not recover_available_goal('thread', lambda: rpc)['ready']
    assert not any(method == 'thread/goal/set' for method, _ in rpc.calls)


def test_available_quota_resumes_status_without_resetting_usage_or_spending_credits():
    rpc = FakeRPC([dict(goal=GOAL), dict(ordinaryUsageAllowed=True),
                   dict(goal=copy.deepcopy(GOAL)), dict(goal=dict(GOAL, status='active'))])
    result = recover_available_goal('thread', lambda: rpc)
    assert result['ready'] and result['tokens_used'] == 123
    assert rpc.calls[-1] == ('thread/goal/set', dict(threadId='thread', status='active'))
    assert rpc.calls[1] == ('account/rateLimits/read', dict(excludeResetCreditDetails=True))
    assert all('consume' not in method.lower() and 'reset' not in method.lower() for method, _ in rpc.calls)
