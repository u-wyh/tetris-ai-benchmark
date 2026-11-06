import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from keep_codex_thread import attach


class RPC:
    def __init__(self, root, goal='active', pause_during_read=False, environment='local'):
        self.goal, self.pause_during_read = goal, pause_during_read
        self.thread = dict(id='target', cwd=str(root), environments=[dict(environmentId=environment)])
        self.calls = []
    def request(self, method, params):
        self.calls.append((method, params))
        if method == 'thread/goal/get':
            return dict(goal=dict(status=self.goal))
        if method == 'thread/read':
            if self.pause_during_read: self.goal='paused'
            return dict(thread=copy.deepcopy(self.thread))
        assert method == 'thread/resume'
        assert params == dict(threadId='target', excludeTurns=True)
        return dict(thread=copy.deepcopy(self.thread))


@pytest.mark.parametrize('goal', ['active','usageLimited','usage_limited'])
def test_persistent_subscription_preserves_saved_settings_without_starting_or_resuming_goal(tmp_path, goal):
    rpc=RPC(tmp_path,goal)
    assert attach(rpc,'target',tmp_path,lambda:True)['id']=='target'
    assert [m for m,_ in rpc.calls]==['thread/goal/get','thread/read','thread/goal/get','thread/resume']


@pytest.mark.parametrize('goal', ['paused','complete','blocked','budgetLimited'])
def test_stopped_goals_are_not_attached_or_modified(tmp_path, goal):
    rpc=RPC(tmp_path,goal)
    assert attach(rpc,'target',tmp_path,lambda:True) is None
    assert [m for m,_ in rpc.calls]==['thread/goal/get']


def test_pause_arriving_during_read_wins(tmp_path):
    rpc=RPC(tmp_path,pause_during_read=True)
    assert attach(rpc,'target',tmp_path,lambda:True) is None
    assert not any(m=='thread/resume' for m,_ in rpc.calls)


def test_stop_arriving_during_read_wins(tmp_path):
    rpc=RPC(tmp_path)
    assert attach(rpc,'target',tmp_path,lambda:len(rpc.calls)<3) is None
    assert not any(m=='thread/resume' for m,_ in rpc.calls)


@pytest.mark.parametrize('environment', ['remote-ssh-discovered:ubuntu','remote-ssh-discovered:lightbook'])
def test_nonlocal_runtime_cannot_be_attached(tmp_path, environment):
    rpc=RPC(tmp_path,environment=environment)
    with pytest.raises(RuntimeError,match='nonlocal'):
        attach(rpc,'target',tmp_path,lambda:True)
    assert not any(m=='thread/resume' for m,_ in rpc.calls)


def test_wrong_workspace_never_attaches(tmp_path):
    rpc=RPC(tmp_path/'other')
    with pytest.raises(RuntimeError,match='workspace'):
        attach(rpc,'target',tmp_path,lambda:True)
