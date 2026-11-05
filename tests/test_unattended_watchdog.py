import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
spec = importlib.util.spec_from_file_location('unattended_watchdog',
    Path(__file__).resolve().parents[1]/'scripts/unattended_watchdog.py')
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


def test_machine_pin_and_stop_are_required(monkeypatch, tmp_path):
    control = dict(enabled=True, execution_host='remote-ssh-discovered:lightbook',
                   execution_machine_id='this-machine', thread_id='target')
    monkeypatch.setattr(watchdog, 'allowed', lambda *args: True)
    assert watchdog.authorized(control, tmp_path, 'this-machine')
    assert not watchdog.authorized(control, tmp_path, 'other-machine')
    assert not watchdog.authorized(dict(control, enabled=False), tmp_path, 'this-machine')
    monkeypatch.setattr(watchdog, 'allowed', lambda *args: False)
    assert not watchdog.authorized(control, tmp_path, 'this-machine')


def test_pause_during_rpc_never_queues_or_restarts_backend(tmp_path, monkeypatch):
    path = tmp_path/'control.json'
    path.write_text(json.dumps(dict(thread_id='target', codex_cli='codex')))
    checks = iter([True, False])
    monkeypatch.setattr(watchdog, 'authorized', lambda *args: next(checks))
    calls = []
    monkeypatch.setattr(watchdog.subprocess, 'run',
        lambda command, **kwargs: (calls.append(command) or SimpleNamespace(stdout='masked\n'*4)))
    class RPC:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def request(self, method, params):
            assert method == 'thread/goal/get'
            return dict(goal=dict(status='active'))
    monkeypatch.setattr(watchdog, 'LocalRPC', RPC)
    assert watchdog.run(path) == 0
    assert calls[0] == ['codex', 'app-server', 'daemon', 'start']
    assert len(calls) == 2
    assert json.loads((tmp_path/'unattended_health.json').read_text())['action'] == 'stopped'


def test_backend_failure_leaves_recoverable_evidence_without_fallback(tmp_path, monkeypatch):
    path = tmp_path/'control.json'
    path.write_text(json.dumps(dict(thread_id='target', codex_cli='codex')))
    monkeypatch.setattr(watchdog, 'authorized', lambda *args: True)
    calls = []
    def failed(command, **kwargs):
        calls.append(command)
        raise TimeoutError('native backend timed out')
    monkeypatch.setattr(watchdog.subprocess, 'run', failed)
    assert watchdog.run(path) == 1
    assert len(calls) == 1
    result = json.loads((tmp_path/'unattended_health.json').read_text())
    assert result['action'] == 'health_check_failed'
    assert result['ai_turn_started'] is False
