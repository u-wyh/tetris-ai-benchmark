import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
spec = importlib.util.spec_from_file_location('desktop_probe',
    Path(__file__).resolve().parents[1]/'scripts/probe_unattended_desktop.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_reused_pid_is_never_signaled(monkeypatch):
    sent, closed = [], []
    monkeypatch.setattr(probe.os, 'pidfd_open', lambda pid: 99)
    monkeypatch.setattr(probe.os, 'close', closed.append)
    monkeypatch.setattr(probe, 'identity', lambda pid: dict(start_ticks='new'))
    monkeypatch.setattr(probe.signal, 'pidfd_send_signal', lambda *args: sent.append(args))
    with pytest.raises(RuntimeError, match='identity changed'):
        probe.send_proxy(dict(pid=123, start_ticks='old'), signal.SIGSTOP)
    assert not sent and closed == [99]


def test_partial_pause_failure_restores_every_recorded_proxy(tmp_path, monkeypatch):
    control = tmp_path/'control.json'
    control.write_text(json.dumps(dict(enabled=True, execution_host='remote-ssh-discovered:lightbook',
        execution_machine_id=Path('/etc/machine-id').read_text().strip(), thread_id='target')))
    (tmp_path/'design.json').write_text(json.dumps(dict(control=str(control),
        seconds=1, backend=dict(pid=101, start_ticks='101'))))
    monkeypatch.setattr(probe, 'allowed', lambda *args: True)
    monkeypatch.setattr(probe, 'identity', lambda pid: dict(pid=pid, start_ticks=str(pid),
        parent=1 if pid == 101 else 101, state='S'))
    monkeypatch.setattr(probe, 'desktop_proxies', lambda: [201, 202])
    monkeypatch.setattr(probe.signal, 'signal', lambda *args: None)
    calls = []
    def send(row, sig):
        calls.append((row['pid'], sig))
        if row['pid'] == 202 and sig == signal.SIGSTOP:
            raise RuntimeError('second proxy vanished')
    monkeypatch.setattr(probe, 'send_proxy', send)
    with pytest.raises(RuntimeError, match='vanished'):
        probe.run(tmp_path)
    assert calls[-2:] == [(201, signal.SIGCONT), (202, signal.SIGCONT)]
    assert json.loads((tmp_path/'state.json').read_text())['restored_at']
    probe.restore(tmp_path)  # ExecStopPost is idempotent.
    assert len(calls) == 4


@pytest.mark.parametrize('problem', [None, 'proxy_running', 'new_proxy', 'foreign_parent', 'restored'])
def test_execution_receipt_requires_paused_transport_and_native_parent(tmp_path, monkeypatch, problem):
    state = dict(phase='proxy_paused', backend=dict(pid=101, start_ticks='101'),
                 proxies=[dict(pid=201, start_ticks='201')])
    if problem == 'restored':
        state['restored_at'] = 'already restored'
    (tmp_path/'state.json').write_text(json.dumps(state))
    parent = 999 if problem == 'foreign_parent' else 101
    def identity(pid):
        return dict(pid=pid, start_ticks=str(pid),
            state='S' if problem == 'proxy_running' or pid != 201 else 'T',
            parent=parent if pid == os.getpid() else 0 if pid == 1 else 1)
    monkeypatch.setattr(probe, 'identity', identity)
    monkeypatch.setattr(probe, 'desktop_proxies', lambda: [201, 202] if problem == 'new_proxy' else [201])
    if problem:
        with pytest.raises(RuntimeError):
            probe.observe(tmp_path)
        assert not (tmp_path/'observation.json').exists()
    else:
        probe.observe(tmp_path)
        assert json.loads((tmp_path/'observation.json').read_text())['executed_while_all_proxies_paused']
