import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("resource_guard", ROOT / "scripts/resource_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
PROFILE = json.loads((ROOT / "configs/training_resources.json").read_text())


def sample(cpu=70, gpu=50, available=8000):
    return dict(cpu_c=cpu, gpu_c=gpu, available_mib=available)


def test_hot_cpu_reduces_load_and_does_not_bounce_back():
    quota, cool, action = guard.control(sample(cpu=83), 250, 0, 100, PROFILE)
    assert (quota, cool, action) == (200, None, "reduce")
    assert guard.control(sample(), quota, cool, 101, PROFILE) == (200, 101, "hold")
    assert guard.control(sample(), quota, 101, 130, PROFILE)[0] == 200
    assert guard.control(sample(), quota, 101, 131, PROFILE)[0] == 250


def test_cpu_or_gpu_temperature_stops_even_at_minimum_quota():
    for reading in (sample(cpu=90), sample(gpu=85)):
        assert guard.control(reading, 100, None, 0, PROFILE)[2] == "stop:temperature"
    assert guard.control(sample(gpu=76), 100, None, 0, PROFILE)[0] == 100


def test_missing_and_nonfinite_telemetry_stop():
    for reading in (dict(cpu_c=70, available_mib=8000), sample(cpu=float("nan"))):
        assert guard.control(reading, 200, None, 0, PROFILE)[2] == "stop:invalid_telemetry"


def test_memory_pressure_and_quota_ceiling():
    assert guard.control(sample(available=500), 200, None, 0, PROFILE)[2] == "stop:memory_pressure"
    ceiling = PROFILE["max_cpu_quota"]
    assert guard.control(sample(), ceiling, 0, 100, PROFILE)[0] == ceiling


@pytest.mark.parametrize('recovered', [False, True])
def test_preflight_timeout_never_starts_work_without_a_fresh_safe_reading(tmp_path, monkeypatch, recovered):
    calls = []
    def telemetry(required):
        calls.append(required)
        if len(calls) == 1 or not recovered:
            raise guard.subprocess.TimeoutExpired('nvidia-smi', 3)
        # A second reading can be fresh yet unsafe. It must still reject work.
        return sample(cpu=90)
    monkeypatch.setattr(guard, 'read_telemetry', telemetry)
    monkeypatch.setattr(guard, 'physical_cpus', lambda *args: [0])
    monkeypatch.setattr(guard.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(guard, 'operation', lambda *args: None)
    def unexpected_launch(*args, **kwargs):
        pytest.fail('No child may launch during failed preflight')
    monkeypatch.setattr(guard.subprocess, 'run', unexpected_launch)
    output = tmp_path/'guard'
    args = SimpleNamespace(profile=ROOT/'configs/training_resources.json', output=output,
                           seconds=10, command=['--', '/bin/true'])
    assert guard.run(args) == 1
    assert len(calls) == 2
    result = json.loads((output/'result.json').read_text())
    assert result['outcome'] == 'startup_failed'
    assert ('stop:temperature' if recovered else 'TimeoutExpired') in result['error']
    assert not (output/'launch.json').exists()


@pytest.mark.parametrize('final_state,expected', [
    (dict(ActiveState='active', SubState='exited', ExecMainStatus='0'), 'completed'),
    (dict(ActiveState='failed', SubState='failed', Result='exit-code'), 'command_failed:exit-code'),
    (dict(ActiveState='active', SubState='running'), 'technical_failure'),
])
def test_cgroup_disappearing_during_sampling_requires_a_confirmed_terminal_state(tmp_path, monkeypatch, final_state, expected):
    states = iter([dict(ActiveState='active', SubState='running',
                       ControlGroup='/__tetris_missing_cgroup__'), final_state])
    monkeypatch.setattr(guard, 'unit_state', lambda unit: next(states))
    monkeypatch.setattr(guard, 'read_telemetry', lambda required: sample())
    monkeypatch.setattr(guard, 'physical_cpus', lambda *args: [0])
    monkeypatch.setattr(guard, 'operation', lambda *args: None)
    monkeypatch.setattr(guard, 'systemctl', lambda *args, **kwargs: None)
    monkeypatch.setattr(guard.subprocess, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr(guard.signal, 'signal', lambda *args: None)
    output = tmp_path/'guard'
    code = guard.run(SimpleNamespace(profile=ROOT/'configs/training_resources.json', output=output,
                                    seconds=10, command=['--', '/bin/true']))
    result = json.loads((output/'result.json').read_text())
    assert result['outcome'].startswith(expected)
    assert code == (0 if expected == 'completed' else 1)
