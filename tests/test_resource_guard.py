import importlib.util
import json
from pathlib import Path


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
