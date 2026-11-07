import copy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_gpu_evaluation_phase as phase
from own_policy_capture import sha
from test_gpu_evaluation_status import fixture


def state(**changes):
    s = dict(status='resumable_partial', can_resume=True, active_lanes=list(range(8)),
        round=0, actual_unique_environment_steps=0, phase_round_limit=20000,
        phase_environment_step_limit=160000, last_manifest_sha256='a'*64, committed_chunks=1,
        identity={'fixture': True})
    s.update(changes)
    return s


@pytest.mark.parametrize('changes,requested,action,chunks,rounds,steps', [
    ({'status': 'fresh', 'active_lanes': []}, 20, 'capture', 20, 5120, 40960),
    ({'phase_round_limit': 256, 'phase_environment_step_limit': 2048}, 20, 'capture', 1, 256, 2048),
    ({'phase_round_limit': 255}, 1, 'preserved', 0, None, None),
    ({'phase_environment_step_limit': 2047}, 1, 'preserved', 0, None, None),
    ({'active_lanes': [0, 3], 'phase_environment_step_limit': 1024}, 20, 'capture', 2, 512, 1024),
    ({'round': 19968, 'actual_unique_environment_steps': 159744}, 20, 'capture', 1, 32, 256),
    ({'active_lanes': [], 'round': 128, 'actual_unique_environment_steps': 800}, 20, 'finalize', 1, 0, 0),
    ({'can_resume': False, 'round': 512, 'phase_round_limit': 512}, 1, 'preserved', 0, None, None),
    ({'status': 'terminal_verified'}, 1, 'preserved', 0, None, None),
    ({'status': 'invalid'}, 1, 'blocked', 0, None, None),
    ({'status': 'blocked'}, 1, 'blocked', 0, None, None)])
def test_conservative_chunk_bounds(changes, requested, action, chunks, rounds, steps):
    p = phase.plan(state(**changes), requested)
    assert (p['action'], p['chunks']) == (action, chunks)
    if rounds is not None:
        assert p['round_upper_bound'] == rounds and p['environment_step_upper_bound'] == steps


@pytest.mark.parametrize('requested', [0, 21, True, 1.5])
def test_invalid_requested_chunks(requested):
    with pytest.raises(ValueError): phase.plan(state(), requested)


def attempt(tmp_path, monkeypatch, initial, mode='capture'):
    # No real model, workers, torch inference or cache creation in these tests.
    root = tmp_path/'root'; root.mkdir(); monkeypatch.setattr(phase, 'ROOT', root)
    job = root/'runs/job'; job.mkdir(parents=True)
    (job/'design.json').write_text('{}')
    run = tmp_path/'attempt'; run.mkdir()
    request = dict(version=phase.VERSION, mode=mode, job_folder=str(job),
        job_design_sha256=sha(job/'design.json'), max_chunks=2, expected_start=phase.boundary(initial))
    (run/'request.json').write_text(json.dumps(request))
    monkeypatch.setattr(phase, 'inspect', lambda *a: copy.deepcopy(initial))
    return run, job, request


def test_closed_phase_never_checks_guard_or_enters_compute_and_is_idempotent(tmp_path, monkeypatch):
    initial = state(can_resume=False, round=512, phase_round_limit=512)
    run, job, _ = attempt(tmp_path, monkeypatch, initial)
    before = {p.name: p.read_bytes() for p in job.iterdir()}
    def forbidden(*args): pytest.fail('Closed phase entered compute/guard')
    monkeypatch.setattr(phase, 'require_authorization', forbidden)
    monkeypatch.setattr(phase, 'execute', forbidden)
    first = phase.run(run); second = phase.run(run)
    assert first['status'] == 'preserved' and second['reused_receipt']
    assert not (run/'started.json').exists()
    assert before == {p.name: p.read_bytes() for p in job.iterdir()}


def test_stale_boundary_blocks_without_worker_or_guard(tmp_path, monkeypatch):
    run, _, _ = attempt(tmp_path, monkeypatch, state())
    monkeypatch.setattr(phase, 'inspect', lambda *a: state(round=256))
    monkeypatch.setattr(phase, 'execute', lambda *a: pytest.fail('Stale request computed'))
    assert phase.run(run)['status'] == 'blocked'


def test_incomplete_previous_attempt_never_repeats_work(tmp_path, monkeypatch):
    run, _, _ = attempt(tmp_path, monkeypatch, state())
    (run/'started.json').write_text('{}')
    monkeypatch.setattr(phase, 'execute', lambda *a: pytest.fail('Repeated interrupted attempt'))
    r = phase.run(run)
    assert r['status'] == 'blocked' and 'Prior attempt' in r['reason']


def test_guard_failure_prevents_compute_and_preserves_original_job(tmp_path, monkeypatch):
    run, job, _ = attempt(tmp_path, monkeypatch, state())
    monkeypatch.setattr(phase, 'require_authorization', lambda *a: None)
    monkeypatch.setattr(phase, 'sealed_sources', lambda *a: None)
    def fail(*a): raise RuntimeError('No live guard')
    monkeypatch.setattr(phase, 'verify_live_guard', fail)
    monkeypatch.setattr(phase, 'execute', lambda *a: pytest.fail('Unprotected compute'))
    before = list(job.iterdir()); r = phase.run(run)
    assert r['status'] == 'blocked' and not (run/'started.json').exists() and before == list(job.iterdir())


def test_old_jobs_cannot_silently_adopt_new_phase_execution():
    with pytest.raises(ValueError, match='old caches cannot migrate'):
        phase.sealed_sources({})


def test_finalization_cannot_fabricate_missing_process_work(tmp_path):
    with pytest.raises(ValueError, match='cannot fabricate'):
        phase.finalization_accounted(tmp_path, state(round=256, actual_unique_environment_steps=2048))


def test_guard_probe_never_calls_model_compute(tmp_path, monkeypatch):
    run, _, _ = attempt(tmp_path, monkeypatch, state(), mode='guard_probe')
    monkeypatch.setattr(phase, 'require_authorization', lambda *a: None)
    monkeypatch.setattr(phase, 'verify_live_guard', lambda *a: {'fixture': True})
    monkeypatch.setattr(phase, 'execute', lambda *a: pytest.fail('Probe entered compute'))
    r = phase.run(run)
    assert r['status'] == 'guard_verified_no_compute' and not (run/'started.json').exists()
    assert r['new_environment_steps'] == r['gpu_forward_calls'] == r['optimizer_steps'] == 0


def test_execution_failure_does_not_fabricate_zero_actual_work(tmp_path, monkeypatch):
    run, _, _ = attempt(tmp_path, monkeypatch, state())
    monkeypatch.setattr(phase, 'require_authorization', lambda *a: None)
    monkeypatch.setattr(phase, 'sealed_sources', lambda *a: None)
    monkeypatch.setattr(phase, 'verify_live_guard', lambda *a: {})
    def interrupted(*a): raise RuntimeError('Interrupted after possible uncommitted work')
    monkeypatch.setattr(phase, 'execute', interrupted)
    r = phase.run(run)
    assert r['status'] == 'blocked' and (run/'started.json').exists()
    assert r['new_environment_steps'] is None and r['gpu_forward_calls'] is None
    assert 'unknown' in r['work_accounting']


@pytest.mark.parametrize('race', [False, True])
def test_additional_boundary_check_holds_under_actual_lock_and_restores_bindings(tmp_path, monkeypatch, race):
    initial = state(); request = dict(expected_start=phase.boundary(initial), max_chunks=1)
    selected = phase.plan(initial, 1)
    saved = dict(round=0, environment_steps=0, workers=[dict(active=True) for _ in range(8)])
    original_chain = lambda *a: (saved, 'a'*64, 1)
    original_inputs = lambda *a: None
    monkeypatch.setattr(phase.capture, 'read_chain', original_chain)
    monkeypatch.setattr(phase.capture, 'check_inputs', original_inputs)
    monkeypatch.setattr(phase, 'require_authorization', lambda *a: None)
    monkeypatch.setattr(phase, 'verify_live_guard', lambda *a: {})
    def fake_compute(folder, chunks):
        assert chunks == 1
        phase.capture.read_chain(folder, initial['identity'])  # original preflight boundary
        with (folder/'capture.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if race: saved['round'] = 256  # Independent writer advanced before this lock.
            phase.capture.read_chain(folder, initial['identity'])
            phase.capture.check_inputs({})
    monkeypatch.setattr(phase.capture, 'run', fake_compute)
    if race:
        with pytest.raises(RuntimeError, match='boundary changed'):
            phase.execute(tmp_path, request, tmp_path, initial, selected, {})
    else:
        assert phase.execute(tmp_path, request, tmp_path, initial, selected, {}) == 2
    assert phase.capture.read_chain is original_chain and phase.capture.check_inputs is original_inputs


def test_original_chain_validation_is_not_bypassed(tmp_path, monkeypatch):
    def fail(*a): raise ValueError('Original identity checksum rejected')
    monkeypatch.setattr(phase.capture, 'read_chain', fail)
    original_inputs = phase.capture.check_inputs
    monkeypatch.setattr(phase.capture, 'run', lambda folder, chunks: phase.capture.read_chain(folder, {}))
    initial = state()
    with pytest.raises(ValueError, match='Original identity checksum rejected'):
        phase.execute(tmp_path, dict(max_chunks=1, expected_start=phase.boundary(initial)), tmp_path,
                      initial, phase.plan(initial, 1), {})
    assert phase.capture.read_chain is fail and phase.capture.check_inputs is original_inputs


def test_real_fixture_active_writer_cannot_enter_phase(tmp_path, monkeypatch):
    job = tmp_path/'root/runs/job'; job.mkdir(parents=True)
    monkeypatch.setattr(phase, 'ROOT', tmp_path/'root')
    fixture(job, monkeypatch, stopped=False, summary=False)
    initial = phase.inspect(job)
    run = tmp_path/'attempt';run.mkdir()
    request = dict(version=phase.VERSION, mode='capture', job_folder=str(job), max_chunks=1,
        job_design_sha256=sha(job/'design.json'), expected_start=phase.boundary(initial))
    (run/'request.json').write_text(json.dumps(request))
    monkeypatch.setattr(phase, 'execute', lambda *a: pytest.fail('Live writer repeated'))
    with (job/'capture.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = phase.run(run)
        assert r['status'] == 'blocked' and not (run/'started.json').exists()
