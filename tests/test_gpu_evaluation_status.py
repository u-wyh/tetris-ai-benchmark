import copy
import fcntl
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import inspect_gpu_evaluation as status
from own_policy_capture import save_commit, sha
from test_gpu_evaluation import job, snapshot, SEEDS, write
from training.candidate_features import FEATURE_NAMES


def fixture(folder, monkeypatch, stopped=True, summary=True):
    design = job(folder, monkeypatch)
    _, identity, _ = status.prepare(folder)
    audit = dict(policy_adam_counts_unchanged=True, global_sampling_rng_unchanged=True,
                 optimizer_steps=0, model_counts={name: spec['counts'] for name, spec in design['models'].items()})
    initial = snapshot(); initial['rng'] = {'fixture': np.arange(4)}
    for w in initial['workers']:
        w['state'].update(core=SimpleNamespace(score=0, lines=0, phase='playing'), episode_reward=0.,
                          numpy_generator={'state': 1}, action_generator={'state': 2}, numpy_seed=42)
    previous = save_commit(folder, 0, identity, None, initial, [], audit)
    final = copy.deepcopy(initial); final.update(round=1, environment_steps=8)
    for lane, w in enumerate(final['workers']):
        w['active'] = not stopped
        w['state'].update(pieces=1, finished=stopped, episode_reward=1.)
        w['state']['core'].phase = 'over' if stopped else 'playing'
        if stopped:
            final['views'][lane] = None
    rows = [dict(round=0, lane=i, pieces=1, episode_seed=seed, action=0, shadow_action=0,
        candidate_features=np.zeros(16, dtype=np.uint8), shadow_features=np.zeros(16, dtype=np.uint8),
        candidate_probability=1., shadow_probability=1., shadow_probability_of_executed=1.,
        candidate_margin=0., shadow_margin=0., reward=1., cleared_lines=0,
        terminated=stopped, truncated=False, holes_before=0, holes_after=0, height_after=0,
        observation_sha256='a'*64, mask_sha256='b'*64) for i, seed in enumerate(SEEDS)]
    save_commit(folder, 1, identity, previous, final, rows, audit)
    process = dict(pid=123, environment_steps_this_process=8,
        candidate_forward_calls_this_process=1, shadow_forward_calls_this_process=1, optimizer_steps=0)
    (folder/'process_123.json').write_text(json.dumps(process))
    if summary:
        endpoints = [dict(lane=i, seed=seed, steps=1, score=0, lines=0, game_over=stopped, cap_censored=False)
                     for i, seed in enumerate(SEEDS)]
        marker = dict(status='passed', identity=identity, final_round=1, environment_steps=8, endpoints=endpoints)
        (folder/'capture_summary.json').write_text(json.dumps(marker))
    return design, identity


def upstream(tmp_path, passed=True):
    p = tmp_path.parent/(tmp_path.name+'-controller.json')
    p.write_text(json.dumps(dict(status='passed' if passed else 'running')))
    return p


def unchanged(folder, call):
    before = status.fingerprint(folder)
    result = call()
    assert before == status.fingerprint(folder)
    assert not (folder/'trajectory_audit.json').exists()
    assert not (folder/'capture.lock').exists()
    return result


def test_fresh_has_no_actual_active_lanes_or_completion(tmp_path, monkeypatch):
    job(tmp_path, monkeypatch)
    r = unchanged(tmp_path, lambda: status.inspect(tmp_path))
    assert r['status'] == 'fresh' and r['actual_unique_environment_steps'] == 0
    assert r['active_lanes'] == [] and r['committed_chunks'] == 0
    assert not r['completed'] and not r['can_resume']


def test_partial_remains_partial_even_with_passed_controller(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch, stopped=False, summary=False)
    r = unchanged(tmp_path, lambda: status.inspect(tmp_path, upstream(tmp_path)))
    assert r['status'] == 'resumable_partial' and r['can_resume'] and not r['completed']
    assert r['active_lanes'] == list(range(8)) and r['round'] == 1
    assert r['actual_unique_environment_steps'] == 8 and not r['trace_endpoints_verified']


def test_stopped_without_summary_requires_finalize_not_new_steps(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch, summary=False)
    r = unchanged(tmp_path, lambda: status.inspect(tmp_path))
    assert r['status'] == 'resumable_partial' and r['can_resume'] and r['active_lanes'] == []
    assert not r['completed'] and 'no new environment steps' in r['next_action']


def test_finite_phase_budget_does_not_authorize_extending_partial_job(tmp_path, monkeypatch):
    design = job(tmp_path, monkeypatch)
    design['phase_round_limit'] = 1; design['phase_environment_step_limit'] = 8
    write(tmp_path, design)
    _, identity, _ = status.prepare(tmp_path)
    initial = snapshot(); audit = dict(policy_adam_counts_unchanged=True, global_sampling_rng_unchanged=True,
        optimizer_steps=0, model_counts={name: spec['counts'] for name, spec in design['models'].items()})
    previous = save_commit(tmp_path, 0, identity, None, initial, [], audit)
    final = copy.deepcopy(initial); final.update(round=1, environment_steps=8)
    for w in final['workers']: w['state']['pieces'] = 1
    save_commit(tmp_path, 1, identity, previous, final, [dict(lane=i) for i in range(8)], audit)
    r = unchanged(tmp_path, lambda: status.inspect(tmp_path))
    assert r['status'] == 'resumable_partial' and r['phase_budget_reached']
    assert not r['completed'] and not r['can_resume'] and 'without automatic extension' in r['next_action']


@pytest.mark.parametrize('receipt', ['missing', 'running'])
def test_terminal_needs_passed_protection_receipt(tmp_path, monkeypatch, receipt):
    fixture(tmp_path, monkeypatch)
    r = unchanged(tmp_path, lambda: status.inspect(tmp_path,
        None if receipt == 'missing' else upstream(tmp_path, passed=False)))
    assert r['status'] == 'blocked' and not r['completed'] and not r['can_resume']


def test_terminal_runs_real_frozen_auditor_without_writing_job(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    r = unchanged(tmp_path, lambda: status.inspect(tmp_path, upstream(tmp_path)))
    assert r['status'] == 'terminal_verified' and r['completed'] and not r['can_resume']
    assert r['terminal_audit']['complete_worker_trace_endpoints_exact']
    assert r['actual_unique_environment_steps'] == 8 and r['trace_endpoints_verified']
    assert r['new_environment_steps'] == r['gpu_forward_calls'] == r['optimizer_steps'] == 0


def test_forged_summary_with_live_lanes_is_invalid(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch, stopped=False)
    r = unchanged(tmp_path, lambda: status.inspect(tmp_path, upstream(tmp_path)))
    assert r['status'] == 'invalid' and not r['completed'] and not r['can_resume']


@pytest.mark.parametrize('change', ['endpoint_count', 'endpoint_score', 'final_round', 'summary_steps', 'identity'])
def test_wrong_marker_cannot_claim_terminal(tmp_path, monkeypatch, change):
    fixture(tmp_path, monkeypatch)
    p = tmp_path/'capture_summary.json'; marker = json.loads(p.read_text())
    if change == 'endpoint_count': marker['endpoints'].pop()
    elif change == 'endpoint_score': marker['endpoints'][0]['score'] = 10
    elif change == 'final_round': marker['final_round'] = 20000
    elif change == 'summary_steps': marker['environment_steps'] = 160000
    else: marker['identity']['device'] = 'cpu'
    p.write_text(json.dumps(marker))
    r = unchanged(tmp_path, lambda: status.inspect(tmp_path, upstream(tmp_path)))
    assert r['status'] == 'invalid' and not r['completed']


@pytest.mark.parametrize('change', ['seed_order', 'bad_hash', 'snapshot_counter', 'model_audit'])
def test_invalid_cache_rejected_without_resume(tmp_path, monkeypatch, change):
    design, _ = fixture(tmp_path, monkeypatch, stopped=False, summary=False)
    if change == 'seed_order':
        design['seeds'] = list(reversed(SEEDS)); write(tmp_path, design)
    elif change == 'bad_hash':
        (tmp_path/'chunk_0001/state.pkl').write_bytes(b'broken')
    elif change == 'snapshot_counter':
        p = tmp_path/'chunk_0001/state.pkl'; state = pickle.loads(p.read_bytes())
        state['workers'][0]['state']['pieces'] = 0; p.write_bytes(pickle.dumps(state, protocol=5))
        manifest = tmp_path/'chunk_0001/manifest.json'; m = json.loads(manifest.read_text())
        m['files']['state.pkl'] = sha(p); manifest.write_text(json.dumps(m))
    else:
        p = tmp_path/'chunk_0001/manifest.json'; m = json.loads(p.read_text())
        m['audit']['optimizer_steps'] = 1; p.write_text(json.dumps(m))
    if change in ['seed_order', 'bad_hash']:
        def forbidden(*args): pytest.fail('Invalid cache reached pickle')
        monkeypatch.setattr(pickle, 'loads', forbidden)
    r = unchanged(tmp_path, lambda: status.inspect(tmp_path))
    assert r['status'] == 'invalid' and not r['completed'] and not r['can_resume']


def test_terminal_auditor_rejects_rehashed_bad_feature_outcome_alignment(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    p = tmp_path/'chunk_0001/rows.npz'
    with np.load(p, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    data['candidate_features'][0, FEATURE_NAMES.index('cleared_lines')] = 1
    np.savez_compressed(p, **data)
    mpath = tmp_path/'chunk_0001/manifest.json'; m = json.loads(mpath.read_text())
    m['files']['rows.npz'] = sha(p); mpath.write_text(json.dumps(m))
    r = unchanged(tmp_path, lambda: status.inspect(tmp_path, upstream(tmp_path)))
    assert r['status'] == 'invalid' and 'alignment' in r['reason'] and not r['completed']


def test_missing_process_work_cannot_claim_terminal(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch); (tmp_path/'process_123.json').unlink()
    r = unchanged(tmp_path, lambda: status.inspect(tmp_path, upstream(tmp_path)))
    assert r['status'] == 'invalid' and 'accounting' in r['reason']


def test_live_writer_is_blocked_and_not_read_or_restarted(tmp_path, monkeypatch):
    job(tmp_path, monkeypatch)
    with (tmp_path/'capture.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(status, 'prepare', lambda *a: pytest.fail('Read live writer'))
        r = status.inspect(tmp_path)
        assert r['status'] == 'blocked' and not r['can_resume'] and not r['completed']
    assert not list(tmp_path.glob('chunk_*'))


def test_inputs_changing_during_inspection_are_blocked(tmp_path, monkeypatch):
    job(tmp_path, monkeypatch)
    original = status.prepare
    def change_after_preflight(view):
        value = original(view)
        (tmp_path/'process_new.json').write_text('{}')
        return value
    monkeypatch.setattr(status, 'prepare', change_after_preflight)
    r = status.inspect(tmp_path)
    assert r['status'] == 'blocked' and not r['can_resume'] and not r['completed']
