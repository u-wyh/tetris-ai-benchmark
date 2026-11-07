import copy
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import capture_gpu_evaluation as entry
import parallel_capture as parallel
from own_policy_capture import save_commit, sha
from verify_feature_backend_benchmark import identical

SEEDS = [2**32-1, 0, 123, 7, 4000, 91, 3, 100015]


def job(folder, monkeypatch):
    import training.train_vector_transaction as training
    import training.evaluation.feature_backend as feature
    import training.native_feature_runtime as native
    monkeypatch.setattr(training, 'training_source_identity', lambda: {'train': 'fixed'})
    monkeypatch.setattr(feature, 'inference_source_identity', lambda **kw: {'infer': 'fixed'})
    monkeypatch.setattr(feature, 'evaluation_runtime_identity', lambda *a: {'runtime': 'fixed'})
    monkeypatch.setattr(native, 'load_runtime', lambda spec: None)
    models = {}
    for name in ['candidate', 'shadow']:
        path = folder/(name+'.zip'); path.write_bytes(name.encode())
        models[name] = dict(path=str(path), sha256=sha(path), counts=[0, 0])
    design = dict(evaluation_version=entry.EVALUATION_VERSION,
        evaluation_sources=entry.evaluation_sources(), dispatch_version='unique-seed-parallel-v1',
        dispatch_mode='parallel', device='cuda', n_envs=8, seeds=SEEDS,
        max_pieces=20000, chunk_rounds=256, frozen_files={}, models=models,
        training_sources={'train': 'fixed'}, inference_sources={'infer': 'fixed'},
        evaluation_runtime={'runtime': 'fixed'}, native_feature_runtime={'fixture': True})
    write(folder, design)
    return design


def write(folder, design):
    (folder/'design.json').write_text(json.dumps(design))


def snapshot():
    return dict(round=0, environment_steps=0, views=[{'digest': 'fixture'}]*8,
        workers=[dict(state=dict(episode_seed=s, episode_count=1, max_pieces=20000,
            pieces=0, finished=False), active=True) for s in SEEDS])


def no_pickle(monkeypatch):
    def fail(*args):
        pytest.fail('Invalid identity/checksum reached pickle')
    monkeypatch.setattr(pickle, 'loads', fail)


@pytest.mark.parametrize('key,value', [
    ('seeds', SEEDS[:7]), ('seeds', [0]*8), ('seeds', [-1]+SEEDS[1:]),
    ('seeds', [2**32]+SEEDS[1:]), ('seeds', [True]+SEEDS[1:]),
    ('device', 'cpu'), ('evaluation_version', 'old'), ('dispatch_mode', 'unknown'),
    ('n_envs', 16), ('max_pieces', 512), ('chunk_rounds', 128)])
def test_invalid_suite_rejected_before_pickle_or_output(tmp_path, monkeypatch, key, value):
    design = job(tmp_path, monkeypatch); design[key] = value; write(tmp_path, design)
    no_pickle(monkeypatch)
    with pytest.raises(ValueError):
        entry.prepare(tmp_path)
    assert not (tmp_path/'capture.lock').exists()


@pytest.mark.parametrize('change', ['zip', 'capture_source', 'training_source', 'runtime', 'native'])
def test_changed_inputs_rejected_before_cache(tmp_path, monkeypatch, change):
    design = job(tmp_path, monkeypatch)
    if change == 'zip':
        Path(design['models']['candidate']['path']).write_bytes(b'changed')
    elif change == 'capture_source':
        design['evaluation_sources'] = {}
    elif change == 'training_source':
        design['training_sources'] = {}
    elif change == 'runtime':
        design['evaluation_runtime'] = {}
    else:
        import training.native_feature_runtime as native
        def invalid(spec):
            raise ValueError('Invalid native manifest')
        monkeypatch.setattr(native, 'load_runtime', invalid)
    write(tmp_path, design); no_pickle(monkeypatch)
    with pytest.raises(ValueError):
        entry.prepare(tmp_path)
    assert not (tmp_path/'capture.lock').exists()


@pytest.mark.parametrize('change', ['order', 'model_identity', 'checksum'])
def test_old_or_corrupt_cache_rejected_before_pickle(tmp_path, monkeypatch, change):
    design = job(tmp_path, monkeypatch)
    _, identity, _ = entry.prepare(tmp_path)
    save_commit(tmp_path, 0, identity, None, snapshot(), [], {})
    if change == 'checksum':
        (tmp_path/'chunk_0000/state.pkl').write_bytes(b'corrupted')
    elif change == 'order':
        design['seeds'] = list(reversed(SEEDS)); write(tmp_path, design)
    else:
        path = Path(design['models']['candidate']['path']); path.write_bytes(b'new-identity')
        design['models']['candidate']['sha256'] = sha(path); write(tmp_path, design)
    protected = {str(p): sha(p) for p in (tmp_path/'chunk_0000').iterdir()}
    no_pickle(monkeypatch)
    with pytest.raises(ValueError, match='identity|checksum'):
        entry.prepare(tmp_path)
    assert all(sha(p) == digest for p, digest in protected.items())
    assert not (tmp_path/'capture.lock').exists()


@pytest.mark.parametrize('field,value', [
    ('episode_seed', 42), ('episode_count', 2), ('max_pieces', 512), ('pieces', 1), ('finished', True)])
def test_bad_snapshot_rejected_before_models_and_lock(tmp_path, monkeypatch, field, value):
    job(tmp_path, monkeypatch); _, identity, _ = entry.prepare(tmp_path)
    state = snapshot(); state['workers'][3]['state'][field] = value
    save_commit(tmp_path, 0, identity, None, state, [], {})
    with pytest.raises(ValueError, match='Cached|Worker'):
        entry.prepare(tmp_path)
    assert not (tmp_path/'capture.lock').exists()


def test_cuda_unavailable_does_not_start_models_workers_or_lock(tmp_path, monkeypatch):
    import torch
    job(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA unavailable'):
        entry.run(tmp_path, 1)
    assert not (tmp_path/'capture.lock').exists()
    assert not list(tmp_path.glob('chunk_*'))


def test_unordered_boundary_seeds_reset_restore_and_stop_without_autoreset(monkeypatch):
    class LocalVector:
        num_envs = 8
        def __init__(self):
            self.envs = [parallel.ParallelCompactEnv(max_pieces=2) for _ in SEEDS]
        def env_method(self, method, *args, indices=None):
            lanes = range(8) if indices is None else indices
            return [getattr(self.envs[i], method)(*args) for i in lanes]
        def close(self):
            for env in self.envs:
                env.close()
    def first(policy, observations, masks, executed=None):
        actions = masks.argmax(axis=1)
        return dict(action=actions, probability=np.ones(len(actions)), margin=np.ones(len(actions)),
                    probability_of_executed=np.ones(len(actions)))
    monkeypatch.setattr(parallel, 'greedy_details', first)
    model = SimpleNamespace(policy=None)
    env = LocalVector()
    try:
        views = entry.reset_ordered(env, SEEDS)
        assert [w['state']['episode_seed'] for w in env.env_method('capture_export')] == SEEDS
        rows = parallel.advance_parallel(env, views, model, model, 0, SEEDS)
        assert [r['episode_seed'] for r in rows] == SEEDS
        saved = copy.deepcopy(env.env_method('capture_export'))
    finally:
        env.close()
    env = LocalVector()
    try:
        views = [env.env_method('capture_restore', state, indices=[i])[0] for i, state in enumerate(saved)]
        assert identical(saved, env.env_method('capture_export'))
        rows = parallel.advance_parallel(env, views, model, model, 1, SEEDS)
        assert [r['episode_seed'] for r in rows] == SEEDS and all(r['truncated'] for r in rows)
        final = env.env_method('capture_export')
        assert views == [None]*8
        assert parallel.advance_parallel(env, views, model, model, 2, SEEDS) == []
        assert identical(final, env.env_method('capture_export'))
        assert all(w['state']['pieces'] == 2 and w['state']['episode_count'] == 1 for w in final)
    finally:
        env.close()
