import json
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import capture_parallel_trajectories as entry
from own_policy_capture import save_commit, sha


def job(folder):
    models = {}
    for name in ['candidate', 'shadow']:
        path = folder/(name+'.zip'); path.write_bytes(name.encode())
        models[name] = dict(path=str(path), sha256=sha(path))
    design = dict(dispatch_version=entry.DISPATCH_VERSION, seeds=list(range(100000, 100008)),
                  max_pieces=20000, frozen_files={}, models=models)
    (folder/'design.json').write_text(json.dumps(design))
    return design


def identity(folder, design):
    return dict(design_sha256=sha(folder/'design.json'), chunk_rounds=256,
        ordered_lanes=design['seeds'], device='cuda', precision='highest', tf32=False,
        batching='active lanes only, in original lane order; no auto-reset')


def before_pickle(monkeypatch):
    def forbidden(*args):
        pytest.fail('Invalid cache reached pickle')
    monkeypatch.setattr(pickle, 'loads', forbidden)


def test_serial_job_cannot_silently_switch_dispatch(tmp_path, monkeypatch):
    design = job(tmp_path); del design['dispatch_version']
    (tmp_path/'design.json').write_text(json.dumps(design))
    before_pickle(monkeypatch)
    with pytest.raises(ValueError, match='serial jobs cannot migrate'):
        entry.prepare(tmp_path)
    assert not (tmp_path/'capture.lock').exists()


def test_changed_checksum_valid_model_identity_cannot_reuse_old_cache(tmp_path, monkeypatch):
    design = job(tmp_path)
    save_commit(tmp_path, 0, identity(tmp_path, design), None, dict(round=0, environment_steps=0), [], {})
    path = Path(design['models']['candidate']['path']); path.write_bytes(b'new-model-identity')
    design['models']['candidate']['sha256'] = sha(path)
    (tmp_path/'design.json').write_text(json.dumps(design))
    protected = {str(p): sha(p) for p in (tmp_path/'chunk_0000').iterdir()}
    before_pickle(monkeypatch)
    with pytest.raises(ValueError, match='identity'):
        entry.prepare(tmp_path)
    assert all(sha(p) == digest for p, digest in protected.items())
    assert not (tmp_path/'capture.lock').exists()


def test_corrupted_snapshot_rejected_before_pickle_and_lock(tmp_path, monkeypatch):
    design = job(tmp_path)
    save_commit(tmp_path, 0, identity(tmp_path, design), None, dict(round=0, environment_steps=0), [], {})
    (tmp_path/'chunk_0000/state.pkl').write_bytes(b'corrupt')
    before_pickle(monkeypatch)
    with pytest.raises(ValueError, match='checksum'):
        entry.prepare(tmp_path)
    assert not (tmp_path/'capture.lock').exists()


def test_wrong_worker_seed_rejected_before_gpu_or_output(tmp_path):
    design = job(tmp_path)
    workers = [dict(state=dict(episode_seed=seed, episode_count=1, finished=False), active=True)
               for seed in design['seeds']]
    workers[3]['state']['episode_seed'] += 10
    save_commit(tmp_path, 0, identity(tmp_path, design), None,
        dict(round=0, environment_steps=0, workers=workers, views=[dict(view='digest')]*8), [], {})
    with pytest.raises(ValueError, match='cached seed'):
        entry.prepare(tmp_path)
    assert not (tmp_path/'capture.lock').exists()


def test_capture_module_bindings_restored_even_on_failure(tmp_path, monkeypatch):
    job(tmp_path)
    original = entry.capture.CaptureNativeEnv, entry.capture.advance_round
    def fail(*args):
        assert entry.capture.CaptureNativeEnv is entry.ParallelNativeEnv
        assert entry.capture.advance_round is not original[1]
        raise RuntimeError('deliberate pre-compute failure')
    monkeypatch.setattr(entry.capture, 'run', fail)
    with pytest.raises(RuntimeError, match='deliberate'):
        entry.run(tmp_path, 1)
    assert (entry.capture.CaptureNativeEnv, entry.capture.advance_round) == original
