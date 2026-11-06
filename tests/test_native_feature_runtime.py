"""Reject native artifact/config changes before mutating resumable runs."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from training.env.bitboard_search_env import validate_training_feature_backend
from training.native_feature_runtime import preflight, runtime_identity, verify_runtime_identity
from training.train_vector_transaction import (TransactionCompactSearchCandidateEnv,
    TransactionNativeSearchCandidateEnv, run, TRAINING_SOURCE_FILES, training_source_identity,
    verify_training_source_identity, verify_task, write_manifest, vector_config)
from training.extend_candidate import prepare


@pytest.fixture
def preset():
    manifest = Path(os.environ['TETRIS_NATIVE_MANIFEST'])
    return dict(candidate_policy=True, fast_candidate_env=True,
        candidate_feature_backend='bitboard_native', candidate_search_backend='bitboard_compact',
        native_feature_runtime=dict(manifest=str(manifest), sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))


def snapshot(folder):
    return {str(p.relative_to(folder)):p.read_bytes() for p in folder.rglob('*') if p.is_file()}


@pytest.mark.parametrize('patch', [dict(candidate_policy=False), dict(fast_candidate_env=False),
    dict(failure_curriculum={'mode':'normal'}), dict(candidate_search_backend='formal'),
    dict(candidate_search_backend='bitboard')])
def test_explicit_native_backend_restrictions(preset, patch):
    assert validate_training_feature_backend(preset) == 'bitboard_native'
    with pytest.raises(ValueError): validate_training_feature_backend(dict(preset, **patch))


@pytest.mark.parametrize('spec', [None, {}, dict(manifest='relative',sha256='0'*64),
    dict(manifest='/absent',sha256='wrong'), dict(manifest='/absent',sha256='0'*64,unknown=True)])
def test_invalid_fresh_runtime_does_not_create_run(tmp_path, preset, spec):
    preset['native_feature_runtime'] = spec
    config = tmp_path/'preset.json'; config.write_text(json.dumps(preset))
    destination = tmp_path/'new_parent'/'new_run'
    with pytest.raises((ValueError,OSError)):
        run(destination, config_file=config)
    assert not destination.parent.exists()


def test_plain_backends_do_not_load_binary(tmp_path, monkeypatch):
    import training.native_feature_runtime as implementation
    monkeypatch.setattr(implementation, 'load_runtime', lambda *a: pytest.fail('Plain backend loaded binary'))
    assert runtime_identity({}) is None
    assert runtime_identity(dict(candidate_feature_backend='bitboard')) is None
    assert preflight(tmp_path/'unused',False,None) is None
    assert verify_runtime_identity({}, {}) is None
    env=TransactionCompactSearchCandidateEnv(max_pieces=10)
    try:
        obs,_=env.reset(seed=42)
        assert obs and env.action_masks().any()
    finally: env.close()


def fake_run(tmp_path, preset):
    source=tmp_path/'source'; source.mkdir()
    identity=runtime_identity(preset)
    config=dict(preset,native_feature_runtime_identity=identity)
    metadata=dict(candidate_feature_backend='bitboard_native',native_feature_runtime_identity=identity)
    (source/'config.json').write_text(json.dumps(config))
    (source/'metadata.json').write_text(json.dumps(metadata))
    (source/'.train.lock').write_bytes(b'existing lock must remain byte identical')
    (source/'working').mkdir(); (source/'working'/'pending').write_bytes(b'preserve')
    (source/'committed').mkdir(); (source/'committed'/'checkpoint').write_bytes(b'preserve')
    return source,config,metadata


@pytest.mark.parametrize('mutation', ['requested_backend','requested_runtime','config_identity',
                                    'metadata_identity','recorded_backend'])
def test_resume_rejection_preserves_entire_run(tmp_path, preset, mutation):
    source,config,metadata=fake_run(tmp_path,preset)
    requested=deepcopy(preset)
    if mutation=='requested_backend': requested['candidate_feature_backend']='bitboard'
    if mutation=='requested_runtime': requested['native_feature_runtime']['sha256']='0'*64
    if mutation=='config_identity':
        config['native_feature_runtime_identity']['artifact']['binary_sha256']='0'*64
        (source/'config.json').write_text(json.dumps(config))
    if mutation=='metadata_identity':
        metadata['native_feature_runtime_identity']['artifact']['flags']=['-O0']
        (source/'metadata.json').write_text(json.dumps(metadata))
    if mutation=='recorded_backend':
        metadata['candidate_feature_backend']='bitboard'
        (source/'metadata.json').write_text(json.dumps(metadata))
    path=tmp_path/'requested.json'; path.write_text(json.dumps(requested))
    before=snapshot(source)
    with pytest.raises((ValueError,RuntimeError)): run(source,resume=True,config_file=path)
    assert snapshot(source)==before


@pytest.mark.parametrize('mutation', ['missing','binary','manifest'])
def test_corrupt_artifact_rejects_resume_and_extension_before_writes(tmp_path,preset,mutation):
    artifact=tmp_path/'artifact'; artifact.mkdir()
    manifest=json.loads(Path(preset['native_feature_runtime']['manifest']).read_text())
    binary=artifact/'candidate_features.so'; binary.write_bytes(Path(manifest['binary']).read_bytes())
    manifest['binary']=str(binary)
    manifest_path=artifact/'manifest.json'; manifest_path.write_text(json.dumps(manifest))
    preset['native_feature_runtime']=dict(manifest=str(manifest_path),sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    source,_,_=fake_run(tmp_path,preset)
    if mutation=='missing': binary.unlink()
    if mutation=='binary':
        # Replace the directory entry: truncating an already dlopen-mapped inode
        # can SIGBUS the test process at shutdown, before identity rejection.
        replacement=artifact/'corrupt.so'; replacement.write_bytes(b'corrupted without changing pinned hash')
        replacement.replace(binary)
    if mutation=='manifest': manifest_path.write_text('{}')
    before=snapshot(source); destination=tmp_path/'new_parent'/'extension'
    with pytest.raises((ValueError,RuntimeError,OSError)): run(source,resume=True)
    assert snapshot(source)==before
    with pytest.raises((ValueError,RuntimeError,OSError)): prepare(source,destination,8192,evaluate_baseline=False)
    assert snapshot(source)==before and not destination.parent.exists()


def test_control_arm_declaration_matches_candidate_identity(preset):
    assert runtime_identity(preset)==runtime_identity(dict(preset,candidate_feature_backend='bitboard'))


def test_real_native_environment_cache_state_and_trajectory(preset):
    left=TransactionCompactSearchCandidateEnv(max_pieces=50)
    right=TransactionNativeSearchCandidateEnv(max_pieces=50,native_feature_runtime=preset['native_feature_runtime'])
    rng=np.random.default_rng(326013)
    try:
        observations=[e.reset(seed=326014)[0] for e in (left,right)]
        for step in range(200):
            for key in observations[0]: np.testing.assert_array_equal(observations[0][key],observations[1][key])
            np.testing.assert_array_equal(left.action_masks(),right.action_masks())
            assert right._features_and_mask() is right._features_and_mask()
            if step==20:
                right.set_state(left.get_state())
            action=int(rng.choice(np.flatnonzero(left.action_masks())))
            result=[e.step(action) for e in (left,right)]
            assert result[0][1:]==result[1][1:]
            observations=[r[0] for r in result]
            if result[0][2] or result[0][3]: observations=[e.reset()[0] for e in (left,right)]
    finally: left.close(); right.close()


@pytest.mark.parametrize('relative', ['training/native_candidate_features.c',
    'training/native_candidate_features.py','training/native_feature_runtime.py','training/extend_candidate.py'])
def test_native_source_closure_rejects_changes(tmp_path,relative):
    root=Path(__file__).resolve().parents[1]
    for name in TRAINING_SOURCE_FILES:
        target=tmp_path/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/name).read_bytes())
    metadata=dict(training_source_sha256=training_source_identity(tmp_path))
    assert relative in metadata['training_source_sha256']
    (tmp_path/relative).write_text('changed')
    with pytest.raises(RuntimeError,match=relative): verify_training_source_identity(metadata,tmp_path)


def test_task_metadata_pins_compiled_identity(tmp_path,preset):
    config=vector_config(candidate_policy=True)
    config.update(preset,native_feature_runtime_identity=runtime_identity(preset))
    metadata=dict(task_id=1,end_step=4096,device='cuda',n_envs=8,n_steps=512,
        worker_digests=['x']*8,reward_version=config['reward_version'],
        observation_version=config['observation_version'],policy=config['policy'],
        candidate_feature_version=config['candidate_feature_version'],candidate_feature_backend='bitboard_native',
        native_feature_runtime_identity=deepcopy(config['native_feature_runtime_identity']))
    for name in ('model.zip','trainer_state.pkl','vector_env_state.pkl','metrics.json'): (tmp_path/name).write_bytes(b'x')
    (tmp_path/'task_meta.json').write_text(json.dumps(metadata)); write_manifest(tmp_path)
    verify_task(tmp_path,1,config)
    metadata['native_feature_runtime_identity']['artifact']['binary_sha256']='wrong'
    (tmp_path/'task_meta.json').write_text(json.dumps(metadata)); write_manifest(tmp_path)
    with pytest.raises(RuntimeError,match='feature runtime'): verify_task(tmp_path,1,config)
