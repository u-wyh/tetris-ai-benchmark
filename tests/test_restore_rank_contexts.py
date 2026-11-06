"""Exact saved-action replay includes automatic reset and full worker RNG state."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from restore_rank_contexts import (observation_digest,restore_vector,load_workers,load_context,
    save_context,replay_segment,selected_targets)
from collect_critic_calibration import array_manifest
from training.train_vector_transaction import TransactionCompactSearchCandidateEnv
from training.vector_env import make_vector_env
from verify_feature_backend_benchmark import identical


def write_checkpoint(folder,state,observation,mask):
    folder.mkdir()
    payload=pickle.dumps(dict(workers=[state],trainer=dict(num_timesteps=0)),protocol=5)
    (folder/'state.pkl').write_bytes(payload)
    (folder/'checkpoint.json').write_text(json.dumps(dict(state_sha256=hashlib.sha256(payload).hexdigest(),
        num_timesteps=0,worker_digests=[observation_digest(observation,mask)])))


def trace(env,steps):
    records=[]
    for step in range(steps):
        observation=env.get_observation();mask=env.action_masks();action=int(np.flatnonzero(mask)[0])
        _,reward,done,truncated,info=env.step(action)
        row={'obs_'+n:v for n,v in observation.items()}
        row.update(action_masks=mask,actions=np.array([action]),raw_env_rewards=reward,
            step_dones=done or truncated,step_truncated=truncated,step_terminated=done)
        row.update({'info_'+n:info[n] for n in
            ['episode_seed','pieces','cleared_lines','holes_before','holes_after','height_after','new_holes']})
        records.append(row)
        if done or truncated:env.reset()
    return {n:np.stack([r[n] for r in records])[:,None] for n in records[0]}


def test_actual_forkserver_replay_matches_trace_auto_resets_and_full_worker_state(tmp_path):
    direct=TransactionCompactSearchCandidateEnv(max_pieces=2);vector=None
    try:
        obs,_=direct.reset(seed=42);initial=direct.get_state()
        write_checkpoint(tmp_path/'initial',initial,obs,direct.action_masks())
        arrays=trace(direct,7);expected=direct.get_state()
        assert arrays['step_truncated'].sum()==3
        vector=make_vector_env(1,max_pieces=2,env_class=TransactionCompactSearchCandidateEnv)
        restored=restore_vector(vector,tmp_path/'initial');seen=[]
        output,count=replay_segment(vector,restored,arrays,lambda t,o,m:seen.append(t))
        assert count==7 and seen==list(range(7))
        assert identical(expected,vector.env_method('get_state')[0])
        for n in output:np.testing.assert_array_equal(output[n][0],direct.get_observation()[n])
        restored=restore_vector(vector,tmp_path/'initial')
        wrong=copy.deepcopy(arrays);wrong['action_masks'][0]=False
        with pytest.raises(AssertionError):replay_segment(vector,restored,wrong,lambda *a:None)
    finally:
        direct.close()
        if vector is not None:vector.close()


def test_snapshot_hash_rejected_before_deserialization(tmp_path,monkeypatch):
    (tmp_path/'state.pkl').write_bytes(b'untrusted changed bytes')
    (tmp_path/'checkpoint.json').write_text(json.dumps(dict(state_sha256='0'*64)))
    monkeypatch.setattr(pickle,'loads',lambda *a:pytest.fail('Corrupt data must not be unpickled'))
    with pytest.raises(ValueError,match='checksum'):load_workers(tmp_path)


def test_context_commit_identity_and_replayed_state_are_immutable(tmp_path):
    env=TransactionCompactSearchCandidateEnv(max_pieces=10)
    try:
        obs,_=env.reset(seed=42);mask=env.action_masks();state=env.get_state();identity=dict(target=1,source='one')
        directory=tmp_path/'context'
        assert save_context(directory,identity,state,obs,mask)
        before={p.name:p.read_bytes() for p in directory.iterdir()}
        assert not save_context(directory,identity,state,obs,mask)
        assert identical(load_context(directory,identity)[0],state)
        assert before=={p.name:p.read_bytes() for p in directory.iterdir()}
        with pytest.raises(ValueError,match='identity'):load_context(directory,dict(identity,source='changed'))
        changed=copy.deepcopy(state);changed['pieces']+=1
        with pytest.raises(ValueError,match='Replayed context'):save_context(directory,identity,changed,obs,mask)
        (directory/'state.pkl').write_bytes(b'corrupted')
        with pytest.raises(ValueError,match='checksum'):load_context(directory,identity)
    finally:env.close()


def test_target_selection_uses_all_four_clear_losses_with_exact_axes(tmp_path):
    arrays=dict(delta_cleared=np.array([-4,0,-4]),chunk=np.array([1,2,4]),worker=np.array([0,3,7]),
        global_step=np.array([5,700,2000]),reference_action=np.array([1,2,3]),candidate_action=np.array([4,5,6]))
    target=tmp_path/'flipped_geometry.npz';np.savez_compressed(target,**arrays)
    (tmp_path/'rows_manifest.json').write_text(json.dumps(dict(file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        arrays=array_manifest(arrays))))
    selected=selected_targets(tmp_path)
    assert [s['key'] for s in selected]==['c01_t005_w0','c04_t464_w7']
    assert [s['reference_action'] for s in selected]==[1,3]
