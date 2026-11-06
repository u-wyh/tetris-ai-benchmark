"""Masked distribution math, policy-mode fidelity and immutable chunk recovery."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_policy_action_ranks import (masked_log_probabilities, compare_forward, terminal_distances,
    commit_cache, read_cache, forward)


def result(logits, masks):
    probabilities=np.exp(masked_log_probabilities(logits,masks)).astype(np.float32)
    return dict(logits=np.asarray(logits,dtype=np.float32),probabilities=probabilities,
                actions=probabilities.argmax(axis=1))


def test_masked_kl_matches_independent_small_categorical_and_features():
    masks=np.array([[True,True,False],[False,True,False]])
    left=result([[0,1,100],[0,0,100]],masks)
    right=result([[1,0,-100],[2,0,-100]],masks)
    features=np.arange(2*3*16,dtype=np.uint8).reshape(2,3,16)
    rows=compare_forward(left,right,masks,features)
    p=np.exp([0,1])/np.exp([0,1]).sum();q=p[::-1]
    np.testing.assert_allclose(rows['masked_kl_reference_candidate'],[np.sum(p*np.log(p/q)),0])
    np.testing.assert_array_equal(rows['disagrees'],[True,False])
    assert rows['reference_top2_logit_margin'][0]==1
    assert np.isinf(rows['reference_top2_logit_margin'][1])
    np.testing.assert_array_equal(rows['reference_features'],features[[0,1],[1,1]])
    np.testing.assert_array_equal(rows['candidate_features'],features[[0,1],[0,1]])


def test_actual_probability_tie_does_not_become_logit_argmax_action():
    masks=np.array([[True,True]])
    observed=dict(logits=np.array([[0,1e-8]],dtype=np.float32),
        probabilities=np.array([[.5,.5]],dtype=np.float32),actions=np.array([0]))
    row=compare_forward(observed,observed,masks,np.zeros((1,2,16),dtype=np.uint8))
    assert row['reference_action'][0]==0 and row['reference_mode_differs_from_logit_argmax'][0]
    assert not row['disagrees'][0] and row['masked_kl_reference_candidate'][0]==0


def test_extreme_legal_logits_and_masked_nan_have_no_probability_floor():
    masks=np.array([[True,True,False]])
    left=result([[0,-10000,np.nan]],masks);right=result([[-10000,0,np.nan]],masks)
    rows=compare_forward(left,right,masks,np.zeros((1,3,16),dtype=np.uint8))
    assert rows['masked_kl_reference_candidate'][0]==10000
    assert rows['candidate_probability_of_reference'][0]==0


@pytest.mark.parametrize('logits,masks', [([[0,0]],[[False,False]]),
    ([[float('nan'),0]],[[True,False]]), ([[0,0]],[[1,0]]), ([[0]],[[True,False]])])
def test_invalid_mask_or_legal_logits_rejected(logits,masks):
    with pytest.raises(ValueError):masked_log_probabilities(np.asarray(logits),np.asarray(masks))


def test_terminal_distance_does_not_cross_timeout_episode_or_chunk_boundary():
    terminated=np.zeros((7,2),dtype=bool);dones=terminated.copy()
    terminated[2,0]=True;dones[2,0]=True
    terminated[6,0]=True;dones[6,0]=True
    dones[2,1]=True  # Timeout: preceding rows must remain censored.
    terminated[5,1]=True;dones[5,1]=True
    expected=np.array([[2,-1],[1,-1],[0,-1],[3,2],[2,1],[1,0],[0,-1]])
    np.testing.assert_array_equal(terminal_distances(terminated,dones),expected)


def test_cache_reuses_completed_chunk_and_quarantines_partial(tmp_path):
    identity=dict(model='one',dataset='frozen',batch=256)
    rows=dict(worker=np.array([0,1]),action=np.array([4,5]))
    pending=tmp_path/'chunk.pending';pending.mkdir();(pending/'partial').write_text('preserve')
    commit_cache(tmp_path/'chunk',identity,rows,{'unchanged':True})
    assert list((tmp_path/'abandoned').glob('*/partial'))
    before={p.name:p.read_bytes() for p in (tmp_path/'chunk').iterdir()}
    cached=read_cache(tmp_path/'chunk',identity)
    np.testing.assert_array_equal(cached['action'],rows['action'])
    assert before=={p.name:p.read_bytes() for p in (tmp_path/'chunk').iterdir()}
    for changed in [dict(identity,model='other'),dict(identity,dataset='changed'),dict(identity,batch=8)]:
        with pytest.raises(ValueError,match='identity'):read_cache(tmp_path/'chunk',changed)
    with pytest.raises(FileExistsError):commit_cache(tmp_path/'chunk',identity,rows,{})
    (tmp_path/'chunk/rows.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='identity'):read_cache(tmp_path/'chunk',identity)


def test_real_forward_preserves_policy_adam_rng_and_predict_mode():
    import copy
    import torch
    from training.env.bitboard_candidate_env import BitboardCandidateTetrisEnv
    from training.candidate_policy import CandidateScoringPolicy
    from verify_feature_backend_benchmark import identical
    torch.set_num_threads(1)
    env=BitboardCandidateTetrisEnv(max_pieces=3)
    try:
        observation,_=env.reset(seed=42)
        observations={name:np.stack([value]*3) for name,value in observation.items()}
        masks=np.stack([env.action_masks()]*3)
        with torch.random.fork_rng(devices=[]):
            policy=CandidateScoringPolicy(env.observation_space,env.action_space,lambda _:3e-4)
            policy.set_training_mode(False)
            weights=copy.deepcopy(policy.state_dict());adam=copy.deepcopy(policy.optimizer.state_dict())
            rng=torch.get_rng_state().clone()
            tensors={name:torch.as_tensor(value) for name,value in observations.items()}
            observed=forward(policy,tensors,masks)
            predicted,_=policy.predict(observations,deterministic=True,action_masks=masks)
            np.testing.assert_array_equal(observed['actions'],predicted)
            assert torch.equal(rng,torch.get_rng_state())
            assert identical(weights,policy.state_dict()) and identical(adam,policy.optimizer.state_dict())
            compare_forward(observed,observed,masks,observations['candidate'])
    finally:env.close()
