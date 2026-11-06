"""Inactive/dead lanes, horizon censoring and immutable fixed-group pair commits."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rank_branch_helpers import BranchCompactEnv,run_group,compare_pair
from restore_rank_contexts import observation_digest
from run_rank_branches import save_pair,read_pair
from training.tetris_core.pieces import make_piece


class LocalVector:
    def __init__(self,width,cap):
        self.num_envs=width;self.envs=[BranchCompactEnv(max_pieces=cap) for _ in range(width)]
    def env_method(self,name,*args,indices=None):
        return [getattr(self.envs[i],name)(*args) for i in (range(self.num_envs) if indices is None else indices)]
    def close(self):
        for e in self.envs:e.close()


class GreedyRecorder:
    def __init__(self):self.batch_sizes=[]
    def predict(self,observations,deterministic,action_masks):
        assert deterministic is True and len(observations['state'])==len(action_masks)
        self.batch_sizes.append(len(action_masks))
        return np.argmax(action_masks,axis=1),None


def specification(cap):
    source=BranchCompactEnv(max_pieces=cap)
    try:
        obs,_=source.reset(seed=42);mask=source.action_masks();action=int(np.flatnonzero(mask)[-1])
        return dict(state=source.get_state(),digest=observation_digest(obs,mask),actions=[action,action])
    finally:source.close()


@pytest.mark.parametrize('cap,horizon,steps,censored,time_limit',[(2,5,2,False,True),(20,3,3,True,False)])
def test_group_forces_first_action_stops_dead_and_padding_and_preserves_episode(cap,horizon,steps,censored,time_limit):
    env=LocalVector(4,cap);policy=GreedyRecorder();spec=specification(cap)
    try:
        pairs,work=run_group(env,[spec],policy,horizon)
        assert work['environment_steps']==steps*2
        assert policy.batch_sizes==[2]*(steps-1)
        for name in ['reference','candidate']:
            result=pairs[0][name]
            assert result['trace'][0]['action']==spec['actions'][0]
            assert result['summary']['horizon_censored']==censored
            assert result['summary']['time_limit']==time_limit
            assert result['summary']['episode_count']==spec['state']['episode_count']
        assert env.envs[2].core is None and env.envs[3].core is None
        assert env.envs[2].branch_step_active(None) is None
        assert env.envs[0].branch_step_active(None) is None
        assert env.envs[0].pieces==steps
        assert compare_pair(pairs[0])['common_reward_delta']==0
    finally:env.close()


def test_lockout_finishes_without_reset_or_additional_step():
    env=BranchCompactEnv(max_pieces=20)
    try:
        env.reset(seed=42);env.core.current=make_piece('I');env.core.current['y']=-2
        for x in range(3,7):env.core.board[0][x]='Z'
        env._invalidate_candidates()
        placement=next(p for p in env.core.get_legal_placements() if all(c['y']<0 for c in p['occupiedCells']))
        state=env.get_state();env.branch_activate(state)
        result=env.branch_step_active(placement['actionId'])
        assert result['row']['terminated'] and result['view'] is None
        assert env.branch_step_active(placement['actionId']) is None
        final=env.branch_finish(5)
        assert final['summary']['steps']==1 and final['summary']['game_over']
        assert not final['summary']['horizon_censored'] and not final['summary']['time_limit']
        assert final['state']['episode_count']==state['episode_count']
    finally:env.close()


def test_pair_cache_replay_compares_values_without_overwriting_and_pins_group(tmp_path):
    env=LocalVector(2,20)
    try:
        pair=run_group(env,[specification(20)],GreedyRecorder(),3)[0][0]
        identity=dict(model='frozen',group_keys=['first','second'],horizon=3)
        audit=dict(policy_adam_counts_unchanged=True)
        path=tmp_path/'pair';assert save_pair(path,identity,pair,audit)
        before={p.name:p.read_bytes() for p in path.iterdir()}
        assert not save_pair(path,identity,pair,audit)
        assert before=={p.name:p.read_bytes() for p in path.iterdir()}
        with pytest.raises(ValueError,match='identity'):read_pair(path,dict(identity,group_keys=['other']))
        changed=copy.deepcopy(pair);changed['candidate']['trace'][0]['reward']+=1
        with pytest.raises(ValueError,match='Fixed-group replay'):save_pair(path,identity,changed,audit)
        (path/'states.pkl').write_bytes(b'corrupted')
        with pytest.raises(ValueError,match='checksum'):read_pair(path,identity)
    finally:env.close()


def test_common_window_does_not_compare_extra_survival_reward():
    row=lambda reward,clears:dict(reward=reward,cleared_lines=clears)
    pair=dict(reference=dict(summary=dict(steps=3,game_over=False),trace=[row(1,4),row(1,0),row(100,4)]),
              candidate=dict(summary=dict(steps=2,game_over=True),trace=[row(1,0),row(-2,0)]))
    result=compare_pair(pair)
    assert result['common_steps']==2 and result['common_reward_delta']==-3
    assert result['common_cleared_delta']==-4 and result['candidate_observed_earlier_death']
