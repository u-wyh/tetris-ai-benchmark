"""Pin the intervention axes and avoid comparing extra post-death reward."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_rank_continuation_factorial import compatible_protocols,contrasts,validate_pair,ROOT


def protocols():
    models=dict(reference=dict(path='original',sha256='old'),candidate=dict(path='candidate',sha256='new'))
    base=dict(contexts_dir='contexts',training_sources={'one':'same'},inference_sources={'two':'same'},
        evaluation_runtime={'torch':'fixed'},native_feature_runtime={'sha256':'fixed'},horizon=256,n_envs=8,
        context_count=15,first_clear_expectations=[4,0],grouping='fixed',
        frozen_files={str(ROOT/p):'same' for p in ['scripts/run_rank_branches.py','scripts/rank_branch_helpers.py']})
    return dict(base,model=models['reference']),dict(base,model=models['candidate']),models


@pytest.mark.parametrize('field', ['contexts_dir','horizon','n_envs','training_sources','native_feature_runtime','grouping'])
def test_changed_protocol_axis_rejected(field):
    left,right,models=protocols();compatible_protocols(left,right,models)
    right[field]='changed'
    with pytest.raises(ValueError,match='protocols'):compatible_protocols(left,right,models)


def test_wrong_model_or_execution_code_rejected():
    left,right,models=protocols();right['model']=models['reference']
    with pytest.raises(ValueError,match='model'):compatible_protocols(left,right,models)
    left,right,models=protocols();right=copy.deepcopy(right)
    right['frozen_files'][str(ROOT/'scripts/run_rank_branches.py')]='changed'
    with pytest.raises(ValueError,match='source'):compatible_protocols(left,right,models)


def outcome(rewards,clears,dead=False):
    trace=[dict(reward=r,cleared_lines=c,action=4,pieces=i+1,terminated=dead and i==len(rewards)-1,truncated=False)
           for i,(r,c) in enumerate(zip(rewards,clears))]
    return dict(summary=dict(steps=len(trace),reward=sum(rewards),cleared_lines=sum(clears),
        game_over=dead,time_limit=False,horizon_censored=not dead),trace=trace)


def test_factorial_holds_first_action_fixed_and_excludes_extra_reference_steps():
    old=dict(reference=outcome([1,1,100],[4,0,4]),candidate=outcome([1,1,200],[0,0,4]))
    new=dict(reference=outcome([1,-2],[4,0],True),candidate=outcome([2,-2],[0,0],True))
    rows=contrasts(old,new)
    assert rows['reference_first_action']['comparison']['common_reward_delta']==-3
    assert rows['candidate_first_action']['comparison']['common_reward_delta']==-2
    assert rows['reference_first_action']['comparison']['common_cleared_delta']==0
    assert rows['reference_first_action']['comparison']['candidate_observed_earlier_death']


@pytest.mark.parametrize('mutation',[None,'forced','counts','reset','censoring'])
def test_pair_trace_and_audit_validation(mutation):
    target=dict(reference_action=4,candidate_action=4)
    manifest=dict(identity=dict(group=0,group_keys=['one'],target=target,context_sha256='context',
        model={'sha256':'model'},torch_float32_matmul_precision='highest',tf32=False))
    data=dict(reference=outcome([1,-2],[4,0],True),candidate=outcome([1,-2],[0,0],True),
        model_audit=dict(policy_adam_counts_unchanged=True,cpu_cuda_sampling_rng_unchanged=True,
                         num_timesteps=8192,ppo_updates=36635))
    if mutation=='forced':data['reference']['trace'][0]['action']=5
    if mutation=='counts':data['model_audit']['num_timesteps']=0
    if mutation=='reset':data['candidate']['trace'][1]['pieces']=1
    if mutation=='censoring':data['candidate']['summary']['horizon_censored']=True
    args=(manifest,data,0,['one'],target,'context',{'sha256':'model'},[8192,36635],256)
    if mutation is None:validate_pair(*args)
    else:
        with pytest.raises(ValueError):validate_pair(*args)
