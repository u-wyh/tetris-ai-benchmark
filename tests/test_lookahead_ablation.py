import copy
import pickle
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from training.evaluation.hybrid_queue import BC
from training.public_lookahead import PublicLookaheadController, PublicSimulator
from training.lookahead_ablation_controller import AblationController, legacy_trace, SPECS
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv
import training.lookahead_ablation as study
from training.lookahead_ablation_report import paired

@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(1)
    return MaskablePPO.load(study.STUDENT,device='cpu'),MaskablePPO.load(BC,device='cpu')

def test_full_original_equivalence_and_fixed_ablation_rules(models):
    student,bc=models;old=PublicLookaheadController(student,bc)
    env=LongCandidateTetrisEnv(max_pieces=18);env.reset(seed=53)
    for step in range(18):
        obs,mask=env.get_observation(),env.action_masks();pub=env.core.get_public_observation()
        before=pickle.dumps(env.core);original,trace=old.decide(obs,mask,pub)
        for name,(roots,depth) in SPECS.items():
            controller=AblationController(student,bc,name);action,t=controller.decide(obs,mask,pub)
            assert mask[action]
            if name=='E':assert action==original and legacy_trace(t)==trace
            if name=='A':assert action==trace['student_action'] and not t['alternatives']
            if name in ('B','C'):assert t['roots']==list(dict.fromkeys((t['student_action'],t['bc_action'])))
            for branch in t['alternatives']:
                assert len(branch['path'])<=depth and branch['known_tokens_consumed']<=3
                assert branch['value']==branch['final_geometry']+.25*branch['max_geometry']+branch['score_contribution']+branch['death_penalty']
            if step%6==0:
                other=copy.deepcopy(env.core);other.rng.state^=0x87654321;other.bag=list('ZZZZZZZ');other.queue[3:]=list('II')
                assert (action,t)==controller.decide(obs,mask,other.get_public_observation())
            study.validate_rollouts(env.core,t)
        assert pickle.dumps(env.core)==before
        env.step(original)
    with pytest.raises(ValueError):AblationController(student,bc,'F')
    with pytest.raises(ValueError,match='Next3'):AblationController(student,bc,'E').decide(obs,mask,{**pub,'next':list('IJLOS')})
    env.close()

def test_hold_consumes_only_known_tokens_and_stops_at_unknown(models):
    env=LongCandidateTetrisEnv(max_pieces=10);env.reset(seed=53)
    public=env.core.get_public_observation();legal=env.core.get_legal_placements()
    hold=next(p for p in legal if p['hold']);sim=PublicSimulator(public);real=copy.deepcopy(env.core)
    pred=sim.step(hold['actionId']);actual=real.step(hold['actionId'])
    assert sim.consumed==2 and pred['board']==actual['board'] and pred['hold']==actual['hold']
    while sim.phase=='playing':
        choices=sim.get_legal_placements();assert choices
        sim.step(choices[0]['actionId'])
    assert sim.phase=='unknown_frontier' and sim.consumed==3 and sim.current is None
    obs,mask=env.get_observation(),env.action_masks();bad=mask.copy();bad[np.flatnonzero(mask)[0]]=False
    with pytest.raises(RuntimeError,match='mask'):AblationController(*models,'E').decide(obs,bad,public)

def test_atomic_resume_preserves_actions_scores_and_component_arrays(models,tmp_path,monkeypatch):
    monkeypatch.setattr(study,'MODELS',{'A':models[0],'BC':models[1]});monkeypatch.setattr(study,'IDENTITY','unit_identity')
    monkeypatch.setattr(study,'CHECKPOINT_INTERVAL',2);monkeypatch.setattr(study,'guard',lambda:None)
    monkeypatch.setattr(study,'RUN',tmp_path/'direct')
    direct=study.game('E',53,8,'unit')
    monkeypatch.setattr(study,'RUN',tmp_path/'resumed');save=study.save_blob;count=0
    def interrupt(path,value):
        nonlocal count
        save(path,value);count+=1
        if count==2:raise RuntimeError('simulated process failure after atomic checkpoint')
    monkeypatch.setattr(study,'save_blob',interrupt)
    with pytest.raises(RuntimeError,match='simulated'):study.game('E',53,8,'unit')
    monkeypatch.setattr(study,'save_blob',save)
    resumed=study.game('E',53,8,'unit')
    from pathlib import Path
    with np.load(Path(direct).with_suffix('.npz')) as a,np.load(Path(resumed).with_suffix('.npz')) as b:
        for key in ('scores','lines','codes','root_codes','costs'):np.testing.assert_array_equal(a[key],b[key])
    row=study.completed(Path(resumed),'E',53,8);assert row['pieces']==8 and row['capped'] and not row['game_over']
    study.game('E',53,8,'unit')  #Completed jobs validate/reuse, never rerun.

def test_paired_bootstrap_uses_complete_seed_intersection_and_censoring():
    def row(n,score,dead=False):return dict(pieces=n,scores=np.linspace(0,score,n+1),game_over=dead)
    left={1:row(20,100),2:row(20,100),3:row(12,80,True)}
    right={1:row(20,110),2:row(20,110),4:row(20,200)}
    r=paired(left,right)
    assert r['games']==2 and r['positive_seeds']==2 and r['common_score_ratio']==1.1
    assert r['observed_survival_delta']==0 and r['survival_difference_ci95']==[0,0]
    assert [p['seed'] for p in r['pairs']]==[1,2] and not r['replicated_score_support']
