import copy
import random

import numpy as np
import pytest
import torch

import training.bc_sequence_safety as s


def test_preplacement_public_risk_and_top_space():
    state=np.zeros(237,dtype=np.float32)
    assert s.visible(state)['tier']=='normal'
    state[80]=1
    assert s.visible(state)['height']==12
    assert s.visible(state)['tier']=='severe'
    state[0]=1
    assert s.visible(state)['spawn']==1
    # Preview/hold changes cannot alter the visible-board risk function.
    expected=s.visible(state);state[200:]=7
    assert s.visible(state)==expected


def test_teacher_survival_alone_is_not_recovery():
    a=dict(death=False,rows=[dict(height=3,holes=0)]*256)
    b=dict(death=False,rows=[dict(height=5,holes=0)]*256)
    risk=dict(height=12,holes=0)
    assert s.qualifies(risk,a,b)
    high=dict(a,rows=[dict(height=9,holes=0)]*256)
    assert not s.qualifies(risk,high,b)
    assert not s.qualifies(risk,dict(a,death=True),b)
    assert not s.qualifies(risk,dict(a,rows=a['rows'][:255]),b)
    bad_holes=dict(a,rows=[dict(height=3,holes=8)]*256)
    assert not s.qualifies(risk,bad_holes,b)


def test_gate_requires_actual_survival_and_broad_score_gain():
    base=dict(games=32,mean_pieces=20000,p10=20000,early=0,capped=32,mean_score=100,
              high_fraction=0,new_holes_per_100=2,mean_max_high_streak=0)
    candidate=dict(base,capped=31,mean_pieces=19900)
    result=dict(summary=dict(bc=base,candidate=candidate),paired=dict(candidate=dict(common_score_ratio=1.09,positive_seeds=26,ratio_ci95=[1.08,1.1])))
    assert s.gate(result)['passed']
    result['summary']['candidate']['early']=1
    assert not s.gate(result)['passed']
    result['summary']['candidate']['early']=0
    result['paired']['candidate']['positive_seeds']=25
    assert not s.gate(result)['passed']
    result['paired']['candidate']['positive_seeds']=26
    result['summary']['candidate']['mean_score']=99
    assert not s.gate(result,long=True)['passed']


def test_long_gate_rejects_score_gain_concentrated_in_one_seed():
    base=dict(games=16,mean_pieces=100000,p10=100000,early=0,capped=16,mean_score=100,
              high_fraction=0,new_holes_per_100=2,mean_max_high_streak=0)
    pairs=[dict(seed=i,baseline_score=100,candidate_score=109) for i in range(16)]
    result=dict(summary=dict(bc=base,candidate=dict(base,mean_score=109)),paired=dict(candidate=dict(
        common_score_ratio=1.09,positive_seeds=16,ratio_ci95=[1.08,1.1],pairs=pairs)))
    assert s.gate(result,long=True)['passed']
    for pair in pairs:pair['candidate_score']=101
    pairs[0]['candidate_score']=400
    result['paired']['candidate']['common_score_ratio']=sum(p['candidate_score'] for p in pairs)/1600
    assert not s.gate(result,long=True)['checks']['trimmed_score']


def test_real_updates_atomic_optimizer_rng_restore(tmp_path,monkeypatch):
    if not s.STUDENT.exists():pytest.skip('Frozen student unavailable')
    monkeypatch.setattr(s,'RUN',tmp_path);s.record(tmp_path/'protocol.json',dict(test=True));s.record(tmp_path/'dataset.json',dict(test=True))
    torch.set_num_threads(1);model=s.MaskablePPO.load(s.STUDENT,device='cpu');opt=torch.optim.Adam(s.actor(model),lr=1e-5)
    env=s.old.LongCandidateTetrisEnv(max_pieces=10);env.reset(seed=93);f=s.public_sample(env)
    f['action']=int(f['legal_ids'][0]);before={n:v.clone() for n,v in model.policy.state_dict().items()}
    for _ in range(2):
        logits,_,_=s.logits(model.policy,[f],'cpu');loss=-logits[0,int(f['legal_ids'][-1])]
        opt.zero_grad();loss.backward();opt.step()
    assert any(not torch.equal(v,before[n]) for n,v in model.policy.state_dict().items() if n.startswith('score_head.'))
    path=s.save_checkpoint(model,opt,dict(updates=2),'smoke')
    expected=random.random();loaded,optimizer,state=s.load_checkpoint(path,'cpu')
    assert random.random()==expected
    assert state['updates']==2 and set(int(v['step']) for v in optimizer.state.values())=={2}
    assert all(torch.equal(v,loaded.policy.state_dict()[n]) for n,v in model.policy.state_dict().items())
    assert all(torch.equal(v,before['value_net.'+n]) for n,v in model.policy.value_net.state_dict().items())
    env.close()
