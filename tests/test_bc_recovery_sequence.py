import copy
from pathlib import Path
import numpy as np
import pytest
import training.bc_recovery_sequence as r
from training.bc_recovery_sequence_train import gate

def test_disk_accounting_tolerates_atomic_temporary_rename(tmp_path):
    existing=tmp_path/'real';existing.write_bytes(b'abc')
    class Disappeared:
        def stat(self):raise FileNotFoundError('atomic temporary rename')
    assert r.retained_bytes([Disappeared(),existing,tmp_path])==3

def branch(n,death,divergence=True):
    return dict(rows=[dict(action=1)]*n,death=death,frames=[dict(action=1,student_action=2 if divergence else 1)])

def test_actual_death_required_not_board_recovery_or_censoring():
    b={k:branch(3000,False) for k in 'ABCD'}
    c=r.classify(b)
    assert not c['strong'] and c['ambiguous']==['B','C','D']
    b['A']=branch(2900,True)
    assert not r.classify(b)['strong']  #100 observed pieces is insufficient.
    b['A']=branch(2744,True)
    assert r.classify(b)['strong']==['B','C','D']
    b['C']['frames'][0]['student_action']=1
    assert r.classify(b)['strong']==['B','D']

def test_multistep_first_same_and_harm_are_separate():
    b=dict(A=branch(1000,True),B=branch(1000,True,False),C=branch(3000,False),D=branch(500,True))
    c=r.classify(b)
    assert c['selected']=='C' and c['multistep'] and c['first_same']
    assert c['harm']==['D']

def test_gate_freezes_score_survival_and_risk():
    base=dict(games=32,mean_pieces=20000,p10=20000,early=0,capped=32,high_fraction=0,new_holes_per_100=2,mean_max_high_streak=0)
    result=dict(summary=dict(bc=base,candidate=dict(base)),paired=dict(candidate=dict(common_score_ratio=1.10,positive_seeds=26)))
    assert gate(result)['passed']
    result['paired']['candidate']['common_score_ratio']=1.099
    assert not gate(result)['passed']
    result['paired']['candidate']['common_score_ratio']=1.10
    result['summary']['candidate']['early']=1
    assert not gate(result)['passed']

def test_same_snapshot_window_handoff_and_atomic_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'RUN',tmp_path);monkeypatch.setattr(r,'WINDOW',32);monkeypatch.setattr(r,'guard',lambda:None)
    r.init('smoke-only')
    env=r.old.LongCandidateTetrisEnv(max_pieces=20000);env.reset(seed=53)
    point=tmp_path/'source.pkl'
    r.save_blob(point,dict(identity='smoke-only',seed=53,piece=0,risk=r.visible(env.get_observation()['state']),env=env.get_state(),rng=r.rng_state()))
    expected=[]
    for _ in range(8):
        row,over,capped=r.old.move(env,'student');expected.append((row['action'],row['score']))
    path=r.counterfactual(point);result=r.read_blob(path)
    assert [(v['action'],v['score']) for v in result['branches']['A']['rows'][:8]]==expected
    for name,length in r.BRANCHES.items():
        b=result['branches'][name]
        assert len(b['rows'])==32 and b['right_censored'] and not b['death']
        assert all(row['control']==('bc' if index<length else 'student') for index,row in enumerate(b['rows']))
        assert len(b['frames'])==min(length,32)
        assert all(f['action'] in f['legal_ids'] for f in b['frames'])
        assert b['env']['max_pieces']==32
    checksum=r.digest(Path(path))
    assert r.counterfactual(point)==path and r.digest(Path(path))==checksum
    #Resume a completed-A partial job from the exact saved B trajectory state.
    partial=copy.deepcopy(result);partial['branches']={'A':partial['branches']['A']}
    b=result['branches']['B'];r.save_blob(__import__('pathlib').Path(path).with_suffix('.checkpoint.pkl'),dict(result=partial,name='B',rows=b['rows'],frames=b['frames'],env=b['env'],rng=b['rng']))
    __import__('pathlib').Path(path).unlink();resumed=r.read_blob(r.counterfactual(point))
    assert {k:[(v['action'],v['score']) for v in b['rows']] for k,b in result['branches'].items()}=={k:[(v['action'],v['score']) for v in b['rows']] for k,b in resumed['branches'].items()}
