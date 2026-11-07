import copy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_topk_preference_evidence as audit
from training.candidate_features import FEATURE_VERSION, candidate_features
from training.tetris_core import TetrisCore


def branch(n, score=100, fours=10, dead=True):
    return dict(pieces=n, initial_score=0, scores=[score]*n,
        four_counts=[min(i+1, fours) for i in range(n)], game_over=dead, censored=not dead, score_gain=score)


@pytest.mark.parametrize('alt_n,score,fours,base_fours,label', [
    (750,95,9,10,'beneficial'), (749,100,10,10,'unsupported_within_window'),
    (750,94,10,10,'unsupported_within_window'), (750,100,8,10,'unsupported_within_window'),
    (750,95,0,2,'beneficial')])
def test_original_strong_boundaries_without_retuning(alt_n,score,fours,base_fours,label):
    result, evidence = audit.preference(branch(500,fours=base_fours),branch(alt_n,score,fours))
    assert result==label
    assert evidence['common_pieces']==500


def test_harmful_is_reverse_strong_not_merely_nonpositive():
    assert audit.preference(branch(750),branch(500))[0]=='harmful'
    assert audit.preference(branch(500),branch(600))[0]=='unsupported_within_window'
    assert audit.preference(branch(5000,dead=False),branch(5000,dead=False))[0]=='both_right_censored_uncertain'


@pytest.mark.parametrize('change', ['censor_short','both_flags','length','score_decrease','fours_jump','gain'])
def test_invalid_or_inconsistent_trajectory_rejected(change):
    b=branch(5000,dead=False)
    if change=='censor_short':b=branch(30,dead=False)
    elif change=='both_flags':b['game_over']=True
    elif change=='length':b['scores'].pop()
    elif change=='score_decrease':b['scores'][10]=1
    elif change=='fours_jump':b['four_counts'][0]=2
    else:b['score_gain']=0
    with pytest.raises(ValueError):audit.validate_branch(b,5000)


def test_dedup_dual_sources_and_cross_window_do_not_inflate_labels():
    r=dict(seed=7,public_hash='x',action=2,horizon=5000,sources=['ppo_top_2'],label='beneficial')
    bc=dict(r,sources=['bc']); short=dict(r,horizon=1000)
    unique,duplicates,conflicts=audit.deduplicate([r,bc,short])
    assert duplicates==1 and not conflicts and len(unique)==1
    assert unique[0]['sources']==['bc','ppo_top_2']


def test_conflicting_public_contexts_are_not_silently_selected():
    r=dict(seed=7,public_hash='x',action=2,horizon=5000,sources=['ppo_top_2'],snapshot_sha256='a',label='beneficial')
    other=dict(r,snapshot_sha256='b',label='harmful')
    unique,_,conflicts=audit.deduplicate([r,other])
    assert not unique and len(conflicts)==1


def test_historical_extension_is_conditioned_and_max_two_alternatives():
    choices=[dict(action=i,sources=['ppo_top_'+str(i+1)]) for i in range(4)]
    candidate=dict(ppo_action=0,choices=choices)
    initial={0:branch(1000,dead=False),1:branch(1000,110,dead=False),
             2:branch(1000,120,dead=False),3:branch(1000,105,dead=False)}
    assert audit.extended_choices(candidate,initial,900)==[]
    assert [c['action'] for c in audit.extended_choices(candidate,initial,2000)]==[0,2,1]


@pytest.mark.parametrize('seal', ['external','historical'])
def test_bad_snapshot_seal_rejected_before_pickle(tmp_path,monkeypatch,seal):
    path=tmp_path/'state.pkl.gz';path.write_bytes(gzip.compress(b'not a pickle'))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    frozen={str(path):digest if seal=='historical' else '0'*64}
    meta=dict(sha256='0'*64 if seal=='historical' else digest)
    def forbidden(*a):pytest.fail('Bad seal reached pickle')
    monkeypatch.setattr(pickle,'loads',forbidden)
    with pytest.raises(ValueError):audit.load_public(path,meta,dict(snapshot_sha256=digest),{},frozen)


def test_public_reconstruction_has_only_current_public_shapes_no_steps(tmp_path):
    core=TetrisCore(7);public=core.get_public_observation();_,mask=candidate_features(public)
    action=int(np.flatnonzero(mask)[0])
    state=dict(core=core,episode_seed=7,pieces=0,episode_count=1,max_pieces=100000,finished=False,candidate_feature_version=FEATURE_VERSION)
    path=tmp_path/'state.pkl.gz';path.write_bytes(gzip.compress(pickle.dumps(state,protocol=5)))
    digest=hashlib.sha256(path.read_bytes()).hexdigest();ph=audit.public_hash(public)
    meta=dict(sha256=digest,rng_state=core.rng.state,public_hash=ph)
    candidate=dict(snapshot_sha256=digest,public_hash=ph,start_score=public['score'],choices=[dict(action=action)],ppo_action=action)
    before=copy.deepcopy(core.__dict__)
    vector,matrix,actual_mask=audit.load_public(path,meta,candidate,dict(seed=7,step=0),{str(path):digest})
    assert vector.shape==(237,) and matrix.shape==(1840,16) and actual_mask.shape==(1840,)
    assert vector.dtype==np.float32 and matrix.dtype==np.uint8 and actual_mask.dtype==np.bool_
    assert audit.public_hash(core.get_public_observation())==ph and core.rng.state==before['rng'].state


def test_explicit_wrong_model_never_falls_back_to_suite_declaration(tmp_path):
    source=tmp_path;folder=source/'counterfactual_5000/seed_7';folder.mkdir(parents=True)
    path=folder/'step_3_action_2.json';choice=dict(action=2,sources=['ppo_top_2'])
    candidate=dict(snapshot_sha256='a'*64,public_hash='b'*64,start_score=0)
    value=dict(branch(5000,dead=False),seed=7,step=3,first_action=2,horizon=5000,
        snapshot_sha256='a'*64,initial_public_hash='b'*64,choice=choice,model_sha256={'ppo':'wrong'})
    path.write_text(json.dumps(value));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    seal=path.with_suffix('.sha256.json');seal.write_text(json.dumps(dict(sha256=digest,snapshot_sha256='a'*64)))
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,seal]}
    with pytest.raises(ValueError,match='model differs'):
        audit.read_branch(source,dict(seed=7,step=3),candidate,choice,5000,frozen,{'ppo':'expected'})
