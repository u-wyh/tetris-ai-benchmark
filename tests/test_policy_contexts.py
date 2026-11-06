"""Natural first action, actual worker restart and pre-write input/cache rejection."""
import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import policy_contexts as contexts
from policy_contexts import prepare_run,activate_states,natural_round
from own_policy_capture import CaptureCompactEnv,save_commit,sha
from restore_own_death_contexts import restore_vector
from restore_rank_contexts import observation_digest
from verify_feature_backend_benchmark import identical
from training.vector_env import make_vector_env
from training.tetris_core.pieces import make_piece


def test_natural_first_action_and_restart_preserve_prefix_state_and_true_stopped_lane(monkeypatch):
    batches=[]
    def choose_last(policy,observations,masks,executed=None):
        actions=masks.shape[1]-1-masks[:,::-1].argmax(axis=1);batches.append(len(actions))
        if len(actions)==2:actions[1]=placement['actionId']
        return dict(action=actions,probability=np.ones(len(actions)),margin=np.ones(len(actions)),
                    probability_of_executed=np.ones(len(actions)))
    monkeypatch.setattr(contexts,'greedy_details',choose_last)
    first=CaptureCompactEnv(max_pieces=4);second=CaptureCompactEnv(max_pieces=4)
    try:
        view=first.capture_reset(100005)
        first.capture_step(int(view['mask'].argmax()))
        second.capture_reset(100006);second.core.current=make_piece('I');second.core.current['y']=-2
        for x in range(3,7):second.core.board[0][x]='Z'
        second._invalidate_candidates()
        placement=next(p for p in second.core.get_legal_placements() if all(c['y']<0 for c in p['occupiedCells']))
        # Both fixtures start after an existing episode prefix; first action must still come from the policy.
        states=[first.get_state(),second.get_state()]
        digests=[observation_digest(first.get_observation(),first.action_masks()),
                 observation_digest(second.get_observation(),second.action_masks())]
    finally:first.close();second.close()
    def vector():return make_vector_env(2,max_pieces=4,env_class=CaptureCompactEnv)
    env=vector()
    try:
        views=activate_states(env,states,digests)
        expected_last=int(np.flatnonzero(views[0]['mask'])[-1])
        rows=natural_round(env,views,None,0)
        assert rows[0]['action']==expected_last and rows[0]['local_action_step']==1 and rows[0]['pieces']==2
        assert batches==[2]
        assert rows[1]['terminated'] and views[1] is None
        saved=dict(workers=env.env_method('capture_export'),views=[None if v is None else observation_digest(**v) for v in views])
        expected=natural_round(env,views,None,1);expected_states=env.env_method('capture_export')
    finally:env.close()
    env=vector()
    try:
        views=restore_vector(env,saved)
        resumed=natural_round(env,views,None,1)
        assert identical(expected,resumed) and identical(expected_states,env.env_method('capture_export'))
        assert [row['lane'] for row in resumed]==[0]
        assert env.env_method('capture_export',indices=[1])[0]['state']['episode_count']==1
        assert batches[-1]==1
    finally:env.close()


def fake_job(tmp_path):
    job=tmp_path/'job';job.mkdir();inputs=tmp_path/'inputs';inputs.mkdir()
    model=inputs/'model.zip';model.write_bytes(b'fixed checkpoint')
    entries=[]
    for index in range(8):
        directory=inputs/f'context{index}';directory.mkdir()
        (directory/'state.pkl').write_bytes(b'own frozen state, not loaded by preflight')
        expected=dict(target=dict(key=str(index)))
        digest=dict(observation='o'*64,mask='m'*64)
        metadata=dict(identity=expected,state_sha256=sha(directory/'state.pkl'),observation_digest=digest)
        (directory/'manifest.json').write_text(json.dumps(metadata))
        entries.append(dict(key=str(index),directory=str(directory),expected_identity=expected,
            manifest_sha256=sha(directory/'manifest.json'),state_sha256=metadata['state_sha256'],
            observation_digest=digest,initial_pieces=6400,episode_seed=100005))
    suite=inputs/'suite.json';suite.write_text(json.dumps(dict(version=1,n_envs=8,horizon=512,max_pieces=20000,contexts=entries)))
    design=dict(frozen_files={},checkpoint=dict(path=str(model),sha256=sha(model)),suite=dict(path=str(suite),sha256=sha(suite)))
    (job/'design.json').write_text(json.dumps(design))
    return job,design,model,suite


@pytest.mark.parametrize('kind',['model','suite','context'])
def test_wrong_external_inputs_are_rejected_before_pickle_or_job_output(tmp_path,monkeypatch,kind):
    job,design,model,suite=fake_job(tmp_path)
    if kind=='model':model.write_bytes(b'changed')
    elif kind=='suite':suite.write_text('{}')
    else:
        entry=json.loads(suite.read_text())['contexts'][0]
        (Path(entry['directory'])/'state.pkl').write_bytes(b'changed')
    before={p.name:p.read_bytes() for p in job.iterdir()}
    monkeypatch.setattr(pickle,'loads',lambda _:pytest.fail('Unpickle before input rejection'))
    with pytest.raises(ValueError):prepare_run(job)
    assert before=={p.name:p.read_bytes() for p in job.iterdir()}


def test_valid_new_checkpoint_with_existing_cache_cannot_change_identity_or_prefix(tmp_path,monkeypatch):
    job,design,model,suite=fake_job(tmp_path)
    _,_,identity,_,_,_=prepare_run(job)
    save_commit(job,0,identity,None,dict(round=0,environment_steps=0),[],{})
    before={str(p.relative_to(job)):p.read_bytes() for p in job.glob('chunk_0000/*')}
    model.write_bytes(b'another valid checkpoint');design['checkpoint']['sha256']=sha(model)
    (job/'design.json').write_text(json.dumps(design))
    monkeypatch.setattr(pickle,'loads',lambda _:pytest.fail('Wrong cache identity reached pickle'))
    with pytest.raises(ValueError,match='identity'):prepare_run(job)
    assert before=={str(p.relative_to(job)):p.read_bytes() for p in job.glob('chunk_0000/*')}
    assert not (job/'evaluation.lock').exists()
