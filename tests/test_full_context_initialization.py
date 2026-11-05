from copy import deepcopy
import hashlib
from importlib.metadata import version
import json

import pytest

from training.initialization import validate_initial_context, context_source_config


def specification(tmp_path):
    source = dict(ppo=dict(learning_rate=0.0003), run_seed=42, max_pieces=10000,
                  device='cuda', candidate_policy=True, policy='CandidateScoringPolicy',
                  task_steps=4096, reward={'game_over':-2}, reward_version='placement-reward-v1',
                  observation_version='candidate-v1', candidate_feature_version='features-v1')
    metadata = {key:version(package) for key,package in [('torch','torch'),('gymnasium','gymnasium'),
                  ('stable_baselines3','stable-baselines3'),('sb3_contrib','sb3-contrib')]}
    files = {'config.json':source,'metadata.json':metadata,'committed/task_000001/manifest.json':{}}
    for name, data in files.items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
    return dict(run_dir=str(tmp_path),task=1,files_sha256={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
                for name in files}), source


@pytest.mark.parametrize('conflict',[
    {'initial_model':{'path':'source'}}, {'initial_optimizer':'restore_state'},
    {'kl_reference':{'beta':0.3}}, {'failure_curriculum':{'mode':'normal'}},
    {'step0_baseline':True}, {'device':'cpu'}, {'n_envs':16}])
def test_full_context_rejects_incompatible_modes(tmp_path,conflict):
    spec,_=specification(tmp_path)
    preset=dict(device='cuda',n_envs=8,candidate_policy=True,initial_context=spec)
    assert validate_initial_context(preset)==spec
    with pytest.raises(ValueError):validate_initial_context(dict(preset,**conflict))


def test_full_context_rejects_changed_recipe_before_restore(tmp_path,monkeypatch):
    spec,config=specification(tmp_path)
    import training.train_vector_transaction as trainer
    calls=[]
    monkeypatch.setattr(trainer,'verify_task',lambda *args:calls.append(args))
    assert context_source_config(spec,config)==config
    changed=deepcopy(config);changed['ppo']['learning_rate']=3e-5
    with pytest.raises(ValueError,match='ppo'):context_source_config(spec,changed)
    assert len(calls)==1


@pytest.mark.parametrize('name',['config.json','metadata.json','committed/task_000001/manifest.json'])
def test_full_context_rejects_changed_source_files(tmp_path,name):
    spec,config=specification(tmp_path)
    with (tmp_path/name).open('a') as stream:stream.write(' ')
    with pytest.raises(RuntimeError,match='Frozen full context changed'):context_source_config(spec,config)
