"""Pure validation checks: no numerical library initialization or training."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from training.initialization import context_source_config, validate_context_model_recipe


@pytest.mark.parametrize('source_patch',[
    {'initial_context':{'task':7325}}, {'source_steps_offset':30003200},
    {'source_steps_offset':-4096}])
def test_nested_context_is_rejected_before_loading_runtime_or_checkpoint(tmp_path,source_patch):
    source=dict(task_steps=4096,**source_patch)
    (tmp_path/'config.json').write_text(json.dumps(source))
    (tmp_path/'metadata.json').write_text('{}')
    spec=dict(run_dir=str(tmp_path),task=1,files_sha256={
        'config.json':hashlib.sha256((tmp_path/'config.json').read_bytes()).hexdigest()})
    with pytest.raises(ValueError,match='Nested context offsets'):
        context_source_config(spec,{})


def test_recorded_offset_must_match_source_task_before_checkpoint_loading(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(task_steps=4096)))
    (tmp_path/'metadata.json').write_text('{}')
    spec=dict(run_dir=str(tmp_path),task=3,files_sha256={
        'config.json':hashlib.sha256((tmp_path/'config.json').read_bytes()).hexdigest()})
    with pytest.raises(ValueError,match='step offset differs'):
        context_source_config(spec,dict(source_steps_offset=4096))


def model_and_config():
    ppo=dict(learning_rate=0.0003,gamma=0.995,gae_lambda=0.95,clip_range=0.2,
             n_steps=512,n_envs=8,batch_size=256,n_epochs=5,ent_coef=0.01,
             vf_coef=0.5,max_grad_norm=0.5)
    # Match the public representation without importing torch/SB3 in these tests.
    ConstantSchedule=type('ConstantSchedule',(),{'__module__':'stable_baselines3.common.utils'})
    FloatSchedule=type('FloatSchedule',(),{'__module__':'stable_baselines3.common.utils'})
    def schedule(value):
        result=FloatSchedule();result.value_schedule=ConstantSchedule();result.value_schedule.val=value
        return result
    fields={k:v for k,v in ppo.items() if k!='clip_range'}
    model=SimpleNamespace(**fields,clip_range=schedule(0.2),lr_schedule=schedule(0.0003),
        normalize_advantage=True,clip_range_vf=None,target_kl=None,seed=42,
        policy=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{'lr':0.0003}])))
    return model,dict(ppo=deepcopy(ppo),run_seed=42)


@pytest.mark.parametrize('name,value',[
    ('gamma',0.999),('learning_rate',0.00003),('n_envs',16),
    ('seed',43),('normalize_advantage',False),('target_kl',0.1)])
def test_saved_model_cannot_silently_disagree_with_recorded_recipe(name,value):
    model,config=model_and_config();validate_context_model_recipe(model,config)
    setattr(model,name,value)
    with pytest.raises(ValueError,match='differs'):validate_context_model_recipe(model,config)


@pytest.mark.parametrize('name',['lr_schedule','clip_range'])
def test_unproven_callable_schedule_is_rejected(name):
    model,config=model_and_config();setattr(model,name,lambda progress:0.2)
    with pytest.raises(ValueError,match='schedule differs'):validate_context_model_recipe(model,config)


def test_optimizer_actual_learning_rate_must_match_constant_schedule():
    model,config=model_and_config();model.policy.optimizer.param_groups[0]['lr']=0.00003
    with pytest.raises(ValueError,match='optimizer learning rate'):validate_context_model_recipe(model,config)
