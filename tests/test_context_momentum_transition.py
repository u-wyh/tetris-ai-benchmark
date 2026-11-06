import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

from training.initialization import context_actor_momentum, restore_initial_context, validate_initial_context
from training.optimizer_transition import zero_actor_first_moment
from scripts.audit_actor_momentum import zero_actor_first_moment as diagnostic_transition
from scripts.verify_feature_backend_benchmark import identical


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=torch.nn.Linear(2,1)
        self.value_net=torch.nn.Linear(2,1)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=.0003,amsgrad=True)
        for p in self.parameters():p.grad=torch.ones_like(p)
        self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize('bad',[None,False,0,'reset','zero_all',[],{}])
def test_unknown_momentum_transition_is_rejected(bad):
    with pytest.raises(ValueError,match='actor_momentum'):
        context_actor_momentum(dict(actor_momentum_after_restore=bad))


def test_opt_in_momentum_is_validated_and_default_preserves(tmp_path):
    spec=dict(run_dir=str(tmp_path),task=1,files_sha256={name:'a'*64 for name in
        ['config.json','metadata.json','committed/task_000001/manifest.json']})
    assert context_actor_momentum(spec)=='preserve'
    spec['actor_momentum_after_restore']='zero_first_moment'
    preset=dict(device='cuda',n_envs=8,candidate_policy=True,initial_context=spec)
    assert validate_initial_context(preset)==spec
    spec['actor_momentum_after_restore']='zero_all'
    with pytest.raises(ValueError,match='actor_momentum'):
        validate_initial_context(preset)


def test_production_transition_is_exactly_the_diagnosed_transition():
    policy=Policy();clone=Policy()
    clone.load_state_dict(copy.deepcopy(policy.state_dict()))
    clone.optimizer.load_state_dict(copy.deepcopy(policy.optimizer.state_dict()))
    assert zero_actor_first_moment(policy)==diagnostic_transition(clone)
    assert identical(policy.state_dict(),clone.state_dict())
    assert identical(policy.optimizer.state_dict(),clone.optimizer.state_dict())


def test_transition_rejects_missing_critic_before_mutating_actor():
    policy=Policy();actor=list(policy.actor.parameters())
    policy.optimizer.param_groups[0]['params']=actor
    before=copy.deepcopy(policy.optimizer.state[actor[0]])
    with pytest.raises(ValueError,match='missing'):
        zero_actor_first_moment(policy)
    assert identical(before,policy.optimizer.state[actor[0]])


@pytest.mark.parametrize('mode',['preserve','zero_first_moment'])
def test_context_restore_applies_declared_transition_once_after_recipe_validation(tmp_path,monkeypatch,mode):
    import training.initialization as init
    import training.train_vector_transaction as trainer
    policy=Policy();weights=copy.deepcopy(policy.state_dict());before=copy.deepcopy(policy.optimizer.state_dict())
    model=SimpleNamespace(policy=policy,num_timesteps=4096,_n_updates=17)
    config=dict(ppo=dict(learning_rate=.0003),source_steps_offset=4096)
    spec=dict(run_dir=str(tmp_path),task=1,actor_momentum_after_restore=mode)
    (tmp_path/'metadata.json').write_text('{}')
    monkeypatch.setattr(init,'context_source_config',lambda *args:dict(ppo=config['ppo'],task_steps=4096))
    monkeypatch.setattr(trainer,'restore_task',lambda *args:(model,dict(sb3_last_episode_starts=[False]*8),None))
    checks=[]
    def validate(*args):
        assert identical(before,policy.optimizer.state_dict())
        checks.append('recipe')
    monkeypatch.setattr(init,'validate_context_model_recipe',validate)
    restored,record=restore_initial_context(spec,config,None)
    assert len(checks)==2 and restored.num_timesteps==0 and restored._n_updates==17
    assert identical(weights,policy.state_dict())
    assert record['actor_momentum_after_restore']==mode
    if mode=='preserve':assert identical(before,policy.optimizer.state_dict())
    else:
        for name,p in policy.named_parameters():
            if name.startswith('actor.'):assert torch.count_nonzero(policy.optimizer.state[p]['exp_avg'])==0
            else:assert torch.count_nonzero(policy.optimizer.state[p]['exp_avg'])>0
