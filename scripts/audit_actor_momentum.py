"""Isolate actor Adam first moments on copies of the original rollout context."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from audit_ppo_gradients import run as gradient_audit
from audit_ppo_update_path import clone_policy, train_clone


def zero_actor_first_moment(policy):
    import torch
    from verify_feature_backend_benchmark import identical
    if not isinstance(policy.optimizer, torch.optim.Adam):
        raise ValueError('Selective transition requires Adam')
    actors = {id(p):name for name,p in policy.named_parameters() if not name.startswith('value_net.')}
    critics = {id(p) for name,p in policy.named_parameters() if name.startswith('value_net.')}
    if not actors or not critics:
        raise ValueError('Expected distinct candidate actor and value_net parameters')
    targets = []
    # Validate every state before mutating any of them.
    for group in policy.optimizer.param_groups:
        for p in group['params']:
            state = policy.optimizer.state.get(p, {})
            if not {'exp_avg','exp_avg_sq','step'} <= state.keys():
                raise ValueError('Incomplete Adam state')
            if (state['exp_avg'].shape != p.shape or state['exp_avg_sq'].shape != p.shape
                    or not torch.isfinite(state['exp_avg']).all() or not torch.isfinite(state['exp_avg_sq']).all()):
                raise ValueError('Invalid Adam moments')
            if id(p) in actors:
                targets.append(p)
    if {id(p) for p in targets} != actors.keys():
        raise ValueError('Adam is missing actor parameters')
    expected = copy.deepcopy(policy.optimizer.state_dict())
    for group,saved in zip(policy.optimizer.param_groups,expected['param_groups']):
        for p,index in zip(group['params'],saved['params']):
            if id(p) in actors:
                expected['state'][index]['exp_avg'].zero_()
    with torch.no_grad():
        for p in targets:
            policy.optimizer.state[p]['exp_avg'].zero_()
    assert identical(expected,policy.optimizer.state_dict()), 'Transition changed other Adam fields'
    return dict(transition='zero_actor_exp_avg_only',actor_parameters=[actors[id(p)] for p in targets],
                second_moments_steps_groups_and_critic_unchanged=True)


class ActorObserver:
    def __init__(self, folder):
        self.folder,self.cases = folder,{}

    def __call__(self, source):
        import torch
        from sb3_contrib import MaskablePPO
        from training.train_transaction import global_rng_state,restore_global_rng
        assert type(source) is MaskablePPO and source.n_envs==8 and source.n_epochs==5
        rng=global_rng_state(source)
        try:
            data=next(source.rollout_buffer.get(source.batch_size))
        finally:
            restore_global_rng(rng)
        probe=dict(observations={k:v[:64].detach().clone() for k,v in data.observations.items()},
                   masks=data.action_masks[:64].detach().clone())
        digest=hashlib.sha256()
        for name,value in sorted({**probe['observations'],'masks':probe['masks']}.items()):
            digest.update(name.encode());digest.update(value.cpu().numpy().tobytes())
        self.probe_sha256=digest.hexdigest()
        with torch.no_grad():
            before=source.policy.get_distribution(probe['observations'],probe['masks']).distribution.probs.detach().clone()
        proxy=copy.copy(source)
        proxy.policy=clone_policy(source.policy,source.device)
        self.transition=zero_actor_first_moment(proxy.policy)
        for case,rate in [('original',3e-4),('low_lr',3e-5)]:
            model,summary=train_clone(proxy,rate,None,probe,before,self.folder,case)
            assert summary['optimizer_steps']==80 and model._n_updates-source._n_updates==5
            self.cases[case]=summary
            (self.folder/'actor_cases.json').write_text(json.dumps(self.cases,indent=2)+'\n')
            del model


def run(folder):
    from resource_guard import now,write_json
    design=json.loads((folder/'design.json').read_text())
    observer=ActorObserver(folder)
    gradient_audit(folder,rollout_observer=observer)
    audit=json.loads((folder/'audit.json').read_text())
    reference=json.loads(Path(design['reference_gradient_audit']).read_text())
    baseline=json.loads(Path(design['reference_update_path']).read_text())
    assert audit['rollout_sha256']==reference['rollout_sha256']==baseline['rollout_sha256']
    assert audit['per_batch']==reference['per_batch']
    assert observer.probe_sha256==baseline['probe_sha256']
    assert audit['policy_and_adam_unchanged'] and audit['optimizer_steps']==0
    comparisons={}
    for case,value in observer.cases.items():
        old=baseline['cases'][case]
        comparisons[case]=dict(final_kl_ratio=value['final_probe']['mean_forward_kl']/old['final_probe']['mean_forward_kl'],
            peak_kl_ratio=value['max_probe_forward_kl']/old['max_probe_forward_kl'],
            greedy_agreement_delta=value['final_probe']['greedy_agreement']-old['final_probe']['greedy_agreement'])
    gate=design['diagnostic_gate']
    eligible=all(r['final_kl_ratio']<gate['final_kl_ratio_below']
        and r['peak_kl_ratio']<gate['peak_kl_ratio_below']
        and r['greedy_agreement_delta']>=gate['greedy_agreement_delta_at_least'] for r in comparisons.values())
    write_json(folder/'actor_momentum.json',dict(at=now(),status='passed',original_optimizer_steps=0,
        source_policy_and_adam_unchanged=True,reference_rollout_batch_summaries_and_probe_exact=True,
        rollout_sha256=audit['rollout_sha256'],probe_sha256=observer.probe_sha256,transition=observer.transition,
        cases=observer.cases,comparisons=comparisons,eligible_for_developing_bounded_quality_screen=eligible,
        scope='Selective first-moment transition only on clones,160 sequential copied optimizer steps,64 training probes. Diagnostic gate is not quality evidence or production adoption. No ordinary checkpoint.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    run(parser.parse_args().run_dir.resolve())
