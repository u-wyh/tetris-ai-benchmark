"""Trace actual MaskablePPO.train on isolated policies and a fixed source rollout."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from audit_adam_updates import distribution_change, flat_parameters
from audit_ppo_gradients import run as gradient_audit


class ObservedBuffer:
    def __init__(self, buffer):
        self.buffer = buffer
        self.last = None

    def __getattr__(self, key):
        return getattr(self.buffer, key)

    def get(self, batch_size):
        for batch in self.buffer.get(batch_size):
            self.last = batch
            yield batch


def clone_policy(source, device):
    import torch
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        clone = type(source)(**source._get_constructor_parameters()).to(device)
    clone.load_state_dict(copy.deepcopy(source.state_dict()))
    clone.optimizer.load_state_dict(copy.deepcopy(source.optimizer.state_dict()))
    assert {id(p) for p in clone.parameters()} == {
        id(p) for group in clone.optimizer.param_groups for p in group['params']}
    return clone


def train_clone(source, learning_rate, target_kl, probe, before_probs, folder, case, traced=True):
    import torch
    from stable_baselines3.common.logger import configure
    from training.train_transaction import global_rng_state, restore_global_rng
    rng = global_rng_state(source)
    model = copy.copy(source)
    model.policy = clone_policy(source.policy, source.device)
    model.env = None  # Calling learn/collect would fail instead of collecting more samples.
    model.rollout_buffer = ObservedBuffer(source.rollout_buffer) if traced else source.rollout_buffer
    model.lr_schedule = lambda _: learning_rate
    model.learning_rate = learning_rate
    model.target_kl = target_kl
    model.verbose = 0
    model.set_logger(configure(str(folder/'logs'/case), []))
    before = flat_parameters(model.policy)
    actors = torch.cat([torch.full((p.numel(),),not name.startswith('value_net.'),
                dtype=torch.bool,device=p.device) for name,p in model.policy.named_parameters()])
    rows, evaluations = [], []
    hook = None
    evaluate_actions = model.policy.evaluate_actions

    def evaluate(*args, **kwargs):
        values, log_prob, entropy = evaluate_actions(*args, **kwargs)
        with torch.no_grad():
            log_ratio = log_prob-model.rollout_buffer.last.old_log_prob
            approximate_kl = float(((log_ratio.exp()-1)-log_ratio).mean().item())
        evaluations.append(dict(evaluation=len(evaluations)+1,
                                epoch=model._n_updates-source._n_updates,approximate_kl=approximate_kl))
        return values, log_prob, entropy

    def after_step(optimizer, args, kwargs):
        with torch.no_grad():
            delta = (flat_parameters(model.policy)-before).double()
            probs = model.policy.get_distribution(probe['observations'],probe['masks']).distribution.probs
            row = dict(case=case,step=len(rows)+1,epoch=model._n_updates-source._n_updates,
                       pre_step_approximate_kl=evaluations[-1]['approximate_kl'],
                       actor_delta_norm=float(delta[actors].norm().item()),
                       critic_delta_norm=float(delta[~actors].norm().item()),
                       **distribution_change(before_probs,probs))
        rows.append(row)
        with (folder/'update_steps.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    try:
        if traced:
            model.policy.evaluate_actions = evaluate
            hook = model.policy.optimizer.register_step_post_hook(after_step)
        restore_global_rng(rng)
        model.train()  # Installed SB3 implementation, not a reimplemented PPO loop.
        with torch.no_grad():
            probs = model.policy.get_distribution(probe['observations'],probe['masks']).distribution.probs
            final = distribution_change(before_probs,probs)
        counts = sorted({int(s['step'].item()) for s in model.policy.optimizer.state.values()})
        source_counts = {int(s['step'].item()) for s in source.policy.optimizer.state.values()}
        assert len(counts) == len(source_counts) == 1
        steps = counts[0]-next(iter(source_counts))
        assert 0 < steps <= model.n_epochs*(model.n_steps*model.n_envs//model.batch_size)
        if traced:
            assert steps == len(rows)
        stopped = target_kl is not None and bool(evaluations) and evaluations[-1]['approximate_kl'] > 1.5*target_kl
        summary = dict(learning_rate=learning_rate,target_kl=target_kl,optimizer_steps=steps,
            ppo_epoch_counter_delta=model._n_updates-source._n_updates,
            early_stopped=stopped,final_probe=final,
            last_evaluated_minibatch=evaluations[-1] if evaluations else None,
            max_probe_forward_kl=max((r['mean_forward_kl'] for r in rows),default=None))
        return model, summary
    finally:
        if hook is not None:
            hook.remove()
        model.policy.evaluate_actions = evaluate_actions
        model.logger.close()
        restore_global_rng(rng)


class UpdateObserver:
    def __init__(self, folder):
        self.folder, self.summary, self.probe_sha256 = folder, {}, None

    def __call__(self, source):
        import torch
        from sb3_contrib import MaskablePPO
        from training.train_transaction import global_rng_state, restore_global_rng
        from verify_feature_backend_benchmark import identical
        assert type(source) is MaskablePPO and source.n_envs == 8 and source.n_epochs == 5
        # Choose the probe without consuming the first training permutation.
        rng = global_rng_state(source)
        try:
            data = next(source.rollout_buffer.get(source.batch_size))
        finally:
            restore_global_rng(rng)
        probe = dict(observations={k:v[:64].detach().clone() for k,v in data.observations.items()},
                     masks=data.action_masks[:64].detach().clone())
        digest = hashlib.sha256()
        for name,value in sorted({**probe['observations'],'masks':probe['masks']}.items()):
            digest.update(name.encode());digest.update(value.cpu().numpy().tobytes())
        self.probe_sha256 = digest.hexdigest()
        with torch.no_grad():
            before_probs = source.policy.get_distribution(probe['observations'],probe['masks']).distribution.probs.detach().clone()
        # Prove instrumentation leaves actual train policy/Adam/epoch counts exact.
        control, control_summary = train_clone(source,3e-4,None,probe,before_probs,self.folder,'control',False)
        for case, rate, kl in [('original',3e-4,None),('original_kl',3e-4,.003),
                              ('low_lr',3e-5,None),('low_lr_kl',3e-5,.003)]:
            model, summary = train_clone(source,rate,kl,probe,before_probs,self.folder,case)
            if case == 'original':
                assert identical(model.policy.state_dict(),control.policy.state_dict())
                assert identical(model.policy.optimizer.state_dict(),control.policy.optimizer.state_dict())
                assert model._n_updates == control._n_updates
                assert summary['optimizer_steps'] == control_summary['optimizer_steps'] == 80
                self.instrumentation_exact = True
                del control
            self.summary[case] = summary
            (self.folder/'update_cases.json').write_text(json.dumps(self.summary,indent=2)+'\n')
            del model


def run(folder):
    from resource_guard import now,write_json
    observer = UpdateObserver(folder)
    gradient_audit(folder,rollout_observer=observer)
    design=json.loads((folder/'design.json').read_text())
    reference=json.loads(Path(design['reference_gradient_audit']).read_text())
    audit=json.loads((folder/'audit.json').read_text())
    assert audit['rollout_sha256']==reference['rollout_sha256']
    assert audit['per_batch']==reference['per_batch']
    assert audit['policy_and_adam_unchanged'] and audit['optimizer_steps']==0
    write_json(folder/'update_path.json',dict(at=now(),status='passed',original_optimizer_steps=0,
        source_policy_and_adam_unchanged=True,reference_rollout_and_batch_summaries_exact=True,
        instrumentation_matches_untraced_train=observer.instrumentation_exact,
        rollout_sha256=audit['rollout_sha256'],probe_sha256=observer.probe_sha256,probe_training_states=64,
        cases=observer.summary,
        scope='One source rollout, sequential train on independent policy/Adam clones, same restored RNG. Probe states are training states; minibatch approximate KL stopping is not a hard global KL bound. No ordinary checkpoint or quality evaluation.'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    run(parser.parse_args().run_dir.resolve())
