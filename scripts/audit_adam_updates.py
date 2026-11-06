"""Counterfactual single Adam steps on clones; original PPO stays unchanged."""

import argparse
import copy
import json
from pathlib import Path
import statistics

from audit_ppo_gradients import run as gradient_audit, snapshot_policy


def flat_parameters(policy):
    import torch
    return torch.cat([p.detach().reshape(-1) for p in policy.parameters()]).clone()


def cloned_step(policy, weights, optimizer_state, gradient, actor_names, max_norm, case):
    """Always reset weights and state; no counterfactual feeds into the next."""
    import torch
    policy.load_state_dict(weights)
    state = copy.deepcopy(optimizer_state)
    if case == 'fresh_global':
        state['state'] = {}
    policy.optimizer.load_state_dict(state)
    policy.optimizer.zero_grad(set_to_none=True)
    if case == 'restored_zero_global':
        gradient = torch.zeros_like(gradient)
    offset = 0
    actors, critics = [], []
    for name, parameter in policy.named_parameters():
        parameter.grad = gradient[offset:offset+parameter.numel()].reshape_as(parameter).detach().clone()
        offset += parameter.numel()
        (actors if name in actor_names else critics).append(parameter)
    assert offset == gradient.numel()
    if case == 'restored_separate':
        torch.nn.utils.clip_grad_norm_(actors, max_norm)
        torch.nn.utils.clip_grad_norm_(critics, max_norm)
    elif case in ('restored_global','fresh_global','restored_zero_global'):
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()), max_norm)
    else:
        raise ValueError('Unknown diagnostic case: '+case)
    policy.optimizer.step()
    result = flat_parameters(policy)
    assert torch.isfinite(result).all()
    return result


def distribution_change(before, after):
    """Normalized float64 distributions keep tiny KLs interpretable."""
    import torch
    before, after = before.double(), after.double()
    before = before / before.sum(dim=-1, keepdim=True)
    after = after / after.sum(dim=-1, keepdim=True)
    safe = torch.finfo(before.dtype).tiny
    kl = (before * (before.clamp_min(safe).log()-after.clamp_min(safe).log())).sum(dim=-1)
    return dict(mean_forward_kl=float(kl.clamp_min(0).mean().item()),
                mean_total_variation=float((.5*(before-after).abs().sum(dim=-1)).mean().item()),
                greedy_agreement=float((before.argmax(-1)==after.argmax(-1)).double().mean().item()))


class AdamObserver:
    CASES = ('restored_global','restored_separate','fresh_global','restored_zero_global')

    def __init__(self, folder):
        self.folder = folder
        self.clone = None
        self.rows = []

    def __call__(self, model, data, gradients, actor_mask, index):
        import torch
        policy = model.policy
        if self.clone is None:
            assert isinstance(policy.optimizer, torch.optim.Adam)
            assert len(policy.optimizer.param_groups) == 1
            # Constructors initialize random weights; isolate their RNG use.
            with torch.random.fork_rng(devices=[model.device.index or 0]):
                self.clone = type(policy)(**policy._get_constructor_parameters()).to(model.device)
            assert {id(p) for p in self.clone.parameters()} == {
                id(p) for group in self.clone.optimizer.param_groups for p in group['params']}
            self.weights = snapshot_policy(policy)
            self.optimizer = copy.deepcopy(policy.optimizer.state_dict())
            self.before = flat_parameters(policy)
            self.actor_names = {name for name, _ in policy.named_parameters() if not name.startswith('value_net.')}
            assert len(self.optimizer['state']) == len(list(policy.parameters()))
            self.source_adam_steps = sorted({float(s['step'].item()) for s in self.optimizer['state'].values()})
        clone = self.clone
        clone.set_training_mode(False)
        gradient = gradients[0]+model.ent_coef*gradients[1]+model.vf_coef*gradients[2]
        with torch.no_grad():
            before_probs = policy.get_distribution(data.observations, data.action_masks).distribution.probs.detach().clone()
        cases, deltas = {}, {}
        for case in self.CASES:
            values = cloned_step(clone, self.weights, self.optimizer, gradient, self.actor_names,
                                 model.max_grad_norm, case)
            delta = (values-self.before).double()
            deltas[case] = delta[actor_mask]
            with torch.no_grad():
                after_probs = clone.get_distribution(data.observations, data.action_masks).distribution.probs.detach().clone()
            metrics = distribution_change(before_probs, after_probs)
            metrics.update(actor_delta_norm=float(delta[actor_mask].norm().item()),
                           critic_delta_norm=float(delta[~actor_mask].norm().item()),
                           adam_steps_after=sorted({float(s['step'].item()) for s in clone.optimizer.state.values()}))
            cases[case] = metrics
        reference = deltas['restored_global']
        reference_norm = reference.norm()
        assert reference_norm > 0
        comparisons = {}
        for case, delta in deltas.items():
            if case == 'restored_global':
                continue
            comparisons[case] = dict(actor_delta_norm_ratio=float((delta.norm()/reference_norm).item()),
                actor_delta_cosine=float(torch.nn.functional.cosine_similarity(delta,reference,dim=0).item()),
                actor_delta_difference_norm_ratio=float(((delta-reference).norm()/reference_norm).item()))
        row = dict(batch=index,cases=cases,compared_with_restored_global=comparisons)
        self.rows.append(row)
        with (self.folder/'adam_batches.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')


def run(folder):
    from resource_guard import now, write_json
    design = json.loads((folder/'design.json').read_text())
    observer = AdamObserver(folder)
    gradient_audit(folder, batch_observer=observer)
    audit = json.loads((folder/'audit.json').read_text())
    reference = json.loads(Path(design['reference_gradient_audit']).read_text())
    assert audit['rollout_sha256'] == reference['rollout_sha256'], 'Source rollout changed'
    assert audit['per_batch'] == reference['per_batch'], 'Fixed-policy batch summaries changed'
    assert len(observer.rows) == 16
    assert audit['policy_and_adam_unchanged'] and audit['optimizer_steps'] == 0
    summary = {}
    for case in observer.CASES:
        summary[case] = {key:dict(min=min(values),median=statistics.median(values),max=max(values))
            for key in ['actor_delta_norm','critic_delta_norm','mean_forward_kl','mean_total_variation','greedy_agreement']
            if (values := [row['cases'][case][key] for row in observer.rows])}
        if case != 'restored_global':
            summary[case]['compared_with_restored_global'] = {
                key:dict(min=min(values),median=statistics.median(values),max=max(values))
                for key in ['actor_delta_norm_ratio','actor_delta_cosine','actor_delta_difference_norm_ratio']
                if (values := [row['compared_with_restored_global'][case][key] for row in observer.rows])}
    result = dict(at=now(),status='passed',source_adam_steps=observer.source_adam_steps,
                  original_optimizer_steps=0,cloned_optimizer_steps=16*len(observer.CASES),
                  source_policy_and_adam_unchanged=True,reference_rollout_and_batch_summaries_exact=True,
                  rollout_sha256=audit['rollout_sha256'],summary=summary,
                  scope='Independent one-step counterfactuals reset to original parameters and Adam for every batch/case. No sequential training, quality evaluation or ordinary checkpoint.')
    write_json(folder/'adam.json', result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
