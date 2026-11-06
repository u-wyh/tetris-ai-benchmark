"""Attribute fixed critic gradients from a durable rollout, without collecting or learning."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from audit_ppo_gradients import gradient_vector, gradient_summary
from collect_critic_calibration import array_manifest


def partition_gradients(per_sample_loss, selected, parameters):
    """Keep the original batch denominator in BOTH complementary contributions."""
    import torch
    if selected.dtype != torch.bool or selected.shape != per_sample_loss.shape:
        raise ValueError('Partition must be a boolean mask with one entry per loss')
    whole = gradient_vector(per_sample_loss.mean(), parameters)
    left = gradient_vector((per_sample_loss*selected).mean(), parameters)
    right = gradient_vector((per_sample_loss*~selected).mean(), parameters)
    error = float((whole-left-right).double().norm())
    relative_error = error/max(float(whole.double().norm()), 1e-12)
    torch.testing.assert_close(whole, left+right, rtol=2e-4, atol=2e-5)
    if relative_error > 2e-5:
        raise AssertionError('Partition gradients do not reconstruct full gradient')
    return whole, left, right, relative_error


def contribution_summary(whole, left, right):
    import torch
    whole, left, right = (v.detach().double() for v in (whole, left, right))
    norm, left_norm, right_norm = (float(v.norm()) for v in (whole, left, right))
    return dict(full_norm=norm, selected_contribution_norm=left_norm,
        other_contribution_norm=right_norm,
        selected_to_full_norm_ratio=left_norm/norm if norm else None,
        selected_projection_on_full=float(torch.dot(left, whole))/(norm**2) if norm else None,
        selected_other_cosine=float(torch.dot(left, right))/(left_norm*right_norm)
            if left_norm and right_norm else None)


def group_masks(arrays, terminal_window, top_fraction):
    residual = arrays['returns'].astype(np.float64)-arrays['values'].astype(np.float64)
    terminal = np.zeros_like(residual, dtype=bool)
    for step, worker in np.argwhere(arrays['step_terminated']):
        start = max(0, step-terminal_window+1)
        starts = np.flatnonzero(arrays['episode_starts'][:step+1, worker])
        if starts.size:
            start = max(start, int(starts[-1]))
        terminal[start:step+1, worker] = True
    top = np.zeros_like(terminal)
    count = max(1, int(np.ceil(residual.size*top_fraction)))
    # Match the durable tail analysis' stable time-major tie order.
    top.flat[np.argsort(residual**2, axis=None, kind='stable')[-count:]] = True
    return dict(terminal_window=terminal, top_residual=top)


def run(folder):
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.buffers import BaseBuffer
    from verify_feature_backend_benchmark import identical
    from resource_guard import now, write_json

    if (folder/'gradient_tail.json').exists() or (folder/'batches.jsonl').exists():
        raise FileExistsError('Preserve existing diagnostic; inspect before retry')
    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for name, sha in design['frozen_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                raise RuntimeError('Frozen gradient input changed: '+name)
    unchanged()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    dataset = Path(design['dataset_dir'])
    metadata = json.loads((dataset/'dataset.json').read_text())
    if hashlib.sha256((dataset/'rollout.npz').read_bytes()).hexdigest() != metadata['file_sha256']:
        raise RuntimeError('Dataset archive changed')
    with np.load(dataset/'rollout.npz', allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    assert array_manifest(arrays) == metadata['arrays']
    masks = group_masks(arrays, design['terminal_window'], design['top_fraction'])
    flat = {name: BaseBuffer.swap_and_flatten(value) for name, value in arrays.items()
            if value.ndim >= 2}
    masks = {name: BaseBuffer.swap_and_flatten(mask).flatten() for name, mask in masks.items()}
    count = metadata['samples']
    permutation = np.random.RandomState(design['permutation_seed']).permutation(count)
    np.testing.assert_array_equal(np.sort(permutation), np.arange(count))
    write_json(folder/'permutation.json', dict(seed=design['permutation_seed'],
        flat_order='SB3 env-major swap_and_flatten; flat_index=worker*512+step',
        indices=permutation.tolist(), scope='New fixed permutation, not historical sampler RNG'))

    # Loading the frozen model without env precludes accidentally collecting steps.
    model = MaskablePPO.load(design['model'], device='cuda')
    assert model.get_env() is None and model.device.type == 'cuda'
    assert model.num_timesteps == metadata['source_steps'] and model._n_updates == metadata['source_ppo_updates']
    assert model.batch_size == 256 and model.clip_range_vf is None
    assert model.normalize_advantage and model.clip_range(1.) == .2
    policy, adam = copy.deepcopy(model.policy.state_dict()), copy.deepcopy(model.policy.optimizer.state_dict())
    parameters = list(model.policy.parameters())
    actor_mask = torch.cat([torch.full((p.numel(),), not name.startswith('value_net.'),
        dtype=torch.bool, device=p.device) for name, p in model.policy.named_parameters()])
    model.policy.set_training_mode(True)
    totals, rows = {}, []
    max_value_difference = max_log_prob_difference = 0.
    try:
        for batch, start in enumerate(range(0, count, model.batch_size)):
            indices = permutation[start:start+model.batch_size]
            tensor = lambda name: torch.as_tensor(flat[name][indices], device=model.device)
            observations = {name[4:]: tensor(name) for name in flat if name.startswith('obs_')}
            values, log_prob, entropy = model.policy.evaluate_actions(observations,
                tensor('actions').long().flatten(), action_masks=tensor('action_masks'))
            values = values.flatten()
            torch.testing.assert_close(values, tensor('values').flatten(), rtol=1e-6, atol=1e-4)
            torch.testing.assert_close(log_prob, tensor('log_probs').flatten(), rtol=1e-5, atol=1e-5)
            max_value_difference = max(max_value_difference, float((values-tensor('values').flatten()).detach().abs().max()))
            max_log_prob_difference = max(max_log_prob_difference, float((log_prob-tensor('log_probs').flatten()).detach().abs().max()))
            advantages = tensor('advantages').flatten()
            advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
            ratio = torch.exp(log_prob-tensor('log_probs').flatten())
            pg_loss = -torch.min(advantages*ratio, advantages*torch.clamp(ratio, .8, 1.2)).mean()
            entropy_loss = -entropy.mean() if entropy is not None else log_prob.mean()
            pg, ent = (gradient_vector(loss, parameters) for loss in (pg_loss, entropy_loss))
            per_sample = (tensor('returns').flatten()-values).square()
            row = dict(batch=batch, samples=len(indices), groups={})
            for name, selected in masks.items():
                mask = torch.as_tensor(selected[indices], device=model.device)
                full, left, right, error = partition_gradients(per_sample, mask, parameters)
                assert torch.count_nonzero(full[actor_mask]) == 0
                assert torch.count_nonzero(pg[~actor_mask]) == 0
                baseline = gradient_summary(pg, ent, full, actor_mask,
                    model.ent_coef, model.vf_coef, model.max_grad_norm)
                without = gradient_summary(pg, ent, right, actor_mask,
                    model.ent_coef, model.vf_coef, model.max_grad_norm)
                row['groups'][name] = dict(contribution_summary(full, left, right),
                    selected_samples=int(mask.sum()), reconstruction_relative_error=error,
                    global_clip_multiplier=baseline['global_clip_multiplier'],
                    actor_only_clip_multiplier=baseline['actor_only_clip_multiplier'],
                    clip_multiplier_if_selected_critic_contribution_removed=without['global_clip_multiplier'],
                    full_combined_actor_norm=baseline['combined_actor_norm'],
                    full_combined_critic_norm=baseline['combined_critic_norm'])
                if name not in totals:
                    totals[name] = [v.detach().double()/16 for v in (full, left, right)]
                else:
                    for accumulated, value in zip(totals[name], (full, left, right)):
                        accumulated.add_(value.detach().double()/16)
            rows.append(row)
            with (folder/'batches.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            write_json(folder/'status_detail.json', dict(at=now(), batches_completed=len(rows)))
            del observations, values, log_prob, entropy, per_sample, pg, ent
        assert len(rows) == 16
        assert identical(policy, model.policy.state_dict()) and identical(adam, model.policy.optimizer.state_dict())
        assert model.num_timesteps == metadata['source_steps'] and model._n_updates == metadata['source_ppo_updates']
        assert all(p.grad is None for p in parameters)
        unchanged()
        summary = {}
        for name in masks:
            group_rows = [row['groups'][name] for row in rows]
            summary[name] = dict(selected_samples=int(masks[name].sum()),
                aggregate_critic_gradient=contribution_summary(*totals[name]),
                median_global_clip_multiplier=statistics.median(r['global_clip_multiplier'] for r in group_rows),
                median_clip_multiplier_without_selected_critic=statistics.median(
                    r['clip_multiplier_if_selected_critic_contribution_removed'] for r in group_rows),
                max_reconstruction_relative_error=max(r['reconstruction_relative_error'] for r in group_rows))
        write_json(folder/'gradient_tail.json', dict(at=now(), status='passed',
            collected_samples=0, samples_reused=count, optimizer_steps=0, env_steps=0,
            policy_adam_ppo_counts_unchanged=True, device='cuda', original_dataset_n_envs=8,
            dataset_sha256=metadata['file_sha256'], permutation_seed=design['permutation_seed'],
            max_saved_value_difference=max_value_difference, max_saved_log_prob_difference=max_log_prob_difference,
            coefficients=dict(value=model.vf_coef, entropy=model.ent_coef, max_grad_norm=model.max_grad_norm),
            groups=summary,
            scope='Fixed original parameters; new declared permutation, not historical minibatch order. '
                  'Both loss partitions use original full minibatch denominator. Aggregate critic gradient '
                  'is an average of 16 batch gradients, not an optimizer update. Norm ratios/projections '
                  'are not sample or error fractions and can exceed one. Removing selected critic contribution '
                  'is a mathematical clipping counterfactual with unchanged full actor objective, not a recipe. '
                  'One terminal event, correlated data, no Adam update/quality/causal claim.'))
    finally:
        del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
