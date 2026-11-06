"""Bounded critic-only feasibility fit on immutable worker-disjoint rollout data."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from collect_critic_calibration import array_manifest, residual_metrics
from verify_feature_backend_benchmark import identical


def actor_snapshot(policy):
    return dict(weights=copy.deepcopy({name: value for name, value in policy.state_dict().items()
                                      if not name.startswith('value_net.')}),
        adam={name: copy.deepcopy(policy.optimizer.state.get(parameter, {}))
              for name, parameter in policy.named_parameters() if not name.startswith('value_net.')})


def fit(policy, states, targets, *, epochs, batch_size, seed, value_coef, max_grad_norm, on_epoch=None):
    """Update only critic parameters, preserving actor gradients and Adam history."""
    import torch
    if len(states) != len(targets) or len(states) % batch_size:
        raise ValueError('Expected complete fixed fit batches')
    before = actor_snapshot(policy)
    critic = []
    for name, parameter in policy.named_parameters():
        selected = name.startswith('value_net.')
        parameter.requires_grad_(selected)
        if selected:
            critic.append(parameter)
    if not critic:
        raise ValueError('Missing separate value_net parameters')
    generator = np.random.RandomState(seed)
    steps = 0
    policy.train(True)
    for epoch in range(epochs):
        total_loss = 0.
        indices = generator.permutation(len(states))
        for start in range(0, len(states), batch_size):
            batch = indices[start:start+batch_size]
            prediction = policy.predict_values({'state': states[batch]}).flatten()
            loss = value_coef*torch.nn.functional.mse_loss(prediction, targets[batch])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite critic loss')
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic, max_grad_norm, error_if_nonfinite=True)
            assert all(p.grad is None for name, p in policy.named_parameters()
                       if not name.startswith('value_net.'))
            policy.optimizer.step()
            steps += 1
            total_loss += float(loss.detach())*len(batch)
        assert identical(before, actor_snapshot(policy)), 'Actor weights or Adam history changed'
        if on_epoch:
            on_epoch(dict(epoch=epoch+1, weighted_fit_mse=total_loss/len(states), optimizer_steps=steps))
    policy.optimizer.zero_grad(set_to_none=True)
    assert identical(before, actor_snapshot(policy))
    return steps


def feasibility_gate(baseline, fitted, thresholds):
    for metric in ('overall', 'terminal_window', 'outside_terminal_window'):
        if baseline[metric]['samples'] == 0 or fitted[metric]['samples'] != baseline[metric]['samples']:
            return False
    return all(fitted[name]['rmse'] <= baseline[name]['rmse']*thresholds[name] for name in thresholds)


def load_data(folder):
    fields = ('obs_state', 'values', 'returns', 'episode_starts', 'step_terminated', 'info_episode_seed')
    collected = {name: [] for name in fields}
    receipt = json.loads((folder/'collection_summary.json').read_text())
    assert receipt['status'] == 'passed' and receipt['samples'] == 16384
    for index in range(1, 5):
        segment = folder/f'chunk_{index:04d}'
        metadata = json.loads((segment/'dataset.json').read_text())
        if hashlib.sha256((segment/'rollout.npz').read_bytes()).hexdigest() != metadata['file_sha256']:
            raise RuntimeError('Dataset archive hash differs')
        assert metadata['design_sha256'] == receipt['design_sha256']
        with np.load(segment/'rollout.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in fields}
        assert array_manifest(arrays) == {name: metadata['arrays'][name] for name in fields}
        for name in fields:
            collected[name].append(arrays[name])
    arrays = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    fit_episodes = set(arrays['info_episode_seed'][:, :4].flatten())
    validation_episodes = set(arrays['info_episode_seed'][:, 4:].flatten())
    if fit_episodes & validation_episodes:
        raise ValueError('Fit/validation episodes overlap')
    assert arrays['obs_state'].shape == (2048, 8, 237)
    return arrays


def run(folder):
    import torch
    from sb3_contrib import MaskablePPO
    from audit_ppo_update_path import clone_policy
    from audit_critic_tail_gradients import group_masks
    from resource_guard import now, write_json

    if any((folder/name).exists() for name in ('feasibility.json', 'epochs.jsonl', 'critic_diagnostic.pt')):
        raise FileExistsError('Preserve completed or partial fit; do not silently repeat')
    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for name, digest in design['frozen_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Frozen critic fit input changed: '+name)
    unchanged()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    arrays = load_data(Path(design['dataset_dir']))
    terminal_mask = group_masks(arrays, design['terminal_window'], .01)['terminal_window']
    states = torch.as_tensor(arrays['obs_state'].reshape(-1, 237), device='cuda')
    source = MaskablePPO.load(design['model'], device='cuda')
    assert source.get_env() is None and source.device.type == 'cuda'
    original = copy.deepcopy(source.policy.state_dict())
    original_adam = copy.deepcopy(source.policy.optimizer.state_dict())
    source_counts = source.num_timesteps, source._n_updates
    policy = clone_policy(source.policy, source.device)
    assert all(group['lr'] == design['learning_rate'] for group in policy.optimizer.param_groups)
    assert source.vf_coef == design['value_coef'] and source.max_grad_norm == design['max_grad_norm']

    def predict(model):
        model.train(False)
        with torch.no_grad():
            predictions = [model.predict_values({'state': states[start:start+256]}).flatten().cpu().numpy()
                           for start in range(0, len(states), 256)]
        return np.concatenate(predictions).reshape(2048, 8)

    baseline_values = predict(source.policy)
    np.testing.assert_allclose(baseline_values, arrays['values'], rtol=1e-6, atol=1e-4)
    fit_states = torch.as_tensor(arrays['obs_state'][:, :4].reshape(-1, 237), device='cuda')
    fit_targets = torch.as_tensor(arrays['returns'][:, :4].reshape(-1), device='cuda')
    critic_steps_before = {name: int(policy.optimizer.state[p]['step'].item())
        for name, p in policy.named_parameters() if name.startswith('value_net.')}
    def log_epoch(row):
        with (folder/'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
    steps = fit(policy, fit_states, fit_targets, epochs=design['epochs'], batch_size=design['batch_size'],
        seed=design['permutation_seed'], value_coef=design['value_coef'],
        max_grad_norm=design['max_grad_norm'], on_epoch=log_epoch)
    assert steps == 160
    assert all(int(policy.optimizer.state[p]['step'].item())-critic_steps_before[name] == steps
               for name, p in policy.named_parameters() if name.startswith('value_net.'))
    assert identical(actor_snapshot(source.policy), actor_snapshot(policy))
    fitted_values = predict(policy)  # Validation is read once after the final fixed epoch.
    assert identical(original, source.policy.state_dict()) and identical(original_adam, source.policy.optimizer.state_dict())
    assert (source.num_timesteps, source._n_updates) == source_counts

    def metrics(predictions, workers):
        predicted, returns = predictions[:, workers], arrays['returns'][:, workers]
        mask = terminal_mask[:, workers]
        return dict(overall=residual_metrics(predicted, returns),
            terminal_window=residual_metrics(predicted[mask], returns[mask]),
            outside_terminal_window=residual_metrics(predicted[~mask], returns[~mask]),
            per_worker=[residual_metrics(predictions[:, worker], arrays['returns'][:, worker]) for worker in workers])
    baseline = {name: metrics(baseline_values, workers) for name, workers in
                [('fit', [0, 1, 2, 3]), ('validation', [4, 5, 6, 7])]}
    fitted = {name: metrics(fitted_values, workers) for name, workers in
              [('fit', [0, 1, 2, 3]), ('validation', [4, 5, 6, 7])]}
    eligible = feasibility_gate(baseline['validation'], fitted['validation'], design['thresholds'])
    # This deliberately cannot be loaded as a normal PPO model.zip.
    to_cpu = lambda state: {key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                             for key, value in state.items()}
    artifact = dict(critic_weights=to_cpu(policy.value_net.state_dict()),
        critic_adam={name: to_cpu(policy.optimizer.state[p]) for name, p in policy.named_parameters()
                     if name.startswith('value_net.')})
    torch.save(artifact, folder/'critic_diagnostic.pt')
    assert identical(artifact, torch.load(folder/'critic_diagnostic.pt', weights_only=True))
    unchanged()
    write_json(folder/'feasibility.json', dict(at=now(), status='passed',
        eligible_for_ppo_path_study=eligible, optimizer_steps_in_critic_copy=steps,
        source_optimizer_steps=0, actor_parameters_and_adam_unchanged=True,
        source_policy_adam_ppo_counters_unchanged=True, env_steps=0, collected_samples=0,
        fit_samples=8192, validation_samples=8192, original_dataset_n_envs=8, device='cuda',
        baseline=baseline, fitted=fitted, thresholds=design['thresholds'],
        artifact_sha256=hashlib.sha256((folder/'critic_diagnostic.pt').read_bytes()).hexdigest(),
        scope='Final fixed five-epoch critic-only copy fit; validation never selects epochs or hyperparameters. '
              'Targets are frozen source GAE with bootstrap, not Monte Carlo truth. Two validation terminal '
              'events/four workers, correlated steps, no actor/PPO quality training, promotion or generalization claim. '
              'Artifact contains only diagnostic critic weights/Adam, not a deployable policy.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
