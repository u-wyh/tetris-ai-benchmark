"""Offline original-policy KL control on immutable CUDA8 rollout data."""
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

from audit_ppo_update_path import clone_policy
from audit_adam_updates import distribution_change
from collect_critic_calibration import array_manifest, verify_targets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_actions(masks, actions):
    if (masks.ndim != 2 or not np.isfinite(masks).all() or
            not np.isin(masks, [0, 1]).all() or not masks.any(axis=1).all()):
        raise ValueError('Every mask must be binary with at least one legal action')
    actions = np.asarray(actions).reshape(-1)
    if (len(actions) != len(masks) or not np.isfinite(actions).all() or
            not np.equal(actions, actions.astype(np.int64)).all() or
            np.any(actions < 0) or np.any(actions >= masks.shape[1])):
        raise ValueError('Invalid stored action')
    if not masks[np.arange(len(actions)), actions.astype(np.int64)].all():
        raise ValueError('Stored action is masked out')


def load_data(folder, expected_metadata_sha, expected_archive_sha):
    if (sha(folder/'dataset.json') != expected_metadata_sha or
            sha(folder/'rollout.npz') != expected_archive_sha):
        raise ValueError('Frozen dataset identity differs before loading arrays')
    metadata = json.loads((folder/'dataset.json').read_text())
    if (metadata['file_sha256'] != expected_archive_sha or metadata['samples'] != 4096 or
            metadata['n_envs'] != 8 or metadata['device'] != 'cuda'):
        raise ValueError('Dataset is not the frozen CUDA8 rollout')
    with np.load(folder/'rollout.npz', allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if array_manifest(arrays) != metadata['arrays']:
        raise ValueError('Dataset array identity differs')
    if (arrays['obs_candidate'].shape != (512, 8, 1840, 16) or
            arrays['obs_state'].shape != (512, 8, 237) or
            arrays['action_masks'].shape != (512, 8, 1840)):
        raise ValueError('Dataset observation shape differs')
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite dataset')
    validate_actions(arrays['action_masks'].reshape(-1, 1840), arrays['actions'])
    verify_targets(arrays, metadata['gamma'], metadata['gae_lambda'])
    from stable_baselines3.common.buffers import BaseBuffer
    flat = {key: BaseBuffer.swap_and_flatten(arrays[key]) for key in
            ['obs_candidate', 'obs_state', 'actions', 'values', 'log_probs', 'advantages', 'returns', 'action_masks']}
    for value in flat.values():
        value.flags.writeable = False
    return metadata, flat


class FixedBuffer:
    """SB3 samples, env-major flattening, explicitly shared epoch permutations."""
    def __init__(self, flat, permutations, device):
        self.flat, self.permutations, self.device = flat, permutations, device
        self.values, self.returns = flat['values'], flat['returns']
        self.epochs = 0

    def get(self, batch_size):
        import torch
        from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBufferSamples
        if self.epochs >= len(self.permutations):
            raise RuntimeError('Declared epoch budget exhausted')
        permutation = self.permutations[self.epochs]
        self.epochs += 1
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start+batch_size]
            tensor = lambda key: torch.as_tensor(self.flat[key][indices], device=self.device)
            yield MaskableDictRolloutBufferSamples(
                observations=dict(candidate=tensor('obs_candidate'), state=tensor('obs_state')),
                actions=tensor('actions'), old_values=tensor('values').flatten(),
                old_log_prob=tensor('log_probs').flatten(), advantages=tensor('advantages').flatten(),
                returns=tensor('returns').flatten(), action_masks=tensor('action_masks'))


def prepare_clone(source, flat, permutations, reference, rate, beta, logger):
    from training.optimizer_transition import zero_actor_first_moment
    from training.kl_maskable_ppo import KLMaskablePPO
    model = copy.copy(source)
    model.policy = clone_policy(source.policy, source.device)
    model.env = None
    model.rollout_buffer = FixedBuffer(flat, permutations, source.device)
    model.learning_rate = rate
    model.lr_schedule = lambda _: rate
    model.target_kl = None
    model.verbose = 0
    model.kl_beta = beta
    model.set_logger(logger)
    zero_actor_first_moment(model.policy)
    KLMaskablePPO.set_reference(model, reference)
    current = {id(p) for p in model.policy.parameters()}
    teacher = {id(p) for p in model.reference_policy.parameters()}
    if current & teacher or current != {id(p) for g in model.policy.optimizer.param_groups for p in g['params']}:
        raise ValueError('Reference or optimizer aliases a trainable clone')
    return model


def gradient_metrics(model, batch):
    import torch
    from training.kl_maskable_ppo import masked_policy_kl
    from audit_ppo_gradients import gradient_vector, gradient_summary
    values, logs, entropy = model.policy.evaluate_actions(batch.observations,
        batch.actions.long().flatten(), action_masks=batch.action_masks)
    advantages = batch.advantages
    if model.normalize_advantage:
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
    ratio = (logs-batch.old_log_prob).exp()
    clip = model.clip_range(model._current_progress_remaining)
    pg = -torch.min(advantages*ratio, advantages*ratio.clamp(1-clip, 1+clip)).mean()
    ent = -entropy.mean() if entropy is not None else logs.mean()
    value = (batch.returns-values.flatten()).square().mean()
    kl = masked_policy_kl(model.policy, model.reference_policy, batch.observations, batch.action_masks)
    parameters = list(model.policy.parameters())
    gradients = [gradient_vector(loss, parameters) for loss in [pg, ent, value, kl]]
    actors = torch.cat([torch.full((p.numel(),), not name.startswith('value_net.'),
                        dtype=torch.bool, device=p.device) for name, p in model.policy.named_parameters()])
    result = gradient_summary(*gradients[:3], actors, model.ent_coef, model.vf_coef, model.max_grad_norm)
    anchor = model.kl_beta*gradients[3]
    norm = float(anchor[actors].norm())
    result.update(current_to_reference_kl=float(kl.detach()), weighted_anchor_actor_gradient_norm=norm,
        reference_kl_critic_gradient_norm=float(gradients[3][~actors].norm()),
        weighted_anchor_to_actor_pg_norm_ratio=norm/result['actor_policy_gradient_norm']
            if result['actor_policy_gradient_norm'] else None)
    return result


def run(folder):
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import obs_as_tensor
    from training.train_transaction import global_rng_state, restore_global_rng
    from training.train_vector_transaction import training_source_identity
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    from training.kl_maskable_ppo import KLMaskablePPO
    from verify_feature_backend_benchmark import identical
    from resource_guard import now, write_json

    if (folder/'kl_audit.json').exists() or (folder/'steps.jsonl').exists():
        raise FileExistsError('Preserve prior diagnostic; inspect instead of repeating updates')
    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for path, digest in design['frozen_files'].items():
            if sha(path) != digest:
                raise ValueError('Frozen KL diagnostic input changed: '+path)
    unchanged()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    if (training_source_identity() != design['training_sources'] or
            inference_source_identity(resumable=True) != design['inference_sources'] or
            evaluation_runtime_identity(False) != design['evaluation_runtime']):
        raise ValueError('Current diagnostic source/runtime identity differs')
    if design['betas'] != [0., .02] or design['learning_rate'] != 3e-5:
        raise ValueError('Predeclared two-case diagnostic differs')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata, flat = load_data(Path(design['dataset_dir']), design['dataset_metadata_sha256'],
                               design['dataset_sha256'])
    source = MaskablePPO.load(design['model'], device='cuda')
    assert source.get_env() is None and source.device.type == 'cuda'
    assert (source.num_timesteps, source._n_updates) == (metadata['source_steps'], metadata['source_ppo_updates'])
    assert (source.n_steps, source.n_envs, source.batch_size, source.n_epochs) == (512, 8, 256, 5)
    assert source.clip_range_vf is None and source.normalize_advantage
    assert (source.ent_coef, source.vf_coef, source.max_grad_norm) == (.01, .5, .5)
    expected = (copy.deepcopy(source.policy.state_dict()), copy.deepcopy(source.policy.optimizer.state_dict()),
                source.num_timesteps, source._n_updates)
    rng = global_rng_state(source)
    reference = copy.copy(source)
    reference.policy = clone_policy(source.policy, source.device)
    reference.policy.set_training_mode(False)
    for parameter in reference.policy.parameters():
        parameter.requires_grad_(False)
    reference_expected = (copy.deepcopy(reference.policy.state_dict()),
                          copy.deepcopy(reference.policy.optimizer.state_dict()))
    generator = np.random.RandomState(design['permutation_seed'])
    permutations = [generator.permutation(4096) for _ in range(5)]
    write_json(folder/'permutations.json', dict(seed=design['permutation_seed'],
        indices=[p.tolist() for p in permutations], flat_order='env-major: worker*512+step',
        scope='New fixed order, not restored historical sampler RNG'))
    probe_indices = permutations[0][:64]
    probe = obs_as_tensor(dict(candidate=flat['obs_candidate'][probe_indices],
                               state=flat['obs_state'][probe_indices]), source.device)
    masks = torch.as_tensor(flat['action_masks'][probe_indices], device=source.device)
    with torch.no_grad():
        before = reference.policy.get_distribution(probe, masks).distribution.probs.detach().clone()
    cases = {}
    try:
        # Confirm saved values/log probabilities belong to this policy before any update.
        maximum_value_difference = maximum_log_prob_difference = 0.
        for batch in FixedBuffer(flat, [np.arange(4096)], source.device).get(256):
            with torch.no_grad():
                values, logs, _ = source.policy.evaluate_actions(batch.observations,
                    batch.actions.long().flatten(), action_masks=batch.action_masks)
            torch.testing.assert_close(values.flatten(), batch.old_values, rtol=1e-6, atol=1e-4)
            torch.testing.assert_close(logs, batch.old_log_prob, rtol=1e-5, atol=1e-5)
            maximum_value_difference = max(maximum_value_difference, float((values.flatten()-batch.old_values).abs().max()))
            maximum_log_prob_difference = max(maximum_log_prob_difference, float((logs-batch.old_log_prob).abs().max()))
        for case, beta in [('control', 0.), ('original_anchor', .02)]:
            model = prepare_clone(source, flat, permutations, reference, design['learning_rate'], beta,
                                  configure(str(folder/'logs'/case), []))
            fixed_batch = next(FixedBuffer(flat, [permutations[0]], source.device).get(256))
            initial_gradients = gradient_metrics(model, fixed_batch)
            rows = []
            def after_step(optimizer, args, kwargs):
                with torch.no_grad():
                    probs = model.policy.get_distribution(probe, masks).distribution.probs
                    forward = distribution_change(before, probs)
                    reverse = distribution_change(probs, before)['mean_forward_kl']
                row = dict(case=case, step=len(rows)+1, **forward, mean_reverse_kl=reverse)
                rows.append(row)
                with (folder/'steps.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
            hook = model.policy.optimizer.register_step_post_hook(after_step)
            try:
                restore_global_rng(rng)
                KLMaskablePPO.train(model)
            finally:
                hook.remove()
                model.logger.close()
                restore_global_rng(rng)
            if (len(rows) != 80 or model._n_updates-source._n_updates != 5 or
                    model.num_timesteps != source.num_timesteps or model.rollout_buffer.epochs != 5):
                raise ValueError('Actual diagnostic update budget differs')
            old_steps = sorted(int(v['step'].item()) for v in expected[1]['state'].values())
            new_steps = sorted(int(v['step'].item()) for v in model.policy.optimizer.state.values())
            if [step+80 for step in old_steps] != new_steps:
                raise ValueError('Clone Adam counters differ from recorded updates')
            cases[case] = dict(beta=beta, optimizer_steps=80, ppo_epoch_delta=5,
                final_probe=rows[-1], maximum_forward_kl=max(r['mean_forward_kl'] for r in rows),
                maximum_reverse_kl=max(r['mean_reverse_kl'] for r in rows),
                mean_training_reference_kl=float(model.logger.name_to_value['train/reference_kl']),
                initial_fixed_batch_gradients=initial_gradients,
                final_fixed_batch_gradients=gradient_metrics(model, fixed_batch))
            if (not identical(expected, (source.policy.state_dict(), source.policy.optimizer.state_dict(),
                                         source.num_timesteps, source._n_updates)) or
                    not identical(reference_expected, (reference.policy.state_dict(), reference.policy.optimizer.state_dict())) or
                    any(p.grad is not None or p.requires_grad for p in reference.policy.parameters()) or
                    not identical(rng, global_rng_state(source))):
                raise ValueError('Source, reference or sampling RNG changed')
            unchanged()
            write_json(folder/'cases.json', cases)
            del model
        control, anchor = cases['control'], cases['original_anchor']
        ratios = dict(final_forward_kl=anchor['final_probe']['mean_forward_kl']/control['final_probe']['mean_forward_kl'],
            peak_forward_kl=anchor['maximum_forward_kl']/control['maximum_forward_kl'],
            final_reverse_kl=anchor['final_probe']['mean_reverse_kl']/control['final_probe']['mean_reverse_kl'],
            greedy_agreement_delta=anchor['final_probe']['greedy_agreement']-control['final_probe']['greedy_agreement'])
        gate = design['diagnostic_gate']
        supported = (ratios['final_forward_kl'] < gate['final_forward_kl_ratio_below'] and
                     ratios['peak_forward_kl'] < gate['peak_forward_kl_ratio_below'] and
                     ratios['greedy_agreement_delta'] >= gate['greedy_agreement_delta_at_least'])
        write_json(folder/'kl_audit.json', dict(at=now(), status='passed',
            cases=cases, ratios=ratios, diagnostic_gate=gate, diagnostic_supported=supported,
            source_and_reference_policy_adam_counters_unchanged=True, global_sampling_rng_unchanged=True,
            original_optimizer_steps=0, clone_optimizer_steps=160, new_environment_steps=0,
            original_source_counts=[source.num_timesteps, source._n_updates],
            maximum_source_value_difference=maximum_value_difference,
            maximum_source_log_probability_difference=maximum_log_prob_difference,
            eligible_for_training_extension=False, policy_promotion='none',
            scope='One cached CUDA8 rollout, two clone update paths, 64 training probes. '
                  'New fixed permutations; not a historical training reproduction or quality result.'))
    finally:
        restore_global_rng(rng)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
