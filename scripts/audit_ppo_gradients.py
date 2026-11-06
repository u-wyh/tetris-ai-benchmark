"""One original-context CUDA8 rollout, gradient attribution, no optimizer step."""

import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def snapshot_policy(policy):
    return copy.deepcopy(policy.state_dict())


def gradient_vector(loss, parameters):
    import torch
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, gradients)])


def gradient_summary(pg, entropy, value, actor_mask, ent_coef, vf_coef, max_grad_norm):
    import torch
    norms = lambda vector: float(torch.linalg.vector_norm(vector).item())
    weighted_entropy, weighted_value = ent_coef * entropy, vf_coef * value
    actor_pg, actor_entropy = pg[actor_mask], weighted_entropy[actor_mask]
    pg_norm, entropy_norm = norms(actor_pg), norms(actor_entropy)
    total = pg + weighted_entropy + weighted_value
    total_norm, actor_norm, critic_norm = norms(total), norms(total[actor_mask]), norms(total[~actor_mask])
    clip = min(1., max_grad_norm / (total_norm + 1e-6))
    actor_clip = min(1., max_grad_norm / (actor_norm + 1e-6))
    return dict(actor_policy_gradient_norm=pg_norm, actor_weighted_entropy_gradient_norm=entropy_norm,
        weighted_entropy_to_policy_norm_ratio=entropy_norm / pg_norm if pg_norm else None,
        actor_policy_entropy_cosine=float(torch.dot(actor_pg, actor_entropy).item()) / (pg_norm * entropy_norm)
            if pg_norm and entropy_norm else None,
        actor_value_gradient_norm=norms(value[actor_mask]), critic_policy_gradient_norm=norms(pg[~actor_mask]),
        combined_actor_norm=actor_norm, combined_critic_norm=critic_norm, combined_global_norm=total_norm,
        global_clip_multiplier=clip, actor_only_clip_multiplier=actor_clip,
        global_to_actor_only_clip_multiplier_ratio=clip / actor_clip)


def run(folder, batch_observer=None):
    from importlib.metadata import version
    import numpy as np
    import torch
    from stable_baselines3.common.logger import configure
    from training.train_vector_transaction import restore_task, TransactionBitboardSearchCandidateEnv
    from training.vector_env import make_vector_env
    from verify_feature_backend_benchmark import identical
    from resource_guard import now, write_json

    if (folder/'audit.json').exists():
        raise FileExistsError('Preserve completed gradient audit')
    design = json.loads((folder/'design.json').read_text())
    for name, expected in design['frozen_files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Audit input changed: '+name)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    source = Path(design['source_context'])
    config = json.loads((source/'config.json').read_text())
    assert config['device'] == 'cuda' and config['ppo']['n_envs'] == 8
    env = make_vector_env(8, config['run_seed'], config['max_pieces'],
        env_class=TransactionBitboardSearchCandidateEnv)
    model = None
    try:
        model, _, _ = restore_task(source, design['source_task'], config, env)
        parameters = list(model.policy.parameters())
        names = [name for name, _ in model.policy.named_parameters()]
        actor_mask = torch.cat([torch.full((p.numel(),), not name.startswith('value_net.'),
                    dtype=torch.bool, device=p.device) for name, p in model.policy.named_parameters()])
        before_policy = snapshot_policy(model.policy)
        before_optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
        source_steps, source_updates = model.num_timesteps, model._n_updates
        assert source_steps == design['source_task'] * config['task_steps']
        model.set_logger(configure(str(folder/'logs'), []))
        _, callback = model._setup_learn(4096, reset_num_timesteps=False, progress_bar=False)
        callback.on_training_start({}, {})
        write_json(folder/'audit_progress.json', dict(at=now(),stage='collecting4096_fixed_policy_samples'))
        assert model.collect_rollouts(env, callback, model.rollout_buffer, model.n_steps, use_masking=True)
        callback.on_training_end()
        assert model.num_timesteps - source_steps == 4096
        buffer = model.rollout_buffer
        digest = hashlib.sha256()
        for key, values in sorted(buffer.observations.items()):
            digest.update(key.encode());digest.update(str(values.dtype).encode());digest.update(values.tobytes())
        for key in ['actions','action_masks','returns','advantages','log_probs','values','rewards']:
            values = getattr(buffer, key)
            digest.update(key.encode());digest.update(str(values.dtype).encode());digest.update(values.tobytes())
        rollout_digest = digest.hexdigest()
        rows = []
        model.policy.set_training_mode(True)
        for index, data in enumerate(buffer.get(model.batch_size)):
            actions = data.actions.long().flatten()
            values, log_prob, entropy = model.policy.evaluate_actions(data.observations, actions,
                                                                      action_masks=data.action_masks)
            advantages = data.advantages
            if model.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ratio = torch.exp(log_prob - data.old_log_prob)
            clip = model.clip_range(model._current_progress_remaining)
            policy_loss = -torch.min(advantages * ratio,
                advantages * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
            assert model.clip_range_vf is None, 'Unimplemented value-clipping audit recipe'
            value_loss = torch.nn.functional.mse_loss(data.returns, values.flatten())
            entropy_loss = -entropy.mean() if entropy is not None else log_prob.mean()
            gradients = [gradient_vector(loss, parameters) for loss in (policy_loss, entropy_loss, value_loss)]
            assert all(torch.isfinite(g).all() for g in gradients)
            row = gradient_summary(*gradients, actor_mask, model.ent_coef, model.vf_coef, model.max_grad_norm)
            row.update(batch=index, policy_loss=float(policy_loss.item()), value_loss=float(value_loss.item()),
                       entropy=float(-entropy_loss.item()), max_ratio_deviation=float((ratio-1).abs().max().item()))
            if batch_observer is not None:
                batch_observer(model, data, gradients, actor_mask, index)
            rows.append(row)
            with (folder/'audit_batches.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            write_json(folder/'audit_progress.json', dict(at=now(),stage='fixed_parameter_gradient_batches',completed=index+1))
        assert len(rows) == 4096 // model.batch_size
        assert identical(before_policy, model.policy.state_dict()), 'Diagnostic changed policy'
        assert identical(before_optimizer, model.policy.optimizer.state_dict()), 'Diagnostic changed Adam'
        assert model._n_updates == source_updates and all(p.grad is None for p in parameters)
        for name, expected in design['frozen_files'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
        keys = ['weighted_entropy_to_policy_norm_ratio','actor_policy_entropy_cosine',
                'combined_global_norm','global_clip_multiplier','global_to_actor_only_clip_multiplier_ratio',
                'actor_value_gradient_norm','critic_policy_gradient_norm']
        aggregate = {key:dict(min=min(values),median=statistics.median(values),max=max(values))
                     for key in keys if (values:=[row[key] for row in rows if row[key] is not None])}
        result = dict(at=now(),status='passed',source_context=str(source),source_task=design['source_task'],
            source_num_timesteps=source_steps,source_ppo_updates=source_updates,collected_samples=4096,
            device='cuda',n_envs=8,minibatches=len(rows),rollout_sha256=rollout_digest,
            policy_and_adam_unchanged=True,ppo_update_count_unchanged=True,optimizer_steps=0,
            parameters=names,coefficients=dict(entropy=model.ent_coef,value=model.vf_coef,max_grad_norm=model.max_grad_norm),
            python_version=platform.python_version(),
            runtime_versions={name:version(name) for name in ['torch','numpy','gymnasium','stable-baselines3','sb3-contrib']},
            aggregate=aggregate,per_batch=rows,
            scope='One source-context rollout, one minibatch permutation at fixed original parameters; no optimizer updates. Gradient norms do not determine Adam updates, long-run drift or policy quality.')
        write_json(folder/'audit.json', result)
        write_json(folder/'audit_progress.json', dict(at=now(),stage='completed',completed=len(rows)))
        print(json.dumps(dict(status='passed',aggregate=aggregate)))
    finally:
        if model is not None and hasattr(model, '_logger'):
            model.logger.close()
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
