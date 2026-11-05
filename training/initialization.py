"""Explicit policy and optimizer-state transfer for a new training transaction."""

from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import torch


def validate_initial_context(preset):
    spec = preset.get('initial_context')
    if spec is None:
        return None
    if (preset.get('initial_model') or preset.get('kl_reference')
            or preset.get('failure_curriculum') or preset.get('step0_baseline')
            or preset.get('initial_optimizer', 'fresh') != 'fresh'):
        raise ValueError('Full context initialization cannot combine with transfer, KL, curriculum or step0 baseline')
    if (preset.get('device') != 'cuda' or preset.get('n_envs') != 8
            or not preset.get('candidate_policy')):
        raise ValueError('Full context initialization requires CUDA8 Candidate policy')
    if (not isinstance(spec, dict) or set(spec) != {'run_dir', 'task', 'files_sha256'}
            or not isinstance(spec['run_dir'], str)
            or not Path(spec['run_dir']).is_absolute()
            or type(spec['task']) is not int or spec['task'] < 1):
        raise ValueError('Invalid frozen full context specification')
    required = {'config.json', 'metadata.json', f'committed/task_{spec["task"]:06d}/manifest.json'}
    hashes = spec['files_sha256']
    if (not isinstance(hashes, dict) or set(hashes) != required
            or any(not isinstance(s, str) or len(s) != 64
                   or any(c not in '0123456789abcdef' for c in s) for s in hashes.values())):
        raise ValueError('Full context requires frozen config, metadata and task manifest hashes')
    return deepcopy(spec)


def context_source_config(spec, config):
    """Verify immutable artifacts and recipe before any new run is created."""
    root = Path(spec['run_dir'])
    for name, expected in spec['files_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen full context changed: '+name)
    source = json.loads((root/'config.json').read_text())
    meta = json.loads((root/'metadata.json').read_text())
    for key in ('ppo', 'run_seed', 'max_pieces', 'device', 'candidate_policy', 'policy',
                'task_steps', 'reward', 'reward_version', 'observation_version', 'candidate_feature_version'):
        if source.get(key) != config.get(key):
            raise ValueError('Full context recipe differs: '+key)
    for key, default in [('reward_profile', 'default'), ('hole_penalty_coef', 0.0),
                         ('height_penalty_coef', 0.0)]:
        if source.get(key, default) != config.get(key, default):
            raise ValueError('Full context recipe differs: '+key)
    if source.get('kl_reference') or source.get('failure_curriculum'):
        raise ValueError('Source context uses an unsupported algorithm or curriculum')
    for key, package in [('torch', 'torch'), ('gymnasium', 'gymnasium'),
                         ('stable_baselines3', 'stable-baselines3'), ('sb3_contrib', 'sb3-contrib')]:
        if meta.get(key) != version(package):
            raise RuntimeError('Full context runtime version differs: '+key)
    from training.train_vector_transaction import verify_task
    verify_task(root/'committed'/f'task_{spec["task"]:06d}', spec['task'], source)
    return source


def restore_initial_context(spec, config, env):
    from training.train_vector_transaction import restore_task

    source_config = context_source_config(spec, config)
    model, trainer, _ = restore_task(Path(spec['run_dir']), spec['task'], source_config, env)
    offset = spec['task'] * source_config['task_steps']
    if model.num_timesteps != offset or config['source_steps_offset'] != offset:
        raise RuntimeError('Full context step offset differs')
    # Only the new run's progress counter is rebased. Preserve optimizer updates,
    # saved observations/episode starts, all workers and all restored RNG state.
    model.num_timesteps = 0
    steps = [float(s['step']) for s in model.policy.optimizer.state.values() if 'step' in s]
    return model, dict(mode='full_context', source_steps_offset=offset,
        new_run_start_step=0, source_task=spec['task'], source_run_dir=spec['run_dir'],
        optimizer_state_entries=len(model.policy.optimizer.state),
        minimum_optimizer_step=min(steps), maximum_optimizer_step=max(steps),
        worker_count=len(trainer['sb3_last_episode_starts']),
        original_source_manifest_present='training_source_sha256' in json.loads(
            (Path(spec['run_dir'])/'metadata.json').read_text()),
        provenance='Explicit new-source continuation; not proof of original source identity')


def load_source_model(model_class, path, target):
    """Loading a saved SB3 policy must not reseed the new run's generators."""
    from training.train_transaction import global_rng_state, restore_global_rng

    state = global_rng_state(target)
    try:
        return model_class.load(path, device='cpu')
    finally:
        restore_global_rng(state)


def validate_initial_optimizer(preset):
    mode = preset.get('initial_optimizer', 'fresh')
    if mode not in ('fresh', 'restore_state'):
        raise ValueError('initial_optimizer must be fresh or restore_state')
    if mode == 'restore_state' and not preset.get('initial_model'):
        raise ValueError('Restoring optimizer state requires an initial_model')
    return mode


def initialize_policy(target, source, mode='fresh'):
    """Copy weights; optionally copy moments/step counters with new run options.

    This does not restore environment state, RNG, or PPO timesteps. It is a new
    transfer run; normal transaction resume restores the complete run instead.
    """
    if mode not in ('fresh', 'restore_state'):
        raise ValueError('Unknown optimizer initialization mode')
    source_parameters = [(name, tuple(p.shape)) for name,p in source.named_parameters()]
    if source_parameters != [(name, tuple(p.shape)) for name,p in target.named_parameters()]:
        raise ValueError('Source and target parameter order or shapes differ')
    target.load_state_dict(source.state_dict(), strict=True)
    if mode == 'restore_state':
        if type(source.optimizer) is not type(target.optimizer):
            raise ValueError('Source and target optimizers differ')
        state = deepcopy(source.optimizer.state_dict())
        if not state['state']:
            raise ValueError('Source checkpoint contains no optimizer state')
        for entry in state['state'].values():
            for value in entry.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError('Source optimizer state is nonfinite')
        groups = target.optimizer.param_groups
        if [len(g['params']) for g in state['param_groups']] != [len(g['params']) for g in groups]:
            raise ValueError('Source and target optimizer groups differ')
        options = [{k:deepcopy(v) for k,v in g.items() if k != 'params'} for g in groups]
        target.optimizer.load_state_dict(state)
        for group, configured in zip(target.optimizer.param_groups, options):
            params = group['params']
            group.clear()
            group.update(params=params, **configured)
    steps = [float(s['step']) for s in target.optimizer.state.values() if 'step' in s]
    return dict(optimizer_mode=mode, optimizer_state_entries=len(target.optimizer.state),
                source_optimizer_state_entries=len(source.optimizer.state),
                minimum_optimizer_step=min(steps) if steps else None,
                maximum_optimizer_step=max(steps) if steps else None,
                learning_rates=[group['lr'] for group in target.optimizer.param_groups])
