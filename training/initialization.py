"""Explicit policy and optimizer-state transfer for a new training transaction."""

from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path


def context_learning_rate(spec):
    rate = spec.get('learning_rate_after_restore')
    if 'learning_rate_after_restore' in spec and (
            isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or rate <= 0):
        raise ValueError('learning_rate_after_restore must be finite and positive')
    return rate


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
    required_keys = {'run_dir', 'task', 'files_sha256'}
    if (not isinstance(spec, dict) or not required_keys <= set(spec)
            or not set(spec) <= required_keys | {'learning_rate_after_restore'}
            or not isinstance(spec['run_dir'], str)
            or not Path(spec['run_dir']).is_absolute()
            or type(spec['task']) is not int or spec['task'] < 1):
        raise ValueError('Invalid frozen full context specification')
    context_learning_rate(spec)
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
    if source.get('initial_context') or source.get('source_steps_offset', 0) != 0:
        raise ValueError('Nested context offsets require a separate continuation protocol')
    expected_offset = spec['task'] * source['task_steps']
    if config.get('source_steps_offset', expected_offset) != expected_offset:
        raise ValueError('Recorded context source step offset differs')
    target_ppo = deepcopy(source['ppo'])
    transition = context_learning_rate(spec)
    if transition is not None:
        if transition > source['ppo']['learning_rate']:
            raise ValueError('Context learning-rate transition only supports reductions')
        target_ppo['learning_rate'] = transition
    if config['ppo'] != target_ppo:
        raise ValueError('Full context recipe differs: ppo')
    for key in ('run_seed', 'max_pieces', 'device', 'candidate_policy', 'policy',
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


def validate_context_model_recipe(model, config):
    """Check actual deserialized PPO fields, including constant SB3 schedules."""
    ppo = config['ppo']
    for key, expected in ppo.items():
        if key != 'clip_range' and getattr(model, key) != expected:
            raise ValueError('Loaded context model recipe differs: '+key)
    if model.seed != config['run_seed']:
        raise ValueError('Loaded context model seed differs')
    for key, expected in [('normalize_advantage', True), ('clip_range_vf', None), ('target_kl', None)]:
        if getattr(model, key) != expected:
            raise ValueError('Loaded context model default differs: '+key)
    for name, expected in [('lr_schedule', ppo['learning_rate']), ('clip_range', ppo['clip_range'])]:
        schedule = getattr(model, name)
        # Scope this entry to SB3's actual constant representation. Merely
        # checking a few progress values cannot prove an arbitrary callable constant.
        cls = type(schedule)
        inner = getattr(schedule, 'value_schedule', None)
        if (cls.__module__ != 'stable_baselines3.common.utils' or cls.__name__ != 'FloatSchedule'
                or type(inner).__module__ != 'stable_baselines3.common.utils'
                or type(inner).__name__ != 'ConstantSchedule' or inner.val != expected):
            raise ValueError('Loaded context model schedule differs: '+name)
    for group in model.policy.optimizer.param_groups:
        if group['lr'] != ppo['learning_rate']:
            raise ValueError('Loaded context optimizer learning rate differs')


def restore_initial_context(spec, config, env):
    from training.train_vector_transaction import restore_task

    source_config = context_source_config(spec, config)
    model, trainer, _ = restore_task(Path(spec['run_dir']), spec['task'], source_config, env)
    validate_context_model_recipe(model, source_config)
    offset = spec['task'] * source_config['task_steps']
    if model.num_timesteps != offset or config['source_steps_offset'] != offset:
        raise RuntimeError('Full context step offset differs')
    rate = config['ppo']['learning_rate']
    if context_learning_rate(spec) is not None:
        from stable_baselines3.common.utils import FloatSchedule
        model.learning_rate = rate
        model.lr_schedule = FloatSchedule(rate)
        model.policy.optimizer.defaults['lr'] = rate
        for group in model.policy.optimizer.param_groups:
            group['lr'] = rate
    validate_context_model_recipe(model, config)
    # Only the new run's progress counter is rebased. Preserve optimizer updates,
    # saved observations/episode starts, all workers and all restored RNG state.
    model.num_timesteps = 0
    steps = [float(s['step']) for s in model.policy.optimizer.state.values() if 'step' in s]
    return model, dict(mode='full_context', source_steps_offset=offset,
        source_learning_rate=source_config['ppo']['learning_rate'],
        learning_rate_after_restore=rate,
        learning_rate_transition_declared='learning_rate_after_restore' in spec,
        source_ppo_update_count=model._n_updates,
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
    import torch

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
