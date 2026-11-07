"""Frozen CUDA score-head geometry; no candidate weights or optimizer updates."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_frozen(frozen):
    for path, expected in frozen.items():
        if sha(path) != expected:
            raise ValueError('Frozen geometry input changed: '+path)


def validation_seed(seed):
    return int(hashlib.sha256(('topk-head-geometry-v1:'+str(seed)).encode()).hexdigest(), 16) % 4 == 0


def equal_episode_direction(differences, margins, seeds):
    differences = np.asarray(differences, dtype=np.float64)
    margins = np.asarray(margins, dtype=np.float64)
    seeds = np.asarray(seeds)
    if (differences.ndim != 2 or len(differences) == 0 or margins.shape != (len(seeds),) or
            len(differences) != len(seeds) or not np.isfinite(differences).all() or
            not np.isfinite(margins).all()):
        raise ValueError('Invalid episode-weighted direction inputs')
    # sigmoid(-margin), evaluated without overflow; temperature fixed at1.
    weights = np.exp(-np.logaddexp(0., margins))
    per_episode = [(differences[seeds == seed]*weights[seeds == seed, None]).mean(axis=0)
                   for seed in sorted(set(seeds))]
    return np.mean(per_episode, axis=0)


def normal_crossing(margins, slopes, tolerance):
    """Conservative first loss of strict reference order for this one direction."""
    margins, slopes = np.asarray(margins), np.asarray(slopes)
    if (margins.shape != slopes.shape or margins.ndim != 1 or len(margins) == 0 or
            tolerance < 0 or not np.isfinite(margins).all() or not np.isfinite(slopes).all()):
        raise ValueError('Invalid normal geometry')
    times = np.full(len(margins), np.inf, dtype=np.float64)
    tied = margins <= tolerance
    times[tied] = 0.
    decreasing = (slopes < 0) & ~tied
    times[decreasing] = margins[decreasing]/-slopes[decreasing]
    index = int(np.argmin(times))
    return float(times[index]), index, int(tied.sum())


def preference_crossing(margin, slope):
    if not math.isfinite(margin) or not math.isfinite(slope):
        raise ValueError('Nonfinite preference geometry')
    if margin > 0:
        return 0.  # Already ahead, not a newly repaired order.
    return max(0., -margin/slope) if slope > 0 else math.inf


def number(value):
    return 'infinity' if math.isinf(value) else float(value)


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0, quantiles={})
    return dict(count=len(values), zero=sum(v == 0 for v in values),
                infinite=sum(math.isinf(v) for v in values),
                quantiles={str(p): number(values[math.ceil(p*(len(values)-1))])
                           for p in (0., .25, .5, .75, .9, .95, 1.)},
                convention='nearest upper order statistic, including infinity')


def load_public(folder, frozen):
    paths = [folder/n for n in ('evidence_audit.json', 'preferences.json', 'public_inputs.npz')]
    for path in paths:
        if str(path) not in frozen or sha(path) != frozen[str(path)]:
            raise ValueError('Public input missing external seal or checksum differs')
    audit, preferences = [json.loads(path.read_text()) for path in paths[:2]]
    if (audit.get('status') != 'passed' or audit.get('states') != 120 or
            audit.get('eligible_for_training') is not False or
            audit.get('per_branch_model_identity_complete') is not False or
            preferences.get('conflicting_public_contexts') != []):
        raise ValueError('Preference evidence scope differs')
    if any(audit['artifacts'][path.name] != frozen[str(path)] for path in paths[1:]):
        raise ValueError('Evidence artifact seal differs')
    with np.load(paths[2], allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    specs = {'observation_state': ((120, 237), np.float32),
             'observation_candidate': ((120, 1840, 16), np.uint8),
             'action_mask': ((120, 1840), np.bool_), 'state_id': ((120,), np.int64)}
    if (set(arrays) != set(specs) or any(arrays[k].shape != shape or arrays[k].dtype != dtype or
            not np.isfinite(arrays[k]).all() for k, (shape, dtype) in specs.items()) or
            not np.array_equal(arrays['state_id'], np.arange(120)) or
            not arrays['action_mask'].any(axis=1).all()):
        raise ValueError('Public observation schema differs')
    states, pairs = preferences['states'], preferences['preferences']
    if len(states) != 120 or [s['state_id'] for s in states] != list(range(120)):
        raise ValueError('Public state order differs')
    seen = set()
    for pair in pairs:
        i, a, b = pair['state_id'], pair['action'], pair['baseline_action']
        if not isinstance(i, int) or not 0 <= i < 120:
            raise ValueError('Invalid preference state')
        state = states[i]
        if (any(pair[k] != state[k] for k in state) or pair['horizon'] != 5000 or
                pair['label'] not in ('beneficial', 'harmful', 'unsupported_within_window',
                                      'both_right_censored_uncertain') or
                not pair['sources'] or not all(isinstance(s, str) for s in pair['sources']) or
                a == b or not 0 <= a < 1840 or not 0 <= b < 1840 or
                not arrays['action_mask'][i, a] or not arrays['action_mask'][i, b] or (i, a) in seen):
            raise ValueError('Preference source, identity or action differs')
        seen.add((i, a))
    for state in states:
        if not arrays['action_mask'][state['state_id'], state['baseline_action']]:
            raise ValueError('Illegal stored baseline action')
    top = [p for p in pairs if any(s.startswith('ppo_top') for s in p['sources'])]
    if sum(p['label'] == 'beneficial' for p in top) != 177:
        raise ValueError('Fixed Top-K selection differs')
    return arrays, states, top, audit


def run(folder):
    import torch
    from sb3_contrib import MaskablePPO
    from training.train_transaction import global_rng_state
    from training.candidate_features import candidate_features
    from training.env.tetris_env import TetrisEnv
    from restore_rank_contexts import observation_digest
    from restore_own_death_contexts import load_verified_chunk
    from verify_feature_backend_benchmark import identical
    from training.evaluation.evaluate_resumable import durable_json
    from resource_guard import now
    design = json.loads((folder/'geometry_design.json').read_text())
    check_frozen(design['frozen_files'])
    if (folder/'geometry_result.json').exists() or (folder/'execution.json').exists():
        raise FileExistsError('Geometry attempt already exists; inspect instead of repeating forwards')
    receipt = dict(at=now(), pid=os.getpid(), status='started', gpu_forward_attempts=0,
                   gpu_forward_calls=0, environment_steps=0, optimizer_steps=0)
    durable_json(folder/'execution.json', {**receipt, 'gpu_forward_attempts': None,
        'gpu_forward_calls': None, 'scope': 'Started attempt; totals unverified until final receipt. No blind retry.'})
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no host/CPU policy fallback')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        arrays, states, pairs, evidence = load_public(Path(design['preferences_dir']), design['frozen_files'])
        source = MaskablePPO.load(design['model'], device='cuda', env=None)
        if (source.get_env() is not None or source.device.type != 'cuda' or
                [source.num_timesteps, source._n_updates] != [30003200, 36625] or
                sha(design['model']) != evidence['model_hashes']['ppo']):
            raise ValueError('Original CUDA source identity/counts differ')
        policy = source.policy
        policy.set_training_mode(False)
        before = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict(),
                                global_rng_state(source), source._n_updates))
        head = policy.score_head[-1]
        if not isinstance(head, torch.nn.Linear) or (head.in_features, head.out_features) != (64, 1):
            raise ValueError('Final score-head architecture differs')
        w = head.weight.detach().cpu().numpy().astype(np.float64).reshape(-1)
        bias = float(head.bias.item())
        max_error = 0.

        def features(observations, masks):
            nonlocal max_error
            if len(masks) != 8 or receipt['gpu_forward_attempts'] >= 94:
                raise ValueError('Fixed CUDA8 forward budget differs')
            captured = []
            def hook(module, inputs, output):
                captured.append((inputs[0].detach().cpu().numpy().astype(np.float64),
                                 output.detach().cpu().numpy().reshape(-1).astype(np.float64)))
            handle = head.register_forward_hook(hook)
            receipt['gpu_forward_attempts'] += 1
            try:
                with torch.inference_mode():
                    obs = {k: torch.as_tensor(v, device='cuda') for k, v in observations.items()}
                    dist = policy.get_distribution(obs, action_masks=masks)
                    modes = dist.distribution.probs.argmax(dim=1).cpu().numpy()
                receipt['gpu_forward_calls'] += 1
            finally:
                handle.remove()
            if len(captured) != 1:
                raise ValueError('Expected one final head call per batch')
            latent, scores = captured[0]
            if len(latent) != int(masks.sum()) or not np.isfinite(latent).all() or not np.isfinite(scores).all():
                raise ValueError('Invalid captured legal features')
            error = float(np.max(np.abs(latent @ w+bias-scores)))
            max_error = max(max_error, error)
            if error > 1e-4:
                raise ValueError('Frozen float64 score reconstruction exceeds1e-4')
            offsets = np.r_[0, np.cumsum(masks.sum(axis=1))]
            result = []
            for i, mask in enumerate(masks):
                result.append(dict(actions=np.flatnonzero(mask), h=latent[offsets[i]:offsets[i+1]],
                                   mode=int(modes[i]), reconstruction_error=error))
            return result

        dangerous = []
        for start in range(0, 120, 8):
            dangerous.extend(features({'state': arrays['observation_state'][start:start+8],
                'candidate': arrays['observation_candidate'][start:start+8]}, arrays['action_mask'][start:start+8]))
        mismatches = [dict(state_id=i, seed=states[i]['seed'], step=states[i]['step'],
                           historical_cpu_action=states[i]['baseline_action'], current_cuda_mode=view['mode'])
                      for i, view in enumerate(dangerous) if view['mode'] != states[i]['baseline_action']]
        pair_geometry = []
        differences = []
        for pair in pairs:
            view = dangerous[pair['state_id']]
            indices = {int(a): i for i, a in enumerate(view['actions'])}
            delta = view['h'][indices[pair['action']]]-view['h'][indices[pair['baseline_action']]]
            margin = float(delta @ w)
            differences.append(delta)
            pair_geometry.append(dict(state_id=pair['state_id'], seed=pair['seed'], action=pair['action'],
                baseline_action=pair['baseline_action'], label=pair['label'], margin=margin,
                split='validation' if validation_seed(pair['seed']) else 'train'))
        train = [i for i, p in enumerate(pair_geometry) if p['label'] == 'beneficial' and p['split'] == 'train']
        g = equal_episode_direction(np.array(differences)[train], [pair_geometry[i]['margin'] for i in train],
                                   [pair_geometry[i]['seed'] for i in train])
        norm = float(np.linalg.norm(g))
        for delta, pair in zip(differences, pair_geometry):
            pair['slope'] = float(delta @ g)
            pair['crossing'] = preference_crossing(pair['margin'], pair['slope'])
            pair['initial_numerical_tie'] = abs(pair['margin']) <= design['tie_tolerance']

        normal = []
        base = Path(design['normal_dir'])
        previous = None
        previous_manifest = None
        for index in range(80):
            directory = base/f'chunk_{index:04d}'
            saved, rows, manifest = load_verified_chunk(directory,
                design['frozen_files'][str(directory/'manifest.json')], design['normal_identity'])
            if (manifest['number'] != index or manifest['previous_manifest_sha256'] != previous_manifest or
                    saved['round'] != min(index*256, 20000) or saved['environment_steps'] != saved['round']*8):
                raise ValueError('Normal source capture chain differs')
            if previous is not None:
                if (list(rows['lane'][:8]) != list(range(8)) or
                        not (rows['round'][:8] == previous['round']).all()):
                    raise ValueError('Normal source next-round teacher rows differ')
                observations = []; masks = []
                for lane, worker in enumerate(previous['workers']):
                    state = worker['state']
                    if (not worker['active'] or state['finished'] or state['episode_count'] != 1 or
                            state['episode_seed'] != 100000+lane):
                        raise ValueError('Normal source episode/active lane differs')
                    public = state['core'].get_public_observation()
                    matrix, mask = candidate_features(public)
                    observation = dict(state=TetrisEnv._encode_observation(public), candidate=matrix)
                    if observation_digest(observation, mask) != previous['views'][lane]:
                        raise ValueError('Normal source public observation or legal mask differs')
                    digest = observation_digest(observation, mask)
                    if (digest != dict(observation=str(rows['observation_sha256'][lane]),
                                       mask=str(rows['mask_sha256'][lane])) or
                            int(rows['episode_seed'][lane]) != 100000+lane):
                        raise ValueError('Normal teacher observation/episode differs from snapshot')
                    observations.append(observation); masks.append(mask)
                views = features({k: np.stack([o[k] for o in observations]) for k in observations[0]}, np.stack(masks))
                for lane, view in enumerate(views):
                    if view['mode'] != int(rows['action'][lane]):
                        raise ValueError('Current GPU mode differs from normal saved teacher action')
                    a0 = int(np.flatnonzero(view['actions'] == view['mode'])[0])
                    competitors = np.arange(len(view['actions'])) != a0
                    delta = view['h'][a0]-view['h'][competitors]
                    margins, slopes = delta @ w, delta @ g
                    t, j, ties = normal_crossing(margins, slopes, design['tie_tolerance'])
                    # Separate more conservative finite-precision envelope, fixed rule before outcomes.
                    safe, _, _ = normal_crossing(margins, slopes,
                        max(design['tie_tolerance'], 2*view['reconstruction_error']))
                    normal.append(dict(chunk=index-1, round=previous['round'], lane=lane, seed=100000+lane,
                        split='train' if lane < 4 else 'validation', action=view['mode'],
                        crossing=t, conservative_crossing=safe, numerical_ties=ties,
                        crossing_competitor=int(view['actions'][competitors][j]),
                        competitor_margin=float(margins[j]), competitor_slope=float(slopes[j]),
                        minimum_initial_margin=float(margins.min())))
            previous, previous_manifest = saved, design['frozen_files'][str(directory/'manifest.json')]
        if len(normal) != 632 or receipt['gpu_forward_calls'] != 94:
            raise ValueError('Fixed normal sample/forward count differs')
        bound = min(p['crossing'] for p in normal if p['split'] == 'train')
        safe_bound = min(p['conservative_crossing'] for p in normal if p['split'] == 'train')
        for pair in pair_geometry:
            pair['already_ahead'] = pair['margin'] > 0
            pair['repair_before_first_train_normal_crossing'] = (
                norm > 0 and pair['margin'] <= 0 and pair['slope'] > 0 and pair['crossing'] < bound)
            pair['repair_before_conservative_train_bound'] = (
                norm > 0 and pair['margin'] <= 0 and pair['slope'] > 0 and pair['crossing'] < safe_bound)
            pair['crossing'] = number(pair['crossing'])
        unchanged = identical(before, (policy.state_dict(), policy.optimizer.state_dict(),
                                     global_rng_state(source), source._n_updates))
        if not unchanged:
            raise ValueError('Original policy/Adam/counts/sampling RNG changed')
        check_frozen(design['frozen_files'])
        result = dict(at=now(), status='passed', model_counts=[source.num_timesteps, source._n_updates],
            source_policy_adam_counts_rng_unchanged=unchanged, direction_norm=norm,
            direction_scaling='unnormalized seed-equal mean softplus descent at temperature1; no parameter step',
            train_preference_seeds=sorted({p['seed'] for p in pair_geometry if p['split'] == 'train' and p['label'] == 'beneficial'}),
            validation_preference_seeds=sorted({p['seed'] for p in pair_geometry if p['split'] == 'validation' and p['label'] == 'beneficial'}),
            max_score_reconstruction_error=max_error, dangerous_current_vs_historical_mode_mismatches=mismatches,
            normal_teacher_action_matches=632, normal_train_bound=number(bound), conservative_train_bound=number(safe_bound),
            normal_crossing_by_split={split: distribution([p['crossing'] for p in normal if p['split'] == split])
                                      for split in ('train', 'validation')},
            normal_crossing_by_seed={str(seed): distribution([p['crossing'] for p in normal if p['seed'] == seed])
                                     for seed in range(100000, 100008)},
            preference_groups={split: {label: dict(count=len(group),
                already_ahead=sum(p['already_ahead'] for p in group), positive_slope=sum(p['slope'] > 0 for p in group),
                possible_pairwise_repairs_before_normal=sum(p['repair_before_first_train_normal_crossing'] for p in group),
                possible_repairs_before_conservative_bound=sum(p['repair_before_conservative_train_bound'] for p in group))
                for label in ('beneficial', 'harmful', 'unsupported_within_window')
                for group in [[p for p in pair_geometry if p['split'] == split and p['label'] == label]]}
                for split in ('train', 'validation')},
            preference_pairs=pair_geometry,
            normal_states=[{**p, 'crossing': number(p['crossing']),
                            'conservative_crossing': number(p['conservative_crossing'])} for p in normal],
            historical_branch_model_identity_complete=False, eligible_for_training=False,
            scope='One fixed direction on correlated, previously explored, conditionally selected episodes. Pairwise order only, not globalTop1, an updated GPU policy, quality or proof about other directions. Historical CPU/runtime/model branch gaps remain. Global RNG comparison starts after source load.',
            work=dict(gpu_forward_calls=94, environment_steps=0, optimizer_steps=0, model_loads=1,
                      new_environments=0, candidate_weight_updates=0))
        durable_json(folder/'geometry_result.json', result)
        receipt.update(status='completed', at=now(), original_state_unchanged=unchanged)
    except BaseException as exc:
        receipt.update(status='failed', at=now(), error=repr(exc))
        raise
    finally:
        durable_json(folder/'execution.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    run(parser.parse_args().run_dir.resolve())
