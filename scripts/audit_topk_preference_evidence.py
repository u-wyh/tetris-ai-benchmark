"""Read-only audit of historical finite-window preferences; no model load or game step."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from own_policy_capture import sha
from training.candidate_features import FEATURE_VERSION, candidate_features
from training.env.tetris_env import TetrisEnv
from training.env.candidate_env import OBSERVATION_VERSION
from training.evaluation.ppo_death_report import compare
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def sealed_bytes(path, frozen):
    path = Path(path).resolve()
    data = path.read_bytes()
    if frozen.get(str(path)) != hashlib.sha256(data).hexdigest():
        raise ValueError('Missing external seal or changed input: '+str(path))
    return data


def sealed_json(path, frozen):
    return json.loads(sealed_bytes(path, frozen))


def public_hash(public):
    return hashlib.sha256(json.dumps(public, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_public(path, meta, candidate, item, frozen):
    compressed = sealed_bytes(path, frozen)
    if hashlib.sha256(compressed).hexdigest() != meta['sha256'] or meta['sha256'] != candidate['snapshot_sha256']:
        raise ValueError('Snapshot seal differs from state/candidate identity')
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        payload = stream.read(4*1024**2+1)
    if len(payload) > 4*1024**2:
        raise ValueError('Unexpectedly large snapshot')
    state = pickle.loads(payload)  # Our local cache, only after external AND historical seals.
    if (state['episode_seed'] != item['seed'] or state['pieces'] != item['step'] or
            state['episode_count'] != 1 or state['max_pieces'] != 100000 or state['finished'] or
            state['candidate_feature_version'] != FEATURE_VERSION or state['core'].rng.state != meta['rng_state']):
        raise ValueError('Snapshot episode, cap, RNG or feature version differs')
    public = state['core'].get_public_observation()
    if public_hash(public) != meta['public_hash'] or meta['public_hash'] != candidate['public_hash']:
        raise ValueError('Snapshot public observation differs from historical hash')
    if public['score'] != candidate['start_score']:
        raise ValueError('Historical candidate initial score differs')
    matrix, mask = candidate_features(public)
    vector = TetrisEnv._encode_observation(public)
    if (matrix.shape != (1840, 16) or matrix.dtype != np.uint8 or mask.shape != (1840,) or
            mask.dtype != np.bool_ or vector.shape != (237,) or vector.dtype != np.float32 or
            not np.isfinite(vector).all() or np.any(vector < 0) or np.any(vector > 1)):
        raise ValueError('Reconstructed public tensor/mask schema differs')
    choices = candidate['choices']
    ids = [c['action'] for c in choices]
    if (not 1 <= len(ids) <= 6 or len(set(ids)) != len(ids) or candidate['ppo_action'] not in ids or
            any(type(a) is not int or not 0 <= a < 1840 or not mask[a] for a in ids)):
        raise ValueError('Historical choice is duplicated or no longer a legal public action')
    if public_hash(public) != meta['public_hash']:
        raise ValueError('Public feature reconstruction mutated its source')
    return vector, matrix, mask


def read_branch(source, item, candidate, choice, horizon, frozen, models):
    path = source/f'counterfactual_{horizon}'/f'seed_{item["seed"]}'/f'step_{item["step"]}_action_{choice["action"]}.json'
    seal = sealed_json(path.with_suffix('.sha256.json'), frozen)
    value = sealed_json(path, frozen)
    if sha(path) != seal['sha256'] or seal['snapshot_sha256'] != candidate['snapshot_sha256']:
        raise ValueError('Historical branch seal changed')
    expected = dict(seed=item['seed'], step=item['step'], first_action=choice['action'], horizon=horizon,
        snapshot_sha256=candidate['snapshot_sha256'], initial_public_hash=candidate['public_hash'],
        initial_score=candidate['start_score'], choice=choice)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Historical branch task identity differs')
    validate_branch(value, horizon)
    explicit = value.get('model_sha256')
    if explicit is not None and explicit != models:
        raise ValueError('Explicit historical branch model differs')
    return value


def validate_branch(value, horizon):
    n = value['pieces']
    if (type(n) is not int or not 1 <= n <= horizon or type(value['game_over']) is not bool or
            type(value['censored']) is not bool or value['game_over'] == value['censored'] or
            (value['censored'] and n != horizon) or len(value['scores']) != n or len(value['four_counts']) != n):
        raise ValueError('Invalid death, censoring or finite branch length')
    previous_score, previous_four = value['initial_score'], 0
    for score, four in zip(value['scores'], value['four_counts']):
        if (type(score) is not int or score < previous_score or type(four) is not int or
                not 0 <= four-previous_four <= 1):
            raise ValueError('Invalid cumulative score or four-clear trajectory')
        previous_score, previous_four = score, four
    if value['score_gain'] != value['scores'][-1]-value['initial_score']:
        raise ValueError('Branch score gain differs from its trajectory')


def extended_choices(candidate, initial, distance):
    """Historical conditional extension rule, without submitting/replaying tasks."""
    base_id = candidate['ppo_action']; base = initial[base_id]; eligible = []
    for choice in candidate['choices']:
        if choice['action'] == base_id:
            continue
        alt = initial[choice['action']]; result = compare(base, alt)
        four_ok = result['base_common_fours'] < 3 or result['four_ratio'] >= .9
        if alt['pieces'] == 1000 and result['score_ratio'] >= .95 and four_ok and (base['game_over'] or distance > 1000):
            eligible.append((result['score_ratio'], choice))
    eligible.sort(key=lambda entry: (-entry[0], entry[1]['action']))
    return [next(c for c in candidate['choices'] if c['action'] == base_id)]+[c for _, c in eligible[:2]] if eligible else []


def preference(base, alt):
    result = compare(base, alt)
    reverse = compare(alt, base)
    if result['strong_local_correction']:
        label = 'beneficial'
    elif reverse['strong_local_correction']:
        label = 'harmful'
    elif result['indeterminate_beyond_window']:
        label = 'both_right_censored_uncertain'
    else:
        label = 'unsupported_within_window'
    return label, result


def deduplicate(rows):
    groups = {}; duplicate_count = 0; conflicts = []
    for row in rows:
        if row['horizon'] != 5000:
            continue
        key = (row['seed'], row['public_hash'], row['action'], row['horizon'])
        core = {k: v for k, v in row.items() if k != 'sources'}
        if key not in groups:
            groups[key] = dict(row, sources=sorted(set(row['sources'])))
        elif {k: v for k, v in groups[key].items() if k != 'sources'} != core:
            conflicts.append(dict(key=list(key), first=groups[key], other=row))
        else:
            duplicate_count += 1
            groups[key]['sources'] = sorted(set(groups[key]['sources']) | set(row['sources']))
    conflicting = {tuple(c['key']) for c in conflicts}
    return [r for k, r in groups.items() if k not in conflicting], duplicate_count, conflicts


def run(folder):
    if (folder/'evidence_audit.json').exists():
        raise FileExistsError('Evidence audit already complete; do not repeat it')
    design = json.loads((folder/'design.json').read_text()); frozen = design['frozen_files']
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise ValueError('Frozen audit input changed: '+path)
    source = Path(design['source_root']).resolve()
    protocol = sealed_json(source/'protocol.json', frozen)
    selection = sealed_json(source/'counterfactual_selection.json', frozen)
    declared = {name: spec['sha256'] for name, spec in design['models'].items()}
    if (protocol['ppo_sha256'] != declared['ppo'] or protocol['bc_sha256'] != declared['bc'] or
            selection['ppo_sha256'] != declared['ppo'] or not 1 <= len(selection['states']) <= 120 or
            any(sha(spec['path']) != spec['sha256'] for spec in design['models'].values())):
        raise ValueError('Historical suite and actual frozen models differ')
    tensors = []; states = []; rows = []; provenance = Counter(); branch_counts = Counter(); actual_paths = set()
    for item in selection['states']:
        seed, step = item['seed'], item['step']
        candidate = sealed_json(source/'counterfactual_candidates'/f'seed_{seed}_step_{step}.json', frozen)
        meta = sealed_json(source/'counterfactual_states'/f'seed_{seed}_step_{step}.pkl.json', frozen)
        trace = sealed_json(source/'traces'/f'seed_{seed}.json', frozen)
        trace_path = source/'traces'/f'seed_{seed}.jsonl.gz'
        if (any(candidate.get(k) != item[k] for k in ['seed', 'step', 'death_distance', 'tier', 'source_trace_sha256']) or
                meta['seed'] != seed or meta['step'] != step or meta['source_trace_sha256'] != item['source_trace_sha256'] or
                trace['ppo_sha256'] != declared['ppo'] or trace['trace_sha256'] != item['source_trace_sha256'] or
                sha(trace_path) != item['source_trace_sha256'] or str(trace_path) not in frozen):
            raise ValueError('State, candidate and original episode trace identity differ')
        tensors.append(load_public(source/'counterfactual_states'/f'seed_{seed}_step_{step}.pkl.gz', meta, candidate, item, frozen))
        state_id = len(states)
        states.append(dict(state_id=state_id, seed=seed, step=step, public_hash=candidate['public_hash'],
            baseline_action=candidate['ppo_action'], snapshot_sha256=candidate['snapshot_sha256']))
        initial = {c['action']: read_branch(source, item, candidate, c, 1000, frozen, declared) for c in candidate['choices']}
        branch_counts[1000] += len(initial)
        provenance['1000_explicit_model_identity'] += sum('model_sha256' in v for v in initial.values())
        extended = extended_choices(candidate, initial, item['death_distance'])
        long = {c['action']: read_branch(source, item, candidate, c, 5000, frozen, declared) for c in extended}
        branch_counts[5000] += len(long)
        provenance['5000_explicit_model_identity'] += sum('model_sha256' in v for v in long.values())
        actual_paths.update(str(source/'counterfactual_5000'/f'seed_{seed}'/f'step_{step}_action_{c["action"]}.json') for c in extended)
        if not extended:
            continue
        base = long[candidate['ppo_action']]
        for choice in extended[1:]:
            alt = long[choice['action']]; label, result = preference(base, alt)
            rows.append(dict(state_id=state_id, seed=seed, step=step, public_hash=candidate['public_hash'],
                snapshot_sha256=candidate['snapshot_sha256'], action=choice['action'], baseline_action=candidate['ppo_action'],
                horizon=5000, sources=choice['sources'], ppo_rank=choice['ppo_rank'], label=label, result=result,
                base_pieces=base['pieces'], alt_pieces=alt['pieces'], base_censored=base['censored'], alt_censored=alt['censored'],
                per_branch_model_identity_explicit='model_sha256' in base and 'model_sha256' in alt))
    present = {str(p) for p in (source/'counterfactual_5000').glob('seed_*/step_*_action_*.json') if '.sha256.' not in p.name}
    if present != actual_paths:
        raise ValueError('Historical conditional extension has missing or extra branches')
    unique, duplicates, conflicts = deduplicate(rows)
    groups = {'all': unique, 'ppo_topk': [r for r in unique if any(s.startswith('ppo_top') for s in r['sources'])],
              'bc': [r for r in unique if 'bc' in r['sources']],
              'both_sources': [r for r in unique if 'bc' in r['sources'] and any(s.startswith('ppo_top') for s in r['sources'])]}
    coverage = {}
    for name, group in groups.items():
        coverage[name] = {label: dict(pairs=len(selected), states=len({r['state_id'] for r in selected}),
            independent_seeds=sorted({r['seed'] for r in selected}))
            for label in ['beneficial', 'harmful', 'both_right_censored_uncertain', 'unsupported_within_window']
            for selected in [[r for r in group if r['label'] == label]]}
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise ValueError('Audit input changed during analysis: '+path)
    np.savez_compressed(folder/'public_inputs.npz',
        observation_state=np.stack([t[0] for t in tensors]), observation_candidate=np.stack([t[1] for t in tensors]),
        action_mask=np.stack([t[2] for t in tensors]), state_id=np.arange(len(tensors), dtype=np.int64))
    durable_json(folder/'preferences.json', dict(states=states, preferences=unique, conflicting_public_contexts=conflicts))
    durable_json(folder/'evidence_audit.json', dict(at=now(), status='passed', states=len(states),
        branches={str(k):v for k,v in branch_counts.items()}, raw_5000_pairs=len(rows), unique_5000_pairs=len(unique),
        duplicate_pairs_removed=duplicates, conflicting_public_context_groups=len(conflicts), provenance=dict(provenance), coverage=coverage,
        suite_declared_models_match_current_zip=True, current_public_reconstruction_verified=True,
        historical_observation_tensor_equality='not recorded; current reconstruction from matching historical public hashes',
        historical_runtime_identity='not pinned; suite protocol commit retained, not relabeled as current31/20',
        per_branch_model_identity_complete=provenance['5000_explicit_model_identity'] == branch_counts[5000],
        original_suite_commit=protocol['git_commit'], observation_version=OBSERVATION_VERSION, feature_version=FEATURE_VERSION,
        model_hashes=declared, frozen_inputs=len(frozen), new_environment_steps=0, gpu_forward_calls=0, optimizer_steps=0,
        eligible_for_training=False, reason='Evidence audit only; branch/runtime provenance, episode isolation and normal-retention protocol require a separate decision',
        artifacts={name:sha(folder/name) for name in ['public_inputs.npz','preferences.json']},
        scope='Fixed conditionally selected5000-window exploratory cases, not325 independent labels, unbiased effect, held-out confirmation or new policy quality. '
              'Only public observation tensors and legal masks exported; seed/step/outcomes are offline grouping/label metadata.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
