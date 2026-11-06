"""Cross-check completed own-policy traces against independent worker endpoints."""
import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from own_policy_capture import read_chain, sha
from verify_feature_backend_benchmark import identical
from training.evaluation.evaluate_resumable import durable_json
from training.candidate_features import FEATURE_NAMES
from resource_guard import now


def run(folder, upstream_summary):
    if json.loads(upstream_summary.read_text())['status'] != 'passed':
        raise ValueError('Capture protection/controller did not finish successfully')
    design = json.loads((folder/'design.json').read_text())
    summary = json.loads((folder/'capture_summary.json').read_text())
    identity = summary['identity']
    if summary['status'] != 'passed' or identity['design_sha256'] != sha(folder/'design.json'):
        raise ValueError('Capture is incomplete or design identity differs')
    snapshot, last_hash, chunks = read_chain(folder, identity)
    if len(snapshot['workers']) != 8 or any(w['active'] for w in snapshot['workers']):
        raise ValueError('Not all eight capture lanes have stopped')
    initial = pickle.loads((folder/'chunk_0000/state.pkl').read_bytes())  # read_chain checked ALL files first
    if not identical(initial['rng'], snapshot['rng']):
        raise ValueError('Parent global RNG changed across complete capture')
    lanes = [dict(seed=seed, steps=0, reward=0., lines=0, disagreements=0,
        first_disagreement_step=None, terminated=0, truncated=0, terminal_row=None,
        final_32_disagreements=[]) for seed in design['seeds']]
    total_rows = 0
    clear_index = FEATURE_NAMES.index('cleared_lines')
    for number in range(1, chunks):
        directory = folder/f'chunk_{number:04d}'
        manifest = json.loads((directory/'manifest.json').read_text())
        audit = manifest['audit']
        if (audit['policy_adam_counts_unchanged'] is not True or
                audit['global_sampling_rng_unchanged'] is not True or audit['optimizer_steps'] != 0 or
                audit['model_counts'] != {name: spec['counts'] for name, spec in design['models'].items()}):
            raise ValueError('Frozen model or sampling audit failed')
        with np.load(directory/'rows.npz', allow_pickle=False) as archive:
            # NpzFile.__getitem__ decompresses on every access. Materialize each
            # bounded chunk once before checking individual rows.
            data = {key: archive[key] for key in archive.files}
            count = manifest['rows']
            if any(len(data[key]) != count or data[key].dtype.hasobject for key in data):
                raise ValueError('Trace array length or dtype differs')
            if data['candidate_features'].shape != (count, 16) or data['shadow_features'].shape != (count, 16):
                raise ValueError('Selected feature shape differs')
            if (data['candidate_features'].dtype != np.uint8 or data['shadow_features'].dtype != np.uint8 or
                    not np.array_equal(data['candidate_features'][:, clear_index], data['cleared_lines']) or
                    any(np.any(data[key] < 0) or np.any(data[key] >= 1840)
                        for key in ['action', 'shadow_action']) or
                    not np.array_equal(np.lexsort((data['lane'], data['round'])), np.arange(count))):
                raise ValueError('Action, feature/outcome alignment or ordered active lanes differ')
            if (not np.isfinite(data['reward']).all() or
                    any(np.any(~np.isfinite(data[key])) or np.any(data[key] < 0) or np.any(data[key] > 1)
                        for key in ['candidate_probability', 'shadow_probability', 'shadow_probability_of_executed']) or
                    any(np.any(np.isnan(data[key])) or np.any(data[key] < 0)
                        for key in ['candidate_margin', 'shadow_margin'])):
                raise ValueError('Invalid observed policy probabilities, margins or reward')
            for row in range(count):
                lane = int(data['lane'][row])
                if not 0 <= lane < 8:
                    raise ValueError('Invalid trace lane')
                current = lanes[lane]
                if (current['terminated'] or current['truncated'] or
                        int(data['round'][row]) != current['steps'] or
                        int(data['pieces'][row]) != current['steps']+1 or
                        int(data['episode_seed'][row]) != current['seed']):
                    raise ValueError('Noncontiguous own trajectory, repeated episode or steps after endpoint')
                current['steps'] += 1
                current['reward'] += float(data['reward'][row])
                current['lines'] += int(data['cleared_lines'][row])
                differs = bool(data['action'][row] != data['shadow_action'][row])
                current['disagreements'] += differs
                current['final_32_disagreements'].append(differs)
                current['final_32_disagreements'] = current['final_32_disagreements'][-32:]
                if differs and current['first_disagreement_step'] is None:
                    current['first_disagreement_step'] = current['steps']
                dead, truncated = bool(data['terminated'][row]), bool(data['truncated'][row])
                if dead and truncated:
                    raise ValueError('Death and cap cannot both apply')
                current['terminated'] += dead; current['truncated'] += truncated
                if dead or truncated:
                    current['terminal_row'] = dict(action=int(data['action'][row]),
                        shadow_action=int(data['shadow_action'][row]),
                        candidate_clear_feature=int(data['candidate_features'][row, clear_index]),
                        shadow_clear_feature=int(data['shadow_features'][row, clear_index]),
                        candidate_probability=float(data['candidate_probability'][row]),
                        shadow_probability_of_executed=float(data['shadow_probability_of_executed'][row]),
                        holes_before=int(data['holes_before'][row]), holes_after=int(data['holes_after'][row]),
                        height_after=int(data['height_after'][row]))
                total_rows += 1
    for lane, (statistics, worker, endpoint) in enumerate(zip(lanes, snapshot['workers'], summary['endpoints'])):
        state = worker['state']; core = state['core']
        if (statistics['terminated']+statistics['truncated'] != 1 or
                statistics['steps'] != state['pieces'] or not state['finished'] or
                state['episode_seed'] != statistics['seed'] or state['episode_count'] != 1 or
                statistics['lines'] != core.lines or not np.isclose(statistics['reward'], state['episode_reward'], rtol=0, atol=1e-8) or
                bool(statistics['terminated']) != (core.phase == 'over') or
                bool(statistics['truncated']) != (state['pieces'] == 20000 and core.phase != 'over') or
                endpoint != dict(lane=lane, seed=state['episode_seed'], steps=state['pieces'], score=core.score,
                                 lines=core.lines, game_over=core.phase == 'over',
                                 cap_censored=state['pieces'] == 20000 and core.phase != 'over')):
            raise ValueError('Trace endpoint disagrees with complete worker state or summary')
        for key in ['numpy_generator', 'action_generator', 'numpy_seed']:
            if not identical(initial['workers'][lane]['state'][key], state[key]):
                raise ValueError('Worker reset/action-sampling RNG changed during single deterministic episode')
        statistics['last_32_disagreements'] = sum(statistics.pop('final_32_disagreements'))
    processes = [json.loads(path.read_text()) for path in folder.glob('process_*.json')]
    if (total_rows != snapshot['environment_steps'] or total_rows != summary['environment_steps'] or
            total_rows > 160000 or sum(p['environment_steps_this_process'] for p in processes) != total_rows or
            sum(p['candidate_forward_calls_this_process'] for p in processes) != snapshot['round'] or
            sum(p['shadow_forward_calls_this_process'] for p in processes) != snapshot['round'] or
            any(p['optimizer_steps'] != 0 for p in processes)):
        raise ValueError('Actual process work, committed trace and summary accounting differ')
    durable_json(folder/'trajectory_audit.json', dict(at=now(), status='passed',
        identity=identity, chunks=chunks, last_manifest_sha256=last_hash,
        actual_unique_environment_steps=total_rows, actual_candidate_forward_calls=snapshot['round'],
        actual_shadow_forward_calls=snapshot['round'], optimizer_steps=0, lanes=lanes,
        process_ids=sorted(p['pid'] for p in processes), complete_worker_trace_endpoints_exact=True,
        global_and_worker_reset_sampling_rng_unchanged=True,
        interpretation='Shadow is observed on candidate own states only; disagreement is descriptive, '
            'not a counterfactual survival result. CUDA active batches do not replace the old CPU quality gate.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--upstream-summary', type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.upstream_summary.resolve())
