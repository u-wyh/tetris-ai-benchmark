"""Replay a paired development loss, verify its outcome, and record board risk.

This describes two trajectories; it does not establish why training regressed.
Run through resource_guard.py, after other heavy jobs have finished.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from compare_development_evaluations import compare

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    return dict(observed_pieces=len(rows), max_height=max(r['height'] for r in rows),
                max_holes=max(r['holes'] for r in rows),
                pieces_at_height_15_or_more=sum(r['height'] >= 15 for r in rows),
                pieces_with_holes=sum(r['holes'] > 0 for r in rows),
                line_clear_histogram=dict(sorted(Counter(r['cleared_lines'] for r in rows).items())),
                final_height=rows[-1]['height'], final_holes=rows[-1]['holes'])


def replay(result, seed):
    from sb3_contrib import MaskablePPO
    from training.env.fast_candidate_env import FastCandidateTetrisEnv

    model_path = Path(result['model_checkpoint'])
    if digest(model_path) != result['model_sha256']:
        raise RuntimeError('Evaluation model identity changed')
    expected = next(r for r in result['per_seed'] if r['seed'] == seed)
    model = MaskablePPO.load(str(model_path), device='cpu')
    env = FastCandidateTetrisEnv(max_pieces=result['max_pieces'])
    rows = []
    try:
        observation, _ = env.reset(seed=seed)
        while True:
            masks = env.action_masks()
            action = int(model.predict(observation, deterministic=True, action_masks=masks)[0])
            observation, _, terminated, capped, info = env.step(action)
            rows.append(dict(piece=info['pieces'], action=action, score=info['score'],
                             height=info['height_after'], holes=info['holes_after'],
                             cleared_lines=info['cleared_lines']))
            if terminated or capped:
                actual = dict(seed=seed, pieces_survived=info['pieces'], lines=info['lines'],
                              score=info['score'], episode_reward=env.episode_reward,
                              game_over=bool(terminated), survived_cap=bool(capped))
                if any(actual[k] != expected[k] for k in actual):
                    raise RuntimeError(f'Replay differs from evaluation: {actual} vs {expected}')
                return dict(model_sha256=result['model_sha256'], verified_outcome=actual,
                            full_summary=summarize(rows), last_100_summary=summarize(rows[-100:]),
                            trace=rows)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing diagnostic; choose a new output')
    sys.path.insert(0, str(ROOT))
    import torch
    from training.train_ppo import atomic_json, now
    from training.train_vector_transaction import training_source_identity

    torch.set_num_threads(1)
    baseline, candidate = (json.loads(p.read_text()) for p in (args.baseline, args.candidate))
    paired = compare(baseline, candidate)
    if not paired['comparable']:
        raise ValueError(paired['reason'])
    evaluator = {name: digest(ROOT/'training/evaluation'/name)
                 for name in baseline['evaluation_source_sha256']}
    if evaluator != baseline['evaluation_source_sha256']:
        raise RuntimeError('Evaluator identity changed')
    losses = [p for p in paired['per_seed'] if p['pieces_delta'] < 0]
    if not losses:
        raise ValueError('No paired survival loss to diagnose')
    seed = min(losses, key=lambda p: p['pieces_delta'])['seed']
    sources = training_source_identity()
    left, right = replay(baseline, seed), replay(candidate, seed)
    if sources != training_source_identity():
        raise RuntimeError('Training/environment source changed during replay')
    end = len(right['trace'])
    payload = dict(at=now(), seed=seed, baseline_evaluation_sha256=digest(args.baseline),
                   candidate_evaluation_sha256=digest(args.candidate),
                   evaluation_source_sha256=evaluator, training_source_sha256=sources,
                   diagnostic_source_sha256=digest(Path(__file__)), baseline=left, candidate=right,
                   baseline_through_candidate_end=summarize(left['trace'][:end]),
                   interpretation='Descriptive deterministic replay, exact outcomes verified; '
                                  'different trajectories do not establish a causal training diagnosis')
    atomic_json(args.output, payload)
    print(json.dumps({k: payload[k] for k in ('at', 'seed', 'baseline_through_candidate_end')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
