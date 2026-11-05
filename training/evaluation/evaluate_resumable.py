"""Evaluate with durable per-seed commits; refuse incompatible cache reuse.

The existing frozen evaluator is unchanged. New experiments must collect both
their baseline and candidate with this evaluator's expanded source identity.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import time


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def durable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def validate_row(row, seed, cap, diagnostics=False):
    if row.get('seed') != seed or type(row.get('seed')) is not int:
        raise ValueError('Cached seed identity differs')
    for name in ('pieces_survived', 'lines', 'score'):
        if type(row.get(name)) is not int or row[name] < 0:
            raise ValueError('Invalid cached count: ' + name)
    if row['pieces_survived'] > cap:
        raise ValueError('Cached survival exceeds the evaluation cap')
    if (type(row.get('game_over')) is not bool or type(row.get('survived_cap')) is not bool
            or row['game_over'] == row['survived_cap']
            or (row['survived_cap'] and row['pieces_survived'] != cap)):
        raise ValueError('Cached result has no valid terminal outcome')
    reward = row.get('episode_reward')
    if type(reward) not in (int, float) or not math.isfinite(reward):
        raise ValueError('Invalid cached reward')
    if diagnostics:
        for name in ('new_holes_events', 'new_holes_total', 'holes_at_end'):
            if type(row.get(name)) is not int or row[name] < 0:
                raise ValueError('Missing or invalid diagnostics: ' + name)


def collect_cached_rows(directory, identity, evaluate_pending):
    """Commit each yielded result, including when a later worker fails.

    evaluate_pending receives only missing seeds and yields completed rows in
    any order. A lock protects both the manifest and the sequence of commits.
    """
    directory.mkdir(parents=True, exist_ok=True)
    seeds, cap = identity['seeds'], identity['max_pieces']
    if (not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)
            or type(cap) is not int or cap <= 0):
        raise ValueError('Invalid evaluation seed list or cap')
    expected = dict(version=1, identity=identity, identity_sha256=digest_json(identity))
    with (directory/'cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = directory/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != expected:
                raise ValueError('Evaluation cache identity differs; choose a new cache')
        else:
            durable_json(manifest, expected)
        rows = {}
        for seed in seeds:
            path = directory/f'seed_{seed}.json'
            if path.exists():
                record = json.loads(path.read_text())
                if (record['identity_sha256'] != expected['identity_sha256']
                        or record['row_sha256'] != digest_json(record['row'])):
                    raise ValueError('Cached row integrity check failed')
                validate_row(record['row'], seed, cap, identity.get('diagnostics', False))
                rows[seed] = record['row']
        reused = list(rows)
        pending = [seed for seed in seeds if seed not in rows]
        if pending:
            for row in evaluate_pending(pending):
                seed = row.get('seed')
                if seed not in pending or seed in rows:
                    raise ValueError('Worker returned an unexpected or duplicate seed')
                validate_row(row, seed, cap, identity.get('diagnostics', False))
                durable_json(directory/f'seed_{seed}.json', dict(identity_sha256=expected['identity_sha256'],
                             row_sha256=digest_json(row), row=row))
                rows[seed] = row
        if len(rows) != len(seeds):
            raise ValueError('Evaluation workers did not return all missing seeds')
        return [rows[seed] for seed in seeds], reused


def source_identity():
    from training.train_vector_transaction import training_source_identity

    root = Path(__file__).resolve().parents[2]
    names = ['training/evaluation/evaluate.py', 'training/evaluation/worker_budget.py',
             'training/evaluation/evaluate_resumable.py']
    return {**training_source_identity(),
            **{name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}}


def evaluate(model, seeds, cap, protocol, steps, output, cache, workers=0, diagnostics=False):
    from importlib.metadata import version
    from training.evaluation import evaluate as original
    from training.evaluation.worker_budget import automatic_workers, detect_resources
    from training.train_ppo import now

    if workers < 0:
        raise ValueError('Evaluation workers must be nonnegative')
    began = time.monotonic()
    model = model.resolve()
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    sources = source_identity()
    identity = dict(model_sha256=model_sha, model_checkpoint=str(model), seeds=list(seeds),
                    max_pieces=cap, protocol=protocol, committed_steps=steps,
                    evaluation_source_sha256=sources, diagnostics=diagnostics,
                    runtime_versions={name:version(name) for name in
                                      ['torch','numpy','gymnasium','stable-baselines3','sb3-contrib']},
                    python_version=platform.python_version())
    budget = detect_resources() if workers == 0 else None
    count = automatic_workers(len(seeds), budget) if workers == 0 else workers

    def pending_rows(pending):
        if count == 1:
            original._init_worker(model)
            for seed in pending:
                yield original._evaluate_worker(seed, cap, diagnostics)
        else:
            with ProcessPoolExecutor(max_workers=min(count, len(pending)), mp_context=get_context('spawn'),
                                     initializer=original._init_worker, initargs=(model,)) as pool:
                futures = [pool.submit(original._evaluate_worker, seed, cap, diagnostics) for seed in pending]
                for future in as_completed(futures):
                    yield future.result()

    rows, reused = collect_cached_rows(cache, identity, pending_rows)
    if hashlib.sha256(model.read_bytes()).hexdigest() != model_sha or source_identity() != sources:
        raise RuntimeError('Model or evaluation source changed during this attempt')
    result = dict(**identity, num_seeds=len(seeds), per_seed=rows, aggregate=original.aggregate(rows, cap),
                  wall_seconds=time.monotonic()-began, completed_at=now(), evaluation_workers=count,
                  resumed_completed_seeds=reused, newly_evaluated_seeds=[s for s in seeds if s not in reused],
                  wall_seconds_scope='current attempt only', per_seed_cache=str(cache.resolve()))
    if budget is not None:
        result['automatic_worker_budget'] = budget
    durable_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--seeds-file', type=Path, required=True)
    parser.add_argument('--seed-set', choices=('validation','final_test'), required=True)
    parser.add_argument('--seed-count', type=int, required=True)
    parser.add_argument('--max-pieces', type=int, required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--committed-steps', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    seeds = json.loads(args.seeds_file.read_text())[args.seed_set][:args.seed_count]
    if args.seed_count <= 0 or len(seeds) != args.seed_count:
        parser.error('Seed count mismatch')
    cache = args.cache_dir or args.output.with_suffix('.seeds')
    evaluate(args.model, seeds, args.max_pieces, args.protocol, args.committed_steps,
             args.output, cache, args.workers, args.diagnostics)


if __name__ == '__main__':
    main()
