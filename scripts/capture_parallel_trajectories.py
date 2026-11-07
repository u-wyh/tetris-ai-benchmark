"""Explicit parallel-dispatch entry point; old frozen capture implementation is reused."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import capture_own_policy_trajectories as capture
from own_policy_capture import read_chain, sha
from parallel_capture import ParallelNativeEnv, advance_parallel, validate_seeds

DISPATCH_VERSION = 'unique-seed-parallel-v1'


def prepare(folder):
    """Read-only checks, including old-cache refusal before locks or unpickling."""
    design = json.loads((folder/'design.json').read_text())
    if design.get('dispatch_version') != DISPATCH_VERSION:
        raise ValueError('An explicit new parallel-dispatch design is required; serial jobs cannot migrate')
    validate_seeds(design['seeds'], 8)
    if design['seeds'] != list(range(100000, 100008)) or design['max_pieces'] != 20000:
        raise ValueError('Frozen CUDA8 capture seed/cap protocol differs')
    for path, digest in design['frozen_files'].items():
        if sha(path) != digest:
            raise ValueError('Frozen parallel capture input changed: '+path)
    if set(design['models']) != {'candidate', 'shadow'}:
        raise ValueError('Both frozen controller and observer models required')
    for spec in design['models'].values():
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('Parallel capture model checksum differs')
    identity = dict(design_sha256=sha(folder/'design.json'), chunk_rounds=256,
        ordered_lanes=design['seeds'], device='cuda', precision='highest', tf32=False,
        batching='active lanes only, in original lane order; no auto-reset')
    snapshot, _, _ = read_chain(folder, identity)
    if snapshot is not None:
        if len(snapshot['workers']) != 8 or len(snapshot['views']) != 8:
            raise ValueError('Missing complete parallel worker snapshot')
        for seed, worker, view in zip(design['seeds'], snapshot['workers'], snapshot['views']):
            state = worker['state']
            if (state['episode_seed'] != seed or state['episode_count'] != 1 or
                    type(worker['active']) is not bool or worker['active'] == state['finished'] or
                    (view is not None) != worker['active']):
                raise ValueError('Parallel cached seed, episode or active slot differs')
    if (folder/'capture_summary.json').exists():
        raise FileExistsError('Capture already complete; inspect instead of restarting it')
    return design, identity


def run(folder, max_chunks):
    design, _ = prepare(folder)
    original_env, original_step = capture.CaptureNativeEnv, capture.advance_round
    try:
        # This standalone process runs one job. Always restore module bindings on exit.
        capture.CaptureNativeEnv = ParallelNativeEnv
        capture.advance_round = partial(advance_parallel, seeds=tuple(design['seeds']))
        capture.run(folder, max_chunks)
    finally:
        capture.CaptureNativeEnv, capture.advance_round = original_env, original_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--max-chunks', type=int, default=20)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.max_chunks)
