"""Profile observation construction on committed local worker snapshots.

Run under resource_guard after other compute has finished. This is a static
snapshot profile, not training, an inference benchmark, or a throughput claim.
"""

import argparse
import cProfile
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import pstats
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--search-backend', choices=['formal','bitboard'], default='formal')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.output.exists():
        raise FileExistsError('Preserve previous profile evidence')

    from training.evaluation.fast_candidate_features import candidate_features
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    if args.search_backend == 'bitboard':
        from training.evaluation.fast_legal import get_legal_placements
    else:
        from training.tetris_core.legal_placements import get_legal_placements
    from resource_guard import now, operation, write_json

    sources = inference_source_identity(resumable=True)
    sources['scripts/profile_candidate_construction.py'] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    snapshots, cores = {}, []
    for path in args.snapshot:
        payload = path.read_bytes()
        snapshots[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
        # The caller supplies this project's own committed local artifacts.
        states = pickle.loads(payload)
        if not isinstance(states, list) or len(states) != 8:
            raise ValueError('Expected a committed eight-worker snapshot')
        for worker, state in enumerate(states):
            core = state['core']
            if 'get_legal_placements' in core.__dict__:
                raise RuntimeError('Snapshot has a temporary placement override')
            cores.append((str(path.resolve()), worker, core))

    def construct(core):
        public = core.get_public_observation()
        placements = get_legal_placements(public)
        features, mask = candidate_features(public, placements)
        return len(placements), features, mask

    def signature(core):
        count, features, mask = construct(core)
        return count, hashlib.sha256(features.tobytes() + mask.tobytes()).hexdigest()

    before = [signature(core) for _, _, core in cores]
    profile = cProfile.Profile()
    began_wall, began_cpu = time.perf_counter(), time.process_time()
    profile.enable()
    for _ in range(args.repeats):
        for _, _, core in cores:
            construct(core)
    profile.disable()
    wall, cpu = time.perf_counter() - began_wall, time.process_time() - began_cpu
    if before != [signature(core) for _, _, core in cores]:
        raise RuntimeError('Construction changed outputs while profiling')
    for name, expected in {**snapshots, **sources}.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Profile input changed: ' + name)
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in pstats.Stats(profile).stats.items():
        try:
            filename = str(Path(filename).relative_to(ROOT))
        except ValueError:
            pass
        rows.append(dict(file=filename, line=line, function=function,
                         primitive_calls=primitive, calls=calls,
                         self_seconds=own, cumulative_seconds=cumulative))
    result = dict(at=now(), scope='Static committed public-state construction with cProfile overhead; excludes environment transitions, IPC, policy and PPO',
        candidate_feature_backend='bitboard', candidate_search_backend=args.search_backend,
        repeats=args.repeats, states=len(cores),
        construction_calls=args.repeats * len(cores), profiled_wall_seconds=wall,
        profiled_process_cpu_seconds=cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        input_snapshot_sha256=snapshots, source_sha256=sources,
        runtime=evaluation_runtime_identity(diagnostics=False),
        fixtures=[dict(snapshot=name, worker=worker, legal_actions=output[0],
                       features_and_mask_sha256=output[1])
                  for (name, worker, _), output in zip(cores, before)],
        top_self_seconds=sorted(rows, key=lambda row: row['self_seconds'], reverse=True)[:25],
        top_cumulative_seconds=sorted(rows, key=lambda row: row['cumulative_seconds'], reverse=True)[:25],
        interpretation='Use to select a performance hypothesis; no adoption or end-to-end speedup established')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    operation('提交worker静态状态的'+args.search_backend+'落点/位图特征函数剖析完成，未训练或改变模型', args.output)
    print(json.dumps({key: result[key] for key in ['states', 'construction_calls', 'profiled_process_cpu_seconds']}))


if __name__ == '__main__':
    main()
