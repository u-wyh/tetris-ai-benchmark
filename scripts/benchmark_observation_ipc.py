"""Measure observation pipe overhead with replay data and no environment/policy work."""
import argparse
import hashlib
import json
import multiprocessing as mp
from multiprocessing.reduction import ForkingPickler
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def replay_worker(connection, frames):
    try:
        connection.send(('ready', os.getpid()))
        while True:
            mode, index = connection.recv()
            if mode == 'close':
                break
            observation, reward, done, info, reset_info = frames[index]
            if mode not in ('full', 'cached'):
                raise ValueError('Unknown replay protocol')
            connection.send((observation if mode == 'full' else index, reward, done, info, reset_info))
    finally:
        connection.close()


class ReplayPipes:
    def __init__(self, frames):
        self.frames, self.connections, self.processes = frames, [], []
        context = mp.get_context('forkserver')
        try:
            for rows in frames:
                parent, child = context.Pipe()
                process = context.Process(target=replay_worker, args=(child, rows), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                if not connection.poll(15) or connection.recv()[0] != 'ready':
                    raise RuntimeError('Replay worker startup failed')
        except BaseException:
            self.close()
            raise

    def request(self, mode, index):
        for connection in self.connections:
            connection.send((mode, index))
        replies = []
        for rank, connection in enumerate(self.connections):
            if not connection.poll(10):
                raise TimeoutError('Replay worker response timed out')
            reply = connection.recv()
            if mode == 'cached':
                if reply[0] != index:
                    raise RuntimeError('Cached observation index differs')
                reply = (self.frames[rank][index][0], *reply[1:])
            replies.append(reply)
        return replies

    def close(self):
        for connection in self.connections:
            try:
                connection.send(('close', 0))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()


def stack_observations(replies):
    from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs
    from gymnasium import spaces
    first = replies[0][0]
    observation_space = spaces.Dict({name: spaces.Box(low=0, high=255,
        shape=value.shape, dtype=value.dtype) for name, value in first.items()})
    return _stack_obs([r[0] for r in replies], observation_space)


def run(folder):
    from resource_guard import now, write_json
    from verify_feature_backend_benchmark import identical
    from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs
    from gymnasium import spaces
    if (folder/'ipc_summary.json').exists() or (folder/'pairs.jsonl').exists():
        raise FileExistsError('Preserve existing wire benchmark')
    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for name, digest in design['frozen_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Frozen IPC input changed: '+name)
    unchanged()
    source = Path(design['dataset_dir'])
    metadata = json.loads((source/'dataset.json').read_text())
    assert hashlib.sha256((source/'rollout.npz').read_bytes()).hexdigest() == metadata['file_sha256']
    with np.load(source/'rollout.npz', allow_pickle=False) as archive:
        selected = design['time_indices']
        observations = {name[4:]: archive[name][selected] for name in archive.files if name.startswith('obs_')}
        rewards, dones = archive['raw_env_rewards'][selected], archive['step_dones'][selected]
        infos = {name[5:]: archive[name][selected] for name in archive.files if name.startswith('info_')}
    assert all(value.shape[:2] == (16, 8) for value in observations.values())
    assert not dones.any(), 'Protocol excludes terminal/reset payloads; inspect fixed sample'
    frames = [[({name: values[index, worker].copy() for name, values in observations.items()},
        float(rewards[index, worker]), False,
        {name: values[index, worker].item() for name, values in infos.items()}, {})
        for index in range(16)] for worker in range(8)]
    space = spaces.Dict({name: spaces.Box(low=0, high=255, shape=value.shape, dtype=value.dtype)
                         for name, value in frames[0][0][0].items()})
    body_bytes = {mode: [len(ForkingPickler.dumps((frame[0] if mode == 'full' else index, *frame[1:])))
        for worker in frames for index, frame in enumerate(worker)] for mode in ('full', 'cached')}
    arrays_bytes = sum(value.nbytes for value in frames[0][0][0].values())
    pipes = ReplayPipes(frames)
    rows = []
    try:
        # Verify EVERY selected response before timing, with the same live workers.
        for index in range(16):
            full, cached = pipes.request('full', index), pipes.request('cached', index)
            assert identical(full, cached) and identical(full, [worker[index] for worker in frames])
            for name, values in _stack_obs([r[0] for r in full], space).items():
                np.testing.assert_array_equal(values, np.stack([worker[index][0][name] for worker in frames]))
        for pair in range(design['pairs']):
            order = ['full', 'cached'] if pair % 2 == 0 else ['cached', 'full']
            timings = {}
            for mode in order:
                began = time.perf_counter()
                for step in range(design['vector_steps']):
                    replies = pipes.request(mode, step % 16)
                    result = _stack_obs([r[0] for r in replies], space)
                timings[mode] = time.perf_counter()-began
                for name, values in result.items():
                    np.testing.assert_array_equal(values,
                        np.stack([worker[(design['vector_steps']-1) % 16][0][name] for worker in frames]))
            row = dict(pair=pair, order=order, seconds=timings,
                full_minus_cached_seconds=timings['full']-timings['cached'],
                wire_speedup=timings['full']/timings['cached'])
            rows.append(row)
            with (folder/'pairs.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
        unchanged()
        median_saved = statistics.median(row['full_minus_cached_seconds'] for row in rows)
        write_json(folder/'ipc_summary.json', dict(at=now(), status='passed', n_workers=8,
            worker_pids=[p.pid for p in pipes.processes], vector_steps_per_case=design['vector_steps'],
            pairs=rows, observation_arrays_bytes_per_worker=arrays_bytes,
            reply_pickle_body_bytes={mode: dict(min=min(sizes), mean=statistics.mean(sizes), max=max(sizes))
                                     for mode, sizes in body_bytes.items()},
            median_full_seconds=statistics.median(row['seconds']['full'] for row in rows),
            median_cached_seconds=statistics.median(row['seconds']['cached'] for row in rows),
            median_saved_seconds=median_saved,
            eligible_for_isolated_transport_prototype=(median_saved >= design['minimum_median_saved_seconds']
                and all(row['full_minus_cached_seconds'] > 0 for row in rows)),
            minimum_median_saved_seconds=design['minimum_median_saved_seconds'],
            all_selected_payloads_and_stacking_exact=True, env_steps=0, policy_optimizer_steps=0,
            scope='Eight live forkserver replay workers, pipe send/recv/pickle + SB3 stacking only. '
                  '16 uniformly spaced saved observations per worker, repeated for timing. Reply metadata '
                  'uses only saved info fields; excludes terminal/reset payloads, mask RPC, env/search/features, '
                  'policy/CUDA/PPO, process startup and per-response correctness hashing. Cached-index case '
                  'uses parent-resident replay arrays, an optimistic control, NOT shared-memory implementation. '
                  'Cannot infer causal share of real rollout wait or end-to-end training speedup.'))
    finally:
        pipes.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
