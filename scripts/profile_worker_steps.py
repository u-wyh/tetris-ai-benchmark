"""Measure worker step/reset CPU during one unchanged CUDA8 source rollout."""

import argparse
import copy
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train_vector_transaction import TransactionCompactSearchCandidateEnv


class ProfiledWorker(TransactionCompactSearchCandidateEnv):
    def start_profile(self):
        self._step_profiler = cProfile.Profile(timer=time.process_time)
        self._profile_counts = dict(step=0, reset=0)
        self._profile_cpu = 0.0

    def _record(self, name, method, *args, **kwargs):
        profiler = getattr(self, '_step_profiler', None)
        if profiler is None:
            return method(*args, **kwargs)
        began = time.process_time()
        profiler.enable()
        try:
            return method(*args, **kwargs)
        finally:
            profiler.disable()
            self._profile_cpu += time.process_time() - began
            self._profile_counts[name] += 1

    def step(self, action):
        return self._record('step', super().step, action)

    def reset(self, **kwargs):
        return self._record('reset', super().reset, **kwargs)

    def finish_profile(self):
        profiler = self._step_profiler
        self._step_profiler = None
        rows = []
        for (filename, line, name), (_, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
            rows.append(dict(file=filename, line=line, function=name, calls=calls,
                             self_cpu_seconds=own, cumulative_cpu_seconds=cumulative))
        return dict(pid=os.getpid(), counts=self._profile_counts, cpu_seconds=self._profile_cpu,
                    functions=rows)


def run(folder):
    import torch
    from stable_baselines3.common.logger import configure
    from training.train_vector_transaction import restore_task
    from training.vector_env import make_vector_env
    from profile_training_stages import rollout_digest
    from verify_feature_backend_benchmark import identical
    from resource_guard import now, write_json

    if (folder/'worker_profile.json').exists():
        raise FileExistsError('Preserve completed profile')
    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for name, sha in design['frozen_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                raise RuntimeError('Profile input changed: ' + name)
    unchanged()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    source = Path(design['source_context'])
    config = json.loads((source/'config.json').read_text())
    assert config['device'] == 'cuda' and config['ppo']['n_envs'] == 8
    env = make_vector_env(8, config['run_seed'], config['max_pieces'], env_class=ProfiledWorker)
    model = None
    try:
        model, _, _ = restore_task(source, design['source_task'], config, env)
        policy = copy.deepcopy(model.policy.state_dict())
        adam = copy.deepcopy(model.policy.optimizer.state_dict())
        updates, steps = model._n_updates, model.num_timesteps
        model.set_logger(configure(str(folder/'logs'), []))
        _, callback = model._setup_learn(4096, reset_num_timesteps=False, progress_bar=False)
        env.env_method('start_profile')
        callback.on_training_start({}, {})
        assert model.collect_rollouts(env, callback, model.rollout_buffer, model.n_steps, use_masking=True)
        callback.on_training_end()
        workers = env.env_method('finish_profile')
        assert len(workers) == 8 and all(w['counts']['step'] == 512 for w in workers)
        assert model.num_timesteps - steps == 4096
        digest = rollout_digest(model.rollout_buffer)
        expected = json.loads(Path(design['reference_rollout_audit']).read_text())['rollout_sha256']
        assert digest == expected, 'Worker-profiled rollout differs from historical audited rollout'
        assert identical(policy, model.policy.state_dict())
        assert identical(adam, model.policy.optimizer.state_dict()) and model._n_updates == updates
        unchanged()
        totals = {}
        for worker in workers:
            for row in worker['functions']:
                key = (row['file'], row['line'], row['function'])
                total = totals.setdefault(key, dict(row, calls=0, self_cpu_seconds=0.0, cumulative_cpu_seconds=0.0))
                for name in ('calls', 'self_cpu_seconds', 'cumulative_cpu_seconds'):
                    total[name] += row[name]
        rows = list(totals.values())
        write_json(folder/'worker_profile.json', dict(at=now(), status='passed', device='cuda', n_envs=8,
            samples=4096, source_optimizer_steps=0, source_policy_adam_ppo_updates_unchanged=True,
            training_search_backend='bitboard_compact', historical_rollout_sha256_exact=True,
            rollout_sha256=digest, workers=workers, aggregate_cpu_seconds=sum(w['cpu_seconds'] for w in workers),
            top_self=sorted(rows, key=lambda r: r['self_cpu_seconds'], reverse=True)[:40],
            top_cumulative=sorted(rows, key=lambda r: r['cumulative_cpu_seconds'], reverse=True)[:40],
            scope='Per-worker process CPU timer with cProfile overhead, step/reset only. '
                  'Excludes worker RPC/serialization/startup and parent/GPU work. '
                  'Cumulative rows overlap; sums across workers are not elapsed wall time. '
                  'No throughput or policy-quality conclusion.'))
    finally:
        if model is not None and hasattr(model, '_logger'):
            model.logger.close()
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
