"""Read-only CUDA8 restoration audit of the original full PPO30M transaction."""

import argparse
import fcntl
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(output):
    import torch
    import gymnasium
    import stable_baselines3
    import sb3_contrib
    from training.train_ppo import atomic_json, now
    from training.train_vector_transaction import (verify_task, restore_task, sha256,
        TransactionCandidateEnv, training_source_identity, observation_and_mask_digests)
    from training.train_transaction import global_rng_state
    from training.vector_env import make_vector_env
    from verify_feature_backend_benchmark import identical

    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'audit.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = ROOT/'runs/ppo_candidate_raw_30m_seed42'
    config = json.loads((source/'config.json').read_text())
    state = json.loads((source/'transaction_state.json').read_text())
    meta = json.loads((source/'metadata.json').read_text())
    number = state['committed_task']
    task = source/'committed'/f'task_{number:06d}'
    before = {str(p.relative_to(source)): sha256(p) for p in [
        source/'config.json', source/'transaction_state.json', source/'metadata.json',
        source/'final/model.zip', *[p for p in task.rglob('*') if p.is_file()]]}
    result = dict(at=now(), status='running', source=str(source), task=number,
                  committed_steps=state['committed_steps'], device='cuda', n_envs=8,
                  original_git_commit=meta.get('git_commit'),
                  original_source_hash_manifest_present='training_source_sha256' in meta,
                  audit_source_sha256=training_source_identity(), training_steps_added=0,
                  model_promotion='none')
    env = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU fallback')
        assert config['device'] == 'cuda' and config['ppo']['n_envs'] == 8
        assert not state.get('pid') and state['status'] == 'completed'
        versions = dict(torch=torch.__version__, gymnasium=gymnasium.__version__,
                        stable_baselines3=stable_baselines3.__version__, sb3_contrib=sb3_contrib.__version__)
        assert all(meta.get(k) == v for k, v in versions.items()), 'Recorded runtime versions differ'
        task_meta = verify_task(task, number, config)
        assert sha256(task/'model.zip') == sha256(source/'final/model.zip')
        torch.set_num_threads(1)
        env = make_vector_env(8, config['run_seed'], config['max_pieces'],
                              env_class=TransactionCandidateEnv)
        model, trainer, workers = restore_task(source, number, config, env)
        restored_rng = global_rng_state(model)
        assert identical(trainer, restored_rng), 'Full trainer/RNG state differs after restoration'
        result.update(trainer_and_rng_exact=True, worker_digests_exact=True)
        restored_workers = env.env_method('get_state')
        # Legacy snapshots predate these explicit default fields. Record every
        # schema addition; never ignore a differing saved value or trajectory key.
        legacy_defaults = dict(reward_version='placement-reward-v1', reward_profile='default',
                               hole_penalty_coef=0.0, height_penalty_coef=0.0)
        additions = []
        normalized = []
        for saved, restored in zip(workers, restored_workers):
            extra = set(restored) - set(saved)
            assert extra <= legacy_defaults.keys(), 'Unknown legacy worker schema addition'
            additions.append(sorted(extra))
            normalized.append(dict(saved, **{k: legacy_defaults[k] for k in extra}))
        result['explicit_legacy_default_fields_per_worker'] = additions
        assert identical(normalized, restored_workers), 'Full worker state differs after explicit legacy defaults'
        _, digests = observation_and_mask_digests(env)
        assert digests == task_meta['worker_digests']
        assert before == {name: sha256(source/name) for name in before}, 'Original artifact changed'
        result.update(status='passed', at=now(), runtime_versions=versions,
                      worker_count=len(workers), worker_digests_exact=True,
                      full_worker_states_exact_after_explicit_legacy_defaults=True, trainer_and_rng_exact=True,
                      model_matches_frozen_final=True, original_files_unchanged=True,
                      cuda_device=str(next(model.policy.parameters()).device),
                      optimizer_state_count=len(model.policy.optimizer.state),
                      interpretation='Current-code restoration verified; old source identity remains unproven. Independent continuation design still required.')
    except BaseException as exc:
        result.update(status='failed', at=now(), error=repr(exc))
        raise
    finally:
        if env is not None:
            env.close()
        atomic_json(output/'summary.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
