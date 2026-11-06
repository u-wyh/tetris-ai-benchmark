"""Recoverable frozen-model sampling diagnostic; never trains or selects a model."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.evaluation.evaluate_resumable import collect_cached_rows, durable_json, validate_row


@contextmanager
def observe_distribution(policy):
    """Observe probabilities without consuming sampling RNG or changing results."""
    original = policy._distribution
    owned = '_distribution' in policy.__dict__
    previous = policy.__dict__.get('_distribution')
    last = dict(calls=0)
    def observed(*args, **kwargs):
        distribution = original(*args, **kwargs)
        probabilities = distribution.distribution.probs
        last.update(calls=last['calls']+1, greedy=int(probabilities.argmax(dim=-1).item()),
                    maximum_probability=float(probabilities.max().item()),
                    entropy=float(distribution.entropy().item()))
        return distribution
    policy._distribution = observed
    try:
        yield last
    finally:
        if owned:
            policy._distribution = previous
        else:
            del policy._distribution


def sampling_identity(model, seeds, action_seeds, cap, sources, runtime, script_sha):
    if (len(seeds) != len(action_seeds) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int for s in [*seeds, *action_seeds]) or type(cap) is not int or cap < 1):
        raise ValueError('Invalid stochastic sampling seeds or cap')
    return dict(model_checkpoint=str(Path(model).resolve()),
        model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        seeds=list(seeds), action_rng_seeds=list(action_seeds), max_pieces=cap,
        deterministic=False, action_rng='torch_cpu_manual_seed_per_episode',
        candidate_feature_backend='bitboard', evaluation_source_sha256=sources,
        diagnostic_source_sha256=script_sha, **runtime)


def validate_sampling_row(row, seed, action_seed, cap):
    validate_row(row, seed, cap)
    if row.get('action_rng_seed') != action_seed or row.get('deterministic') is not False:
        raise ValueError('Stochastic row sampling identity differs')
    if (type(row.get('prediction_count')) is not int
            or row['prediction_count'] != row['pieces_survived'] or row['prediction_count'] < 1):
        raise ValueError('Stochastic prediction count differs')
    for name, high in [('mean_entropy_nats', math.log(1840)),
                       ('mean_greedy_probability', 1), ('sampled_greedy_fraction', 1)]:
        value = row.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not -1e-6 <= value <= high+1e-6:
            raise ValueError('Invalid stochastic statistic: '+name)


def evaluate_seed(seed, action_seed, cap):
    import torch
    from training.evaluation import evaluate as evaluator
    from training.env.bitboard_candidate_env import BitboardCandidateTetrisEnv
    from verify_feature_backend_benchmark import identical

    model = evaluator._WORKER_MODEL
    policy_before = copy.deepcopy(model.policy.state_dict())
    optimizer_before = copy.deepcopy(model.policy.optimizer.state_dict())
    updates, timesteps = model._n_updates, model.num_timesteps
    env = BitboardCandidateTetrisEnv(max_pieces=cap)
    try:
        observation, _ = env.reset(seed=seed)
        count = greedy_count = 0
        entropy = maximum_probability = 0.0
        with torch.random.fork_rng(devices=[]), observe_distribution(model.policy) as last:
            torch.manual_seed(action_seed)
            while True:
                action, _ = model.predict(observation, deterministic=False, action_masks=env.action_masks())
                count += 1
                assert last['calls'] == count
                greedy_count += int(int(action) == last['greedy'])
                entropy += last['entropy']
                maximum_probability += last['maximum_probability']
                observation, _, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    break
        assert identical(policy_before, model.policy.state_dict())
        assert identical(optimizer_before, model.policy.optimizer.state_dict())
        assert (model._n_updates, model.num_timesteps) == (updates, timesteps)
        row = dict(seed=seed, action_rng_seed=action_seed, deterministic=False,
            pieces_survived=info['pieces'], lines=info['lines'], score=info['score'],
            episode_reward=env.episode_reward, game_over=bool(terminated), survived_cap=bool(truncated),
            prediction_count=count, mean_entropy_nats=entropy/count,
            mean_greedy_probability=maximum_probability/count, sampled_greedy_fraction=greedy_count/count,
            policy_optimizer_ppo_state_unchanged=True)
        validate_sampling_row(row, seed, action_seed, cap)
        return row
    finally:
        env.close()


def run(folder):
    from training.evaluation import evaluate as evaluator
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    from resource_guard import now

    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for name, sha in design['frozen_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                raise RuntimeError('Frozen stochastic input changed: '+name)
        if inference_source_identity(resumable=True) != design['evaluation_source_sha256']:
            raise RuntimeError('Inference source changed')
        if evaluation_runtime_identity(False) != design['evaluation_runtime']:
            raise RuntimeError('Evaluation runtime changed')
    unchanged()
    seeds, action_seeds, cap = design['seeds'], design['action_rng_seeds'], design['max_pieces']
    seed_map = dict(zip(seeds, action_seeds))
    results, greedy = {}, {}
    for label, specification in design['models'].items():
        unchanged()
        model = Path(specification['path'])
        identity = sampling_identity(model, seeds, action_seeds, cap,
            design['evaluation_source_sha256'], design['evaluation_runtime'],
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if identity['model_sha256'] != specification['sha256']:
            raise RuntimeError('Stochastic model identity differs')
        old = json.loads(Path(specification['greedy_result']).read_text())
        if (old['model_sha256'] != identity['model_sha256'] or old['seeds'] != seeds
                or old['evaluation_source_sha256'] != design['evaluation_source_sha256']
                or old['candidate_feature_backend'] != 'bitboard'
                or any(old.get(key) != design['evaluation_runtime'][key]
                       for key in ('runtime_versions', 'python_version'))):
            raise RuntimeError('Frozen greedy comparison identity differs')
        greedy[label] = dict(original_cap=old['max_pieces'],
            known_survival_per_seed=[r['pieces_survived'] for r in old['per_seed']],
            all_known_to_survive_short_cap=all(r['pieces_survived'] >= cap for r in old['per_seed']))

        def pending_rows(pending):
            with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn'),
                    initializer=evaluator._init_worker, initargs=(model, 'bitboard')) as pool:
                futures = [pool.submit(evaluate_seed, seed, seed_map[seed], cap) for seed in pending]
                for future in as_completed(futures):
                    row = future.result()
                    validate_sampling_row(row, row['seed'], seed_map[row['seed']], cap)
                    yield row
        rows, reused = collect_cached_rows(folder/label/'seeds', identity, pending_rows)
        for row in rows:
            validate_sampling_row(row, row['seed'], seed_map[row['seed']], cap)
        unchanged()
        result = dict(at=now(), **identity, per_seed=rows, aggregate=evaluator.aggregate(rows, cap),
            reused_seeds=reused, all_policy_optimizer_ppo_states_unchanged=all(
                row.get('policy_optimizer_ppo_state_unchanged') is True for row in rows))
        assert result['all_policy_optimizer_ppo_states_unchanged']
        total = sum(row['prediction_count'] for row in rows)
        result['sampling_statistics'] = {name: sum(row[name]*row['prediction_count'] for row in rows)/total
            for name in ('mean_entropy_nats', 'mean_greedy_probability', 'sampled_greedy_fraction')}
        results[label] = result
        durable_json(folder/(label+'.json'), result)
        durable_json(folder/'stochastic_progress.json', dict(at=now(), completed_models=list(results)))
    reference, candidate = results['reference']['aggregate'], results['candidate']['aggregate']
    ratio = candidate['mean_pieces']/reference['mean_pieces']
    all_capped = all(r['survived_cap'] for result in results.values() for r in result['per_seed'])
    descriptive_gain = (not all_capped and ratio >= 1.05
                        and candidate['game_over_rate'] <= reference['game_over_rate'])
    unchanged()
    durable_json(folder/'stochastic_summary.json', dict(at=now(), status='passed',
        seeds=seeds, action_rng_seeds=action_seeds, max_pieces=cap, deterministic=False,
        source_optimizer_steps=0, no_model_promotion=True, final_test_used=False,
        greedy_survival_lower_bounds=greedy, stochastic_survival_ratio=ratio,
        all_episodes_censored=all_capped, descriptive_stochastic_gain=descriptive_gain,
        stochastic_aggregate={name: result['aggregate'] for name, result in results.items()},
        sampling_statistics={name: result['sampling_statistics'] for name, result in results.items()},
        interpretation='Eight development seeds, one fixed action RNG per seed, capped sampling diagnostic. '
            'Greedy survival lower bounds are reused; scores across different caps are not compared. '
            'No causal objective-alignment claim, significance or model selection.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
