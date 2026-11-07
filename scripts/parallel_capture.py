"""Ordered concurrent worker dispatch for single-episode, unique-seed captures."""
import numpy as np

from own_policy_capture import CaptureNativeEnv, CaptureCompactEnv, greedy_details
from restore_rank_contexts import observation_digest


def validate_seeds(seeds, count):
    if (len(seeds) != count or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Capture dispatch requires one distinct uint32 seed per lane')


class SeedDispatch:
    def capture_step_by_seed(self, actions):
        if not getattr(self, '_capture_active', False):
            return None
        if (type(actions) is not dict or self._capture_seed not in actions or
                any(type(seed) is not int or type(action) is not int or not 0 <= action < 1840
                    for seed, action in actions.items())):
            raise ValueError('Invalid seed-to-action dispatch map')
        return self.capture_step(actions[self._capture_seed])


class ParallelNativeEnv(SeedDispatch, CaptureNativeEnv):
    pass


class ParallelCompactEnv(SeedDispatch, CaptureCompactEnv):
    pass


def advance_parallel(env, views, candidate, shadow, round_index, seeds):
    """Caller must verify worker episode seeds against this fixed order on restoration."""
    validate_seeds(seeds, env.num_envs)
    if len(views) != env.num_envs:
        raise ValueError('Missing lane views')
    active = [lane for lane, view in enumerate(views) if view is not None]
    if not active:
        return []
    observations = {key: np.stack([views[lane]['observation'][key] for lane in active])
                    for key in views[active[0]]['observation']}
    masks = np.stack([views[lane]['mask'] for lane in active])
    chosen = greedy_details(candidate.policy, observations, masks)
    observed = greedy_details(shadow.policy, observations, masks, chosen['action'])
    rows = []
    actions = {}
    for batch, lane in enumerate(active):
        action, other = int(chosen['action'][batch]), int(observed['action'][batch])
        digest = observation_digest(**views[lane])
        rows.append(dict(round=round_index, lane=lane, action=action, shadow_action=other,
            observation_sha256=digest['observation'], mask_sha256=digest['mask'],
            candidate_features=observations['candidate'][batch, action].copy(),
            shadow_features=observations['candidate'][batch, other].copy(),
            candidate_probability=float(chosen['probability'][batch]),
            candidate_margin=float(chosen['margin'][batch]),
            shadow_probability=float(observed['probability'][batch]),
            shadow_margin=float(observed['margin'][batch]),
            shadow_probability_of_executed=float(observed['probability_of_executed'][batch])))
        actions[seeds[lane]] = action
    # SubprocVecEnv sends to every target before receiving the first result.
    outcomes = env.env_method('capture_step_by_seed', actions, indices=active)
    if len(outcomes) != len(active):
        raise RuntimeError('Missing concurrent worker responses')
    for lane, row, outcome in zip(active, rows, outcomes):
        if outcome is None or outcome['row']['episode_seed'] != seeds[lane]:
            raise RuntimeError('Worker dispatch seed or active state differs')
        row.update(outcome['row'])
        views[lane] = outcome['view']
    return rows
