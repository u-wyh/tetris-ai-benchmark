"""Fixed mixed-policy lanes and strict reproduction of captured source deaths."""
import copy

import numpy as np

from restore_rank_contexts import observation_digest
from verify_feature_backend_benchmark import identical


ARMS = ('A', 'B', 'C', 'D')
CONTINUATIONS = ('reference', 'reference', 'candidate', 'candidate')


class SourceReproductionError(RuntimeError):
    def __init__(self, event, work):
        super().__init__('Captured source D trajectory was not reproduced: '+str(event))
        self.event = event
        self.work = copy.deepcopy(work)


def run_factorial(env, specifications, models, horizon, source_traces=None):
    if not 1 <= len(specifications) <= 2 or env.num_envs != len(specifications)*4 or horizon < 1:
        raise ValueError('Invalid fixed mixed-policy factorial scope')
    views = []
    for lane in range(env.num_envs):
        specification = specifications[lane//4]
        view = env.env_method('branch_activate', specification['state'], indices=[lane])[0]
        if (not identical(env.env_method('get_state', indices=[lane])[0], specification['state']) or
                observation_digest(**view) != specification['digest']):
            raise ValueError('Factorial initial clone state, observation or mask differs')
        views.append(view)
    work = dict(environment_steps=0, policy_forward_calls=dict(reference=0, candidate=0), optimizer_steps=0)
    source_checked = [0]*len(specifications)
    def mismatch(case, time, phase, reason):
        raise SourceReproductionError(dict(case=case, key=specifications[case]['key'],
            local_action_step=time+1, phase=phase, reason=reason), work)
    for time in range(horizon):
        active = [lane for lane, view in enumerate(views) if view is not None]
        if not active:
            break
        actions = {}
        if time == 0:
            for lane in active:
                specification = specifications[lane//4]
                actions[lane] = specification['actions'][lane%2]
        else:
            for name in ('reference', 'candidate'):
                selected = [lane for lane in active if CONTINUATIONS[lane%4] == name]
                if not selected:
                    continue
                observations = {key: np.stack([views[lane]['observation'][key] for lane in selected])
                                for key in views[selected[0]]['observation']}
                masks = np.stack([views[lane]['mask'] for lane in selected])
                predicted, _ = models[name].predict(observations, deterministic=True, action_masks=masks)
                actions.update(zip(selected, map(int, predicted)))
                work['policy_forward_calls'][name] += 1
        # Verify every D before advancing any lane in this decision round.
        if source_traces is not None:
            for case, source in enumerate(source_traces):
                lane = case*4+3
                if lane not in active:
                    continue
                if time >= len(source):
                    mismatch(case, time, 'pre_step', 'Source is already terminal but D is active')
                expected = source[time]
                if (actions[lane] != expected['action'] or
                        observation_digest(**views[lane]) != expected['digest'] or
                        not np.array_equal(views[lane]['observation']['candidate'][actions[lane]], expected['features'])):
                    mismatch(case, time, 'pre_step', 'Action, observation, mask or selected feature differs')
        for lane in active:
            action = actions[lane]
            if not views[lane]['mask'][action]:
                raise ValueError('Factorial selected an illegal action')
            outcome = env.env_method('branch_step_active', action, indices=[lane])[0]
            if outcome is None:
                raise RuntimeError('Active factorial lane did not advance')
            work['environment_steps'] += 1
            if time == 0:
                expected = specifications[lane//4].get('expected_first_clears')
                if expected is not None and outcome['row']['cleared_lines'] != expected[lane%2]:
                    raise ValueError('Forced first action cleared an unexpected number of rows')
            if source_traces is not None and lane%4 == 3:
                case = lane//4
                expected = source_traces[case][time]['outcome']
                if any(value != expected[key] for key, value in outcome['row'].items()):
                    mismatch(case, time, 'post_step', 'Reward, endpoint, pieces, score or geometry information differs')
                source_checked[case] += 1
            views[lane] = outcome['view']
    outcomes = []
    for case in range(len(specifications)):
        result = {arm: env.env_method('branch_finish', horizon, indices=[case*4+index])[0]
                  for index, arm in enumerate(ARMS)}
        if source_traces is not None:
            if source_checked[case] != len(source_traces[case]) or result['D']['summary']['steps'] != len(source_traces[case]):
                mismatch(case, result['D']['summary']['steps'], 'end', 'D did not reproduce the entire original source endpoint')
        outcomes.append(result)
    return outcomes, dict(work, source_D_checked_steps=source_checked,
        source_D_reproduction_passed=source_traces is not None)
