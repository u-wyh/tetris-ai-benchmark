"""Mixed continuation routing, stopped lanes and fail-closed source reproduction."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from own_death_factorial_helpers import run_factorial, SourceReproductionError, ARMS
from rank_branch_helpers import BranchCompactEnv
from restore_rank_contexts import observation_digest
from run_own_death_factorial import comparisons, save_group, read_group


class LocalVector:
    def __init__(self, width, cap):
        self.num_envs = width
        self.envs = [BranchCompactEnv(max_pieces=cap) for _ in range(width)]
    def env_method(self, name, *args, indices=None):
        return [getattr(self.envs[lane], name)(*args) for lane in
                (range(self.num_envs) if indices is None else indices)]
    def close(self):
        for env in self.envs:
            env.close()


class Recorder:
    def __init__(self, last):
        self.last = last; self.batch_sizes = []
    def predict(self, observations, deterministic, action_masks):
        assert deterministic is True and len(observations['state']) == len(action_masks)
        self.batch_sizes.append(len(action_masks))
        actions = (action_masks.shape[1]-1-action_masks[:,::-1].argmax(axis=1)
                   if self.last else action_masks.argmax(axis=1))
        return actions, None


def specification(cap=3, prefix=0, seed=42):
    env = BranchCompactEnv(max_pieces=cap)
    try:
        env.reset(seed=seed)
        for _ in range(prefix):
            env.step(int(env.action_masks().argmax()))
        observation = env.get_observation(); mask = env.action_masks()
        return dict(key=f'context_{seed}', state=env.get_state(), digest=observation_digest(observation,mask),
                    actions=[int(np.flatnonzero(mask)[0]),int(np.flatnonzero(mask)[-1])])
    finally:
        env.close()


def source_trace(spec, cap=3):
    env = BranchCompactEnv(max_pieces=cap)
    rows = []
    try:
        view = env.branch_activate(spec['state'])
        while view is not None:
            action = spec['actions'][1] if not rows else int(np.flatnonzero(view['mask'])[-1])
            expected = dict(action=action, digest=observation_digest(**view),
                            features=view['observation']['candidate'][action].copy())
            result = env.branch_step_active(action)
            expected['outcome'] = result['row']; rows.append(expected); view = result['view']
        return rows
    finally:
        env.close()


def test_mixed_models_force_first_then_route_only_ordered_active_lanes_without_reset():
    env = LocalVector(8,3)
    models = dict(reference=Recorder(False), candidate=Recorder(True))
    specs = [specification(prefix=1),specification(seed=43)]
    try:
        outcomes, work = run_factorial(env, specs, models, 5)
        assert models['reference'].batch_sizes == [4,2]
        assert models['candidate'].batch_sizes == [4,2]
        assert work['environment_steps'] == 20 and work['policy_forward_calls'] == dict(reference=2,candidate=2)
        for case, outcome in enumerate(outcomes):
            for index, arm in enumerate(ARMS):
                assert outcome[arm]['trace'][0]['action'] == specs[case]['actions'][index%2]
                assert outcome[arm]['summary']['steps'] == (2 if case == 0 else 3)
                assert outcome[arm]['summary']['time_limit'] and not outcome[arm]['summary']['horizon_censored']
                assert outcome[arm]['state']['episode_count'] == 1
            # B and D have identical first action/state; their next action uses different models.
            assert outcome['B']['trace'][1]['action'] != outcome['D']['trace'][1]['action']
        assert all(env.env_method('branch_step_active',None)[lane] is None for lane in range(8))
    finally:
        env.close()


def test_source_D_reproduction_checks_complete_path_and_group_commit_is_immutable(tmp_path):
    env = LocalVector(4,3); spec = specification(); source = source_trace(spec)
    try:
        outcomes, work = run_factorial(env,[spec],dict(reference=Recorder(False),candidate=Recorder(True)),5,[source])
        assert work['source_D_reproduction_passed'] and work['source_D_checked_steps'] == [3]
        assert outcomes[0]['D']['trace'] == [row['outcome'] for row in source]
        assert comparisons(outcomes[0])['first_action_with_candidate']['common_steps'] == 3
        identity = dict(group='fixed-all-lanes', horizon=5)
        result = dict(work=work, summaries={arm:outcomes[0][arm]['summary'] for arm in ARMS})
        states = {arm:outcomes[0][arm]['state'] for arm in ARMS}
        directory = tmp_path/'group'
        save_group(directory,identity,result,states)
        before = {p.name:p.read_bytes() for p in directory.iterdir()}
        save_group(directory,identity,result,states)
        assert before == {p.name:p.read_bytes() for p in directory.iterdir()}
        with pytest.raises(ValueError, match='identity'):
            read_group(directory,dict(group='other'))
        with pytest.raises(ValueError, match='differs'):
            save_group(directory,identity,dict(result,work={}),states)
    finally:
        env.close()


@pytest.mark.parametrize('corruption',['action','digest','reward'])
def test_source_mismatch_preserves_failure_phase_and_actual_work(corruption):
    env = LocalVector(4,3); spec = specification(); source = copy.deepcopy(source_trace(spec))
    if corruption == 'action':
        source[0]['action'] = -1
    elif corruption == 'digest':
        source[0]['digest']['mask'] = 'bad'
    else:
        source[0]['outcome']['reward'] += 1
    try:
        with pytest.raises(SourceReproductionError) as captured:
            run_factorial(env,[spec],dict(reference=Recorder(False),candidate=Recorder(True)),5,[source])
        error = captured.value
        assert error.event['local_action_step'] == 1 and error.event['case'] == 0
        assert error.work['environment_steps'] == (4 if corruption == 'reward' else 0)
        assert error.event['phase'] == ('post_step' if corruption == 'reward' else 'pre_step')
    finally:
        env.close()
