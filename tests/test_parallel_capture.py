import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import own_policy_capture as serial
import parallel_capture as parallel
from verify_feature_backend_benchmark import identical
from training.vector_env import make_vector_env
from training.tetris_core.pieces import make_piece


@pytest.mark.parametrize('seeds', [[1, 1], [True, 2], [-1, 2], [2**32, 2], [1]])
def test_invalid_seed_routes_rejected_before_inference_or_dispatch(seeds):
    env = SimpleNamespace(num_envs=2)
    with pytest.raises(ValueError, match='distinct uint32'):
        parallel.advance_parallel(env, [None, None], None, None, 0, seeds)


@pytest.mark.parametrize('mapping', [{}, {100000: True}, {100000: 1840}])
def test_bad_mapping_does_not_advance_worker(mapping):
    env = parallel.ParallelCompactEnv(max_pieces=3)
    try:
        env.capture_reset(100000)
        before = env.capture_export()
        with pytest.raises(ValueError, match='dispatch map'):
            env.capture_step_by_seed(mapping)
        assert identical(before, env.capture_export())
    finally:
        env.close()


def test_true_death_stays_inactive_after_seed_dispatch():
    env = parallel.ParallelCompactEnv(max_pieces=20)
    try:
        env.capture_reset(100000)
        env.core.current = make_piece('I'); env.core.current['y'] = -2
        for x in range(3, 7):
            env.core.board[0][x] = 'Z'
        env._invalidate_candidates()
        placement = next(p for p in env.core.get_legal_placements()
                         if all(c['y'] < 0 for c in p['occupiedCells']))
        result = env.capture_step_by_seed({100000: placement['actionId']})
        assert result['row']['terminated'] and not result['row']['truncated']
        saved = env.capture_export()
        assert env.capture_step_by_seed({}) is None
        assert identical(saved, env.capture_export())
    finally:
        env.close()


def test_real_concurrent_workers_match_serial_and_resume_stopped_slots(monkeypatch):
    def first(policy, observations, masks, executed=None):
        actions = masks.argmax(axis=1)
        return dict(action=actions, probability=np.ones(len(actions)), margin=np.ones(len(actions)),
                    probability_of_executed=np.ones(len(actions)))
    monkeypatch.setattr(serial, 'greedy_details', first)
    monkeypatch.setattr(parallel, 'greedy_details', first)
    model = SimpleNamespace(policy=None)
    seeds = [100000, 100001]
    def vector():
        return make_vector_env(2, base_seed=100000, max_pieces=3, env_class=parallel.ParallelCompactEnv)
    env = vector()
    try:
        views = [env.env_method('capture_reset', seed, indices=[lane])[0] for lane, seed in enumerate(seeds)]
        # Stop the second lane by real cap transitions, not a fabricated finished flag.
        for _ in range(3):
            action = int(np.flatnonzero(views[1]['mask'])[0])
            views[1] = env.env_method('capture_step', action, indices=[1])[0]['view']
        initial = env.env_method('capture_export')
        expected = [serial.advance_round(env, views, model, model, r) for r in range(3)]
        end = env.env_method('capture_export')
        views = [env.env_method('capture_restore', state, indices=[lane])[0]
                 for lane, state in enumerate(initial)]
        calls = []
        original_method = env.env_method
        def observe(method, *args, **kwargs):
            if method == 'capture_step_by_seed':
                calls.append(kwargs['indices'])
            return original_method(method, *args, **kwargs)
        monkeypatch.setattr(env, 'env_method', observe)
        first_rows = parallel.advance_parallel(env, views, model, model, 0, seeds)
        saved = env.env_method('capture_export')
        assert calls == [[0]] and identical(first_rows, expected[0])
    finally:
        env.close()
    env = vector()  # New worker processes, same full state, no reset.
    try:
        views = [env.env_method('capture_restore', state, indices=[lane])[0]
                 for lane, state in enumerate(saved)]
        assert identical(saved, env.env_method('capture_export'))
        observed = [parallel.advance_parallel(env, views, model, model, r, seeds) for r in [1, 2]]
        assert identical(observed, expected[1:]) and identical(end, env.env_method('capture_export'))
        assert views == [None, None]
        assert parallel.advance_parallel(env, views, model, model, 3, seeds) == []
    finally:
        env.close()
