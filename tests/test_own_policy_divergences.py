"""Select ALL true deaths and preserve the action versus pre-state offset."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from locate_own_policy_divergences import latest_disagreements


def rows(seeds, steps, actions, shadow):
    size = len(seeds)
    return dict(episode_seed=np.array(seeds), pieces=np.array(steps), round=np.array(steps)-1,
        lane=np.array(seeds)-100000, action=np.array(actions), shadow_action=np.array(shadow),
        candidate_features=np.tile(np.arange(16, dtype=np.uint8), (size, 1)),
        shadow_features=np.tile(np.arange(16, dtype=np.uint8)+1, (size, 1)),
        candidate_probability=np.full(size, .6), shadow_probability=np.full(size, .7),
        shadow_probability_of_executed=np.full(size, .3),
        observation_sha256=np.full(size, 'o'*64), mask_sha256=np.full(size, 'm'*64))


def test_last_disagreement_includes_all_deaths_and_not_cap_lane():
    chunks = [(1, rows([100000,100002,100005], [3,5,8], [1,1,1], [2,2,2])),
              (2, rows([100002,100005,100002], [260,261,280], [3,4,5], [9,8,5]))]
    targets = latest_disagreements(chunks, {100002:300, 100005:270})
    assert [t['seed'] for t in targets] == [100002,100005]
    assert [t['step'] for t in targets] == [260,261]
    assert [t['round'] for t in targets] == [259,260]
    assert [t['actions_to_observed_death_including_selected'] for t in targets] == [41,10]
    assert all(t['source_chunk'] == 2 and t['previous_chunk'] == 1 for t in targets)
    assert targets[1]['candidate_action'] == 4 and targets[1]['shadow_action'] == 8


def test_missing_death_cannot_be_dropped_and_fatal_disagreement_is_allowed():
    data = rows([100002], [300], [1], [2])
    target = latest_disagreements([(2,data)], {100002:300})[0]
    assert target['actions_to_observed_death_including_selected'] == 1
    with pytest.raises(ValueError, match='no preceding'):
        latest_disagreements([(2,data)], {100002:300,100005:700})
    with pytest.raises(ValueError, match='outside'):
        latest_disagreements([(2,data)], {100002:299})


def test_disagreement_input_order_and_step_offset_are_checked():
    data = rows([100002,100002], [8,7], [1,1], [2,2])
    with pytest.raises(ValueError, match='increasing'):
        latest_disagreements([(1,data)], {100002:20})
    data = rows([100002], [8], [1], [2]);data['round'][0] = 8
    with pytest.raises(ValueError, match='outside'):
        latest_disagreements([(1,data)], {100002:20})
