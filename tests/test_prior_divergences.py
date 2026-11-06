"""Temporal selection includes every predeclared recent point, excluding the cutoff."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from select_prior_divergences import prior_disagreements


def rows(seeds,steps,disagree):
    n=len(steps)
    return dict(episode_seed=np.array(seeds),pieces=np.array(steps),round=np.array(steps)-1,
        lane=np.array(seeds)-100000,action=np.ones(n,dtype=int),shadow_action=np.array(disagree,dtype=int)+1,
        candidate_features=np.zeros((n,16),np.uint8),shadow_features=np.ones((n,16),np.uint8),
        candidate_probability=np.full(n,.5),shadow_probability=np.full(n,.6),
        shadow_probability_of_executed=np.full(n,.4),observation_sha256=np.full(n,'o'*64),mask_sha256=np.full(n,'m'*64))


def test_recent_points_are_chronological_and_cutoff_other_seed_and_agreement_are_excluded():
    chunks=[(1,rows([100005]*5,[1,2,3,4,5],[1]*5)),
            (2,rows([100005,100000,100005,100005,100005],[6,7,7,8,9],[1,1,0,1,1]))]
    targets,available=prior_disagreements(chunks,100005,9,3,20)
    assert available==7 and [target['step'] for target in targets]==[5,6,8]
    assert [target['actions_to_observed_death_including_selected'] for target in targets]==[16,15,13]
    assert [target['source_chunk'] for target in targets]==[1,2,2]


def test_count_cannot_shrink_and_duplicate_or_unordered_inputs_are_rejected():
    with pytest.raises(ValueError,match='Not enough'):
        prior_disagreements([(1,rows([100005],[1],[1]))],100005,10,8,20)
    with pytest.raises(ValueError,match='chronological'):
        prior_disagreements([(1,rows([100005,100005],[2,1],[1,1]))],100005,10,2,20)
    with pytest.raises(ValueError,match='scope'):
        prior_disagreements([],100005,10,9,20)
