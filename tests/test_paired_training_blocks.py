from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import paired_branch_capture as branch
from audit_head_direction import validation_seed
from test_paired_branch_validation import validation_data


def data():
    preferences = validation_data()
    preferences['preferences'] += [dict(seed=seed, step=step, state_id=30+i*2+j, action=10+j,
        baseline_action=5, public_hash='public', snapshot_sha256='snapshot', label='beneficial', sources=['ppo_top_3'])
        for i, seed in enumerate(branch.TRAINING_EXTENSION_SEEDS) for j, step in enumerate([300, 100])]
    return preferences


def test_new_block_preserves_old_training_and_validation_choices():
    assert branch.selected_cases(data()) == branch.selected_cases(validation_data())
    assert branch.selected_cases(data(), 'validation') == branch.selected_cases(validation_data(), 'validation')
    cases = branch.selected_cases(data(), 'train', 1)
    assert [c['seed'] for c in cases] == branch.TRAINING_EXTENSION_SEEDS
    assert [c['step'] for c in cases] == [100]*4
    assert all(not validation_seed(c['seed']) for c in cases)
    assert not set(c['seed'] for c in cases) & set(branch.SEEDS+branch.VALIDATION_SEEDS)
    assert len({s['branch_id'] for s in branch.specs(cases)}) == 8


def test_missing_new_seed_cannot_use_existing_training_or_validation_instead():
    preferences = data()
    preferences['preferences'] = [p for p in preferences['preferences'] if p['seed'] != branch.TRAINING_EXTENSION_SEEDS[0]]
    with pytest.raises(ValueError, match='predetermined train'):
        branch.selected_cases(preferences, 'train', 1)


@pytest.mark.parametrize('split,block', [('train', True), ('train', 2), ('train', -1), ('validation', 1)])
def test_unregistered_block_or_validation_reassignment_is_refused(split, block):
    with pytest.raises(ValueError, match='training block'):
        branch.selected_cases(data(), split, block)
