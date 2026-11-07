import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import paired_branch_capture as branch
from continue_paired_branch_job import phase_configuration
from test_paired_branch_capture import selection_data


def validation_data():
    data = selection_data()
    data['preferences'] += [dict(seed=seed, step=step, state_id=i*3+j, action=10+j,
        baseline_action=5, public_hash='public', snapshot_sha256='snapshot', label='beneficial', sources=['ppo_top_3'])
        for i, seed in enumerate(branch.VALIDATION_SEEDS) for j, step in enumerate((300, 100, 200))]
    return data


def test_old_training_selection_and_ids_are_unchanged():
    cases = branch.selected_cases(validation_data())
    assert [c['seed'] for c in cases] == branch.SEEDS
    assert [s['branch_id'] for s in branch.specs(cases)] == [f's{seed}_{role}' for seed in branch.SEEDS for role in ['base', 'alternative']]


def test_validation_is_episode_isolated_and_fixed_before_any_gpu_result():
    data = validation_data()
    cases = branch.selected_cases(data, 'validation')
    assert [c['seed'] for c in cases] == branch.VALIDATION_SEEDS+[992011]
    assert [c['step'] for c in cases] == [100, 100, 100, 200]
    assert not set(c['seed'] for c in cases) & set(branch.SEEDS)
    data['preferences'].reverse()
    assert branch.selected_cases(data, 'validation') == cases
    assert len({s['branch_id'] for s in branch.specs(cases)}) == 8


def test_same_state_different_interventions_still_have_independent_ids():
    case = branch.selected_cases(validation_data(), 'validation')[0]
    other = {**case, 'action': case['action']+1}
    assert len({s['branch_id'] for s in branch.specs([case, other])}) == 4
    with pytest.raises(ValueError, match='Duplicate'):
        branch.specs([case, copy.deepcopy(case)])


def test_missing_validation_episode_cannot_be_replaced_by_training():
    data = validation_data()
    data['preferences'] = [p for p in data['preferences'] if p['seed'] != 992027]
    with pytest.raises(ValueError, match='predetermined validation'):
        branch.selected_cases(data, 'validation')


def test_insufficient_fourth_case_cannot_reuse_an_existing_validation_case():
    data = validation_data()
    data['preferences'] = [p for p in data['preferences'] if p['step'] == 100]
    with pytest.raises(ValueError, match='fourth predetermined'):
        branch.selected_cases(data, 'validation')


def test_validation_phase_namespace_and_budget_do_not_reuse_original_units():
    phases, prefix, suffix = phase_configuration(dict(first_phase=1, last_phase=10, max_phases=10,
        unit_prefix='tetris-paired-validation', unit_suffix='20261107'))
    assert list(phases) == list(range(1, 11)) and prefix == 'tetris-paired-validation'
    assert suffix == '20261107'
    old, prefix, _ = phase_configuration({})
    assert list(old) == list(range(3, 12)) and prefix == 'tetris-paired-r2'


@pytest.mark.parametrize('fields', [dict(first_phase=True), dict(last_phase=99), dict(max_phases=100), dict(unit_prefix='../old')])
def test_unbounded_or_invalid_driver_configuration_is_rejected(fields):
    with pytest.raises(ValueError):
        phase_configuration(fields)
