"""Protect extension decisions against regressions hidden by capped averages."""

import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_development_evaluations import compare
from run_longer_development_screen import extension_eligible, global_control_allowed, run


GATE = dict(mean_pieces_ratio_at_least=0.99, mean_observed_score_ratio_at_least=0.99,
            no_seed_survival_ratio_below=0.9)


def test_global_stop_disable_or_wrong_machine_prevents_quality_work(tmp_path):
    path=tmp_path/'control.json'
    control=dict(enabled=True,thread_id='target',execution_host='remote-ssh-discovered:lightbook',
                 execution_machine_id=Path('/etc/machine-id').read_text().strip())
    protocol=dict(thread_id='target',control=str(path))
    path.write_text(json.dumps(control))
    assert global_control_allowed(protocol)
    (tmp_path/'STOP').touch()
    assert not global_control_allowed(protocol)
    (tmp_path/'STOP').unlink()
    control['enabled']=False;path.write_text(json.dumps(control))
    assert not global_control_allowed(protocol)
    control.update(enabled=True,execution_machine_id='wrong-machine');path.write_text(json.dumps(control))
    assert not global_control_allowed(protocol)


def fixture():
    return dict(model_sha256='baseline', evaluation_source_sha256={'evaluate.py': 'fixed'},
                seeds=list(range(8)), num_seeds=8, max_pieces=20000, committed_steps=0,
                per_seed=[dict(seed=n, pieces_survived=2000 if n == 7 else 20000,
                               score=2000 if n == 7 else 20000, survived_cap=n != 7)
                          for n in range(8)])


def test_single_large_loss_is_not_hidden_by_seven_capped_games():
    baseline = fixture()
    candidate = copy.deepcopy(baseline)
    candidate['per_seed'][7].update(pieces_survived=1700, score=1700)
    paired = compare(baseline, candidate)
    assert paired['mean_pieces_ratio'] > 0.99 and paired['mean_observed_score_ratio'] > 0.99
    assert not extension_eligible(paired, dict(baseline=1, candidate=1), GATE)


def test_more_game_overs_fail_even_when_means_are_similar():
    paired = compare(fixture(), fixture())
    assert not extension_eligible(paired, dict(baseline=1, candidate=2), GATE)


def test_matched_survival_with_large_score_loss_is_not_extended():
    baseline = fixture()
    candidate = copy.deepcopy(baseline)
    for row in candidate['per_seed']:
        row['score'] *= 0.98
    assert not extension_eligible(compare(baseline, candidate), dict(baseline=1, candidate=1), GATE)
    assert extension_eligible(compare(baseline, baseline), dict(baseline=1, candidate=1), GATE)


def test_boot_does_not_replay_a_completed_historical_screen(tmp_path):
    (tmp_path/'summary.json').write_text('{}')
    (tmp_path/'status.json').write_text(json.dumps(dict(stage='completed')))
    # No protocol or worker is needed when restarting an already finished job.
    run(tmp_path)
    assert not (tmp_path/'queue.lock').exists()
