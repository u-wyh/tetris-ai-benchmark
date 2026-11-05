"""Unattended gates must stop safely without an operator or another AI call."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_unattended_extension as extension


def result():
    return dict(model_sha256='frozen', evaluation_source_sha256={'evaluate.py': 'fixed'},
                seeds=[1], num_seeds=1, max_pieces=5000, committed_steps=524288,
                per_seed=[dict(seed=1, pieces_survived=5000, score=1000,
                               survived_cap=True, game_over=False)])


def test_short_gate_requires_survival_score_and_matching_evaluation():
    baseline = result()
    assert extension.short_gate(baseline, baseline)
    for change in [dict(score=980), dict(survived_cap=False), dict(game_over=True)]:
        candidate = copy.deepcopy(baseline)
        candidate['per_seed'][0].update(change)
        assert not extension.short_gate(baseline, candidate)
    candidate = copy.deepcopy(baseline)
    candidate['max_pieces'] = 20000
    assert not extension.short_gate(baseline, candidate)


def test_restarting_a_terminal_development_run_does_not_retrain(tmp_path):
    for outcome in ['not_eligible', 'regression_stop', 'completed_budget']:
        (tmp_path/'summary.json').write_text(json.dumps(dict(outcome=outcome)))
        # There is deliberately no protocol, live upstream, or trainer here.
        extension.run(tmp_path)
        assert not (tmp_path/'queue.lock').exists()


@pytest.mark.parametrize('scenario', ['not_eligible', 'dead', 'inconsistent'])
def test_failed_gate_or_missing_live_controller_never_starts_training(tmp_path, monkeypatch, scenario):
    upstream = tmp_path/'upstream'
    upstream.mkdir()
    if scenario != 'dead':
        (upstream/'summary.json').write_text(json.dumps(dict(eligible_for_training_extension=False)))
    (upstream/'status.json').write_text(json.dumps(dict(stage='completed' if scenario == 'not_eligible' else 'baseline')))
    protocol = dict(wall_seconds=60, thread_id='test', longer_screen=str(upstream),
                    upstream_service='test.service', frozen_files={})
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    monkeypatch.setattr(extension.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(extension, 'inspect_goal', lambda *args: ('usageLimited', 0))
    monkeypatch.setattr(extension, 'operation', lambda *args: None)
    monkeypatch.setattr(extension.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=3, stdout='0'))
    def unexpected_child(*args, **kwargs):
        pytest.fail('Failed gate or dead upstream must not launch work')
    monkeypatch.setattr(extension.subprocess, 'Popen', unexpected_child)
    if scenario == 'not_eligible':
        extension.run(tmp_path)
        summary = json.loads((tmp_path/'summary.json').read_text())
        assert summary['outcome'] == 'not_eligible' and summary['model_promotion'] == 'none'
    else:
        with pytest.raises(RuntimeError, match='Upstream'):
            extension.run(tmp_path)
        assert json.loads((tmp_path/'status.json').read_text())['stage'] == 'stopped'
