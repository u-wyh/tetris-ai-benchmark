"""Rejected performance experiments must not start dependent GPU validation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import verify_context_guards_when_idle as staged


@pytest.mark.parametrize('value', [True, False])
def test_explicit_passed_eligibility(tmp_path, value):
    path = tmp_path/'upstream.json'
    path.write_text(json.dumps(dict(status='passed', eligible_for_new_experiments=value)))
    assert staged.upstream_eligible(path) is value


@pytest.mark.parametrize('receipt', [dict(status='failed', eligible_for_new_experiments=True),
    dict(status='passed'), dict(status='passed', eligible_for_new_experiments=1),
    dict(status='passed', eligible_for_new_experiments='true')])
def test_missing_or_ambiguous_receipt_fails_closed(tmp_path, receipt):
    path = tmp_path/'upstream.json'
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError):
        staged.upstream_eligible(path)


def test_rejected_upstream_skips_without_creating_guard_or_process(tmp_path, monkeypatch):
    receipt = tmp_path/'gpu_summary.json'
    receipt.write_text(json.dumps(dict(status='passed', eligible_for_new_experiments=False)))
    design = dict(thread_id='test', wall_seconds=60, frozen_files={}, upstream_units=[],
                  required_upstream_receipt=str(receipt))
    (tmp_path/'design.json').write_text(json.dumps(design))
    monkeypatch.setattr(staged, 'allowed', lambda *args: True)
    monkeypatch.setattr(staged.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(staged.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=''))
    def forbidden(*args, **kwargs):
        pytest.fail('Rejected upstream started a process')
    monkeypatch.setattr(staged.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(staged, 'operation', lambda *args: None)
    staged.run(tmp_path)
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['status'] == 'skipped_ineligible_upstream'
    assert summary['validation_started'] is False
    assert not (tmp_path/'guard').exists()
