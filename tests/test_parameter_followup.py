"""Test the complete filesystem/service handoff without launching GPU work."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_parameter_followup as followup
from evaluation_identity import RESUMABLE


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def setup(tmp_path,monkeypatch,eligible=True,status='active'):
    upstream=tmp_path/'screen';heldout=tmp_path/'extra';extension=heldout/'extension'
    save(upstream/'status.json',dict(stage='completed_screen'))
    save(upstream/'summary.json',dict(recommended_for_review='strong' if eligible else None,
         results=[dict(case='strong',eligible_for_longer_development=eligible)]))
    model=upstream/'strong/stage1_checkpoint/task_000064/model.zip'
    model.parent.mkdir(parents=True);model.write_bytes(b'candidate')
    baseline_model=tmp_path/'frozen.zip';baseline_model.write_bytes(b'baseline')
    def result(path,step):
        return dict(model_checkpoint=str(path),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    evaluation_source_sha256={'evaluate.py':'fixed'},committed_steps=step,
                    seeds=list(range(100000,100008)),num_seeds=8,max_pieces=20000,
                    per_seed=[dict(seed=n,pieces_survived=20000,score=1000,survived_cap=True,game_over=False)
                              for n in range(100000,100008)])
    save(tmp_path/'baseline.json',result(baseline_model,0))
    save(upstream/'strong/eval.json',result(model,262144))
    save(upstream/'strong/preset.json',{})
    save(upstream/'strong/training/config.json',{})
    save(upstream/'protocol.json',dict(baseline=str(tmp_path/'baseline.json'),
         evaluation_filename='eval.json',stage1_steps=262144,training_source_sha256={}))
    save(tmp_path/'training_resources.json',{})
    save(tmp_path/'evaluation_resources.json',{})
    save(tmp_path/'protocol.json',dict(thread_id='thread',wall_seconds=60,parameter_screen=str(upstream),
         upstream_service='screen.service',heldout=str(heldout),extension=str(extension),
         heldout_service='pair@extra.service',extension_service='extension@extra.service',
         extra_seeds=list(range(100008,100016)),frozen_files={},extension_gate={},
         training_resources=str(tmp_path/'training_resources.json'),
         evaluation_resources=str(tmp_path/'evaluation_resources.json'),short_baseline=str(tmp_path/'baseline.json')))
    calls=[]
    def service(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=3 if 'is-active' in args else 0)
    monkeypatch.setattr(followup,'inspect',lambda *args:(status,0))
    monkeypatch.setattr(followup,'operation',lambda *args:None)
    monkeypatch.setattr(followup.signal,'signal',lambda *args:None)
    monkeypatch.setattr(followup.subprocess,'run',service)
    return heldout,extension,calls


def test_eligible_candidate_handoff_uses_extra_seeds_and_persistent_guarded_services(tmp_path,monkeypatch):
    heldout,extension,calls=setup(tmp_path,monkeypatch)
    followup.run(tmp_path)
    protocol=json.loads((heldout/'protocol.json').read_text())
    assert protocol['seeds']==list(range(100008,100016)) and not protocol['diagnostic_required']
    assert not set(protocol['seeds']) & set(range(100000,100008))
    continuation=json.loads((extension/'protocol.json').read_text())
    assert continuation['training']==str(tmp_path/'screen/strong/training')
    assert continuation['checkpoints'][-1]==2097152
    assert ['systemctl','--user','enable','extension@extra.service'] in calls
    assert ['systemctl','--user','start','pair@extra.service'] in calls
    assert ['systemctl','--user','start','extension@extra.service'] in calls
    count=len(calls);followup.run(tmp_path)
    assert len(calls)==count  # A second invocation cannot launch duplicate jobs.


def test_no_eligible_candidate_does_not_spend_more_evaluation_or_training(tmp_path,monkeypatch):
    heldout,extension,calls=setup(tmp_path,monkeypatch,eligible=False)
    followup.run(tmp_path)
    assert not heldout.exists() and not extension.exists()
    assert not any('start' in call or 'enable' in call for call in calls)


def test_user_pause_stops_handoff_before_starting_any_service(tmp_path,monkeypatch):
    heldout,_,calls=setup(tmp_path,monkeypatch,status='paused')
    with pytest.raises(RuntimeError,match='Goal stopped'):
        followup.run(tmp_path)
    assert not heldout.exists() and not any('start' in call for call in calls)


def test_resumable_method_and_runtime_survive_validation_and_extension_handoff(tmp_path,monkeypatch):
    heldout,extension,_=setup(tmp_path,monkeypatch)
    path=tmp_path/'screen/protocol.json';source=json.loads(path.read_text());source['evaluation_module']=RESUMABLE
    path.write_text(json.dumps(source))
    followup.run(tmp_path)
    pair=json.loads((heldout/'protocol.json').read_text())
    continuation=json.loads((extension/'protocol.json').read_text())
    assert pair['evaluation_module']==continuation['evaluation_module']==RESUMABLE
    assert pair['runtime_versions'] and pair['python_version'] and pair['diagnostics'] is False


def test_validation_only_handoff_never_creates_or_starts_training_extension(tmp_path,monkeypatch):
    heldout,extension,calls=setup(tmp_path,monkeypatch)
    path=tmp_path/'protocol.json';protocol=json.loads(path.read_text());protocol['extension_enabled']=False
    for key in ('extension','extension_service','training_resources','short_baseline'):protocol.pop(key)
    path.write_text(json.dumps(protocol))
    followup.run(tmp_path)
    assert (heldout/'protocol.json').exists() and not extension.exists()
    assert ['systemctl','--user','enable','pair@extra.service'] in calls
    assert ['systemctl','--user','start','pair@extra.service'] in calls
    assert not any('extension@extra.service' in call for call in calls)
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['extension_enabled'] is False and result['extension'] is None


def test_live_bootstrap_is_waited_for_before_screen_starts(tmp_path,monkeypatch):
    _,_,calls=setup(tmp_path,monkeypatch,eligible=False)
    path=tmp_path/'protocol.json';protocol=json.loads(path.read_text());protocol['bootstrap_service']='bootstrap.service'
    path.write_text(json.dumps(protocol))
    summary_path=tmp_path/'screen/summary.json';summary=summary_path.read_text();summary_path.unlink()
    (tmp_path/'screen/status.json').unlink()
    original=followup.subprocess.run
    def service(args,**kwargs):
        if 'show' in args:
            calls.append(args)
            return SimpleNamespace(returncode=0,stdout='99' if 'bootstrap.service' in args else '0')
        return original(args,**kwargs)
    monkeypatch.setattr(followup.subprocess,'run',service)
    def finish(seconds):
        assert seconds==10
        summary_path.write_text(summary)
        save(tmp_path/'screen/status.json',dict(stage='completed_screen'))
    monkeypatch.setattr(followup.time,'sleep',finish)
    followup.run(tmp_path)
    assert any('bootstrap.service' in call and 'show' in call for call in calls)
    assert json.loads((tmp_path/'summary.json').read_text())['outcome']=='no_eligible_candidate'


def test_bootstrap_failure_cannot_be_misreported_as_pending(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch)
    path=tmp_path/'protocol.json';protocol=json.loads(path.read_text());protocol['bootstrap_service']='bootstrap.service'
    path.write_text(json.dumps(protocol));(tmp_path/'screen/summary.json').unlink()
    monkeypatch.setattr(followup.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout='0'))
    with pytest.raises(RuntimeError,match='stopped before'):followup.run(tmp_path)
