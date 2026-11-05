import importlib.util
from pathlib import Path
import sqlite3
import json
from types import SimpleNamespace


spec=importlib.util.spec_from_file_location('hourly_continue',Path(__file__).resolve().parents[1]/'scripts/hourly_continue.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_paused_or_completed_goal_is_never_reawakened():
    for status in ('paused','complete','completed','blocked','budgetLimited','budget_limited'):
        assert module.decision(True,status,0)!='queue'
    assert module.decision(False,'active',0)=='disabled'
    assert module.decision(True,None,0)=='no_active_goal'


def test_pending_user_or_retry_message_prevents_duplicates():
    assert module.decision(True,'active',1)=='already_queued'
    assert module.decision(True,'active',0)=='queue'
    assert module.decision(True,'usage_limited',0)=='queue'
    assert module.decision(True,'usageLimited',0)=='queue'
    assert module.decision(True,'unknown_future_status',0)=='unsupported_goal_status'


def test_read_only_status_inspection_is_scoped_to_target_thread(tmp_path):
    with sqlite3.connect(tmp_path/'goals_1.sqlite') as db:
        db.execute('CREATE TABLE thread_goals(thread_id TEXT,status TEXT)')
        db.executemany('INSERT INTO thread_goals VALUES (?,?)',[('target','active'),('other','paused')])
    with sqlite3.connect(tmp_path/'queue_1.sqlite') as db:
        db.execute('CREATE TABLE queued_items(thread_id TEXT)')
        db.execute("INSERT INTO queued_items VALUES ('other')")
    assert module.inspect(tmp_path,'target')==('active',0)
    assert module.inspect(tmp_path,'other')==('paused',1)


def control_file(tmp_path):
    path=tmp_path/'control.json'
    path.write_text(json.dumps(dict(enabled=True,thread_id='target',codex_cli='fake-cli',
                                   interval_seconds=3600,created_at='1970-01-01T00:00:00+00:00',prompt='continue')))
    return path


def test_fast_manual_restart_cannot_queue_twice_within_hour(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'inspect',lambda *args:('active',0))
    calls=[]
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kw:(calls.append(args) or SimpleNamespace(returncode=0,stdout='queued')))
    path=control_file(tmp_path)
    assert module.run(path,clock=lambda:4000)==0
    assert module.run(path,clock=lambda:4001)==0
    assert len(calls)==1
    assert module.run(path,clock=lambda:7600)==0
    assert len(calls)==2


def test_failed_delivery_also_waits_before_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'inspect',lambda *args:('usageLimited',0))
    calls=[]
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kw:(calls.append(args) or SimpleNamespace(returncode=1,stdout='')))
    path=control_file(tmp_path)
    assert module.run(path,clock=lambda:4000)==1
    assert module.run(path,clock=lambda:4001)==0
    assert len(calls)==1
