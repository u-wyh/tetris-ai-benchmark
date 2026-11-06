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
    monkeypatch.setattr(module,'recover_goal',lambda thread:dict(ready=True,action='goal_changed'))
    calls=[]
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kw:(calls.append(args) or SimpleNamespace(returncode=1,stdout='')))
    path=control_file(tmp_path)
    assert module.run(path,clock=lambda:4000)==1
    assert module.run(path,clock=lambda:4001)==0
    assert len(calls)==1


def test_live_quota_unavailable_waits_an_hour_without_delivering(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'inspect',lambda *args:('usageLimited',0))
    reads=[]
    monkeypatch.setattr(module,'recover_goal',lambda thread:(reads.append(thread) or dict(ready=False,action='quota_unavailable')))
    def forbidden(*args,**kwargs):
        raise AssertionError('Unavailable quota must not start a turn')
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    path=control_file(tmp_path)
    assert module.run(path,clock=lambda:4000)==0
    assert module.run(path,clock=lambda:4001)==0
    assert reads==['target']
    events=[json.loads(line) for line in (tmp_path/'hourly.jsonl').read_text().splitlines()]
    assert [e['action'] for e in events]==['quota_unavailable','interval_not_elapsed']


def test_wrong_execution_machine_never_reads_quota_or_delivers(tmp_path, monkeypatch):
    path=control_file(tmp_path)
    control=json.loads(path.read_text())
    control.update(execution_host='remote-ssh-discovered:lightbook', execution_machine_id='wrong-machine')
    path.write_text(json.dumps(control))
    def forbidden(*args, **kwargs):
        raise AssertionError('Wrong host must not inspect or dispatch')
    monkeypatch.setattr(module,'inspect',forbidden)
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    assert module.run(path,clock=lambda:4000)==1
    event=json.loads((tmp_path/'hourly.jsonl').read_text())
    assert event['action']=='check_failed'


def test_pending_message_does_not_deadlock_quota_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    states=iter([('usageLimited',1),('active',1),('active',1)])
    monkeypatch.setattr(module,'inspect',lambda *args:next(states))
    reads=[]
    monkeypatch.setattr(module,'recover_goal',lambda thread:(reads.append(thread) or
        dict(ready=True,action='quota_available_goal_resumed')))
    def forbidden(*args, **kwargs):
        raise AssertionError('Existing pending message must not be duplicated')
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    path=control_file(tmp_path)
    assert module.run(path,clock=lambda:4000)==0
    assert module.run(path,clock=lambda:4001)==0
    assert reads==['target']
    assert [json.loads(line)['action'] for line in (tmp_path/'hourly.jsonl').read_text().splitlines()]==[
        'already_queued','already_queued']


def test_pending_message_with_unavailable_quota_still_waits_one_hour(tmp_path, monkeypatch):
    monkeypatch.setattr(module,'inspect',lambda *args:('usageLimited',1))
    reads=[]
    monkeypatch.setattr(module,'recover_goal',lambda thread:(reads.append(thread) or
        dict(ready=False,action='quota_unavailable')))
    path=control_file(tmp_path)
    assert module.run(path,clock=lambda:4000)==0
    assert module.run(path,clock=lambda:4001)==0
    assert reads==['target']
    assert [json.loads(line)['action'] for line in (tmp_path/'hourly.jsonl').read_text().splitlines()]==[
        'quota_unavailable','interval_not_elapsed']


def test_local_stop_disables_quota_recovery_and_dispatch(tmp_path,monkeypatch):
    path=control_file(tmp_path);(tmp_path/'STOP').touch()
    monkeypatch.setattr(module,'inspect',lambda *args:('usageLimited',1))
    def forbidden(*args,**kwargs):raise AssertionError('STOP forbids recovery and delivery')
    monkeypatch.setattr(module,'recover_goal',forbidden)
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    assert module.run(path,clock=lambda:4000)==0
    assert json.loads((tmp_path/'hourly.jsonl').read_text())['action']=='disabled'


def test_pending_active_message_reaches_headless_dispatch_without_duplicate_queue(tmp_path,monkeypatch):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import start_queued_turn
    path=control_file(tmp_path)
    control=json.loads(path.read_text());control['start_queued_turns']=True
    path.write_text(json.dumps(control))
    monkeypatch.setattr(module,'inspect',lambda *args:('active',1))
    calls=[]
    monkeypatch.setattr(start_queued_turn,'start_queued_turn',lambda *args:
        (calls.append(args[:3]) or dict(action='thread_busy')))
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kwargs:
        (_ for _ in ()).throw(AssertionError('Do not add a duplicate')))
    assert module.run(path,clock=lambda:4000)==0
    assert len(calls)==1
    assert json.loads((tmp_path/'hourly.jsonl').read_text())['queued_start']['action']=='thread_busy'
