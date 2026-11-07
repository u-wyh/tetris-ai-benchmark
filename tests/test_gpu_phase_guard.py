import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import gpu_phase_guard as guard
from own_policy_capture import sha


@pytest.fixture
def protected(tmp_path, monkeypatch):
    import torch
    profile = json.loads((guard.ROOT/'configs/training_resources.json').read_text())
    root = tmp_path/'root'; root.mkdir(); monkeypatch.setattr(guard, 'ROOT', root)
    attempt = root/'attempt'; attempt.mkdir(); (attempt/'request.json').write_text('{}')
    controller = root/'controller'; controller.mkdir(); live = controller/'guard';live.mkdir()
    job = root/'job';job.mkdir();(job/'design.json').write_text('{}')
    files = [root/'configs/training_resources.json',controller/'profile.json']
    for p in files: p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(profile))
    for name in ['run_gpu_evaluation_phase.py','gpu_phase_guard.py','inspect_gpu_evaluation.py','resource_guard.py','guarded_process.py']:
        p=root/'scripts'/name;p.parent.mkdir(exist_ok=True);p.write_text('fixture')
        files.append(p)
    files += [attempt/'request.json',job/'design.json']
    command = [str(root/'.venv/bin/python'),str(root/'scripts/run_gpu_evaluation_phase.py'),'--attempt-dir',str(attempt)]
    d = dict(validation_command=command,thread_id='fixture',control=str(guard.CONTROL),command_seconds=180,
             frozen_files={str(p):sha(p) for p in files})
    (controller/'design.json').write_text(json.dumps(d))
    launch = dict(command=command,unit='tetris-guard-fixture',seconds=180,profile=profile,physical_cpus=[4,6,8,10,12,14])
    (live/'launch.json').write_text(json.dumps(launch))
    (live/'heartbeat.json').write_text(json.dumps(dict(pid=900,monotonic=time.monotonic())))
    group = '/user.slice/tetris-guard-fixture.service'
    kernel = {'/proc/self/cgroup':'0::'+group+'\n', '/proc/101/stat':'101 (fixture) S 200 0 0',
        '/sys/fs/cgroup'+group+'/cpu.max':'50000 20000',
        '/sys/fs/cgroup'+group+'/memory.max':str(6*1024**3),
        '/sys/fs/cgroup'+group+'/memory.swap.max':'0'}
    original = Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**kw:kernel[str(p)] if str(p) in kernel else original(p,*a,**kw))
    properties = dict(ActiveState='active', ControlGroup=group,MainPID='200',KillMode='control-group',OOMPolicy='stop',RuntimeMaxUSec='3min')
    monkeypatch.setattr(guard.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='\n'.join(k+'='+v for k,v in properties.items())))
    args={200:['python',str(root/'scripts/guarded_process.py'),'--heartbeat',str(live/'heartbeat.json'),'--guardian-pid','900'],
          900:['python',str(root/'scripts/resource_guard.py'),'--output',str(live),'--profile',str(controller/'profile.json')]}
    monkeypatch.setattr(guard,'process_args',lambda pid:args[pid])
    monkeypatch.setattr(guard,'require_authorization',lambda *a:None)
    monkeypatch.setattr(guard,'os',SimpleNamespace(getpid=lambda:101,sched_getaffinity=lambda pid:{4,6,8,10,12,14}))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    return dict(request=dict(controller_design=str(controller/'design.json'),guard_dir=str(live)),attempt=attempt,
        job=dict(job_folder=str(job),thread_id='fixture'),kernel=kernel,properties=properties,args=args,live=live,root=root)


def verify(p):
    return guard.verify_live_guard(p['request'],p['attempt'],p['job'])


def test_live_kernel_guard_ancestry_heartbeat_profile_and_cuda(protected):
    r=verify(protected)
    assert r['guardian_pid']==900 and r['watchdog_pid']==200 and r['cpu_cores_limit']==2.5
    assert r['memory_max_bytes']==6*1024**3 and r['memory_swap_bytes']==0


@pytest.mark.parametrize('change', ['cgroup','cpu','memory','swap','affinity','runtime','heartbeat','watchdog','guardian','cuda','source','ended'])
def test_unprotected_evidence_rejected(protected,monkeypatch,change):
    p=protected; base='/sys/fs/cgroup/user.slice/tetris-guard-fixture.service/'
    if change=='cgroup':p['kernel']['/proc/self/cgroup']='0::/user.slice/unprotected.scope\n'
    elif change=='cpu':p['kernel'][base+'cpu.max']='max 20000'
    elif change=='memory':p['kernel'][base+'memory.max']='max'
    elif change=='swap':p['kernel'][base+'memory.swap.max']='1024'
    elif change=='affinity':monkeypatch.setattr(guard.os,'sched_getaffinity',lambda pid:{0,1})
    elif change=='runtime':p['properties']['RuntimeMaxUSec']='infinity'
    elif change=='heartbeat':monkeypatch.setattr(guard,'heartbeat_ok',lambda *a:False)
    elif change=='watchdog':p['args'][200][1]='fake-watchdog.py'
    elif change=='guardian':p['args'][900][1]='fake-guardian.py'
    elif change=='cuda':
        import torch
        monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    elif change=='source':(p['root']/'scripts/gpu_phase_guard.py').write_text('changed')
    else:(p['live']/'result.json').write_text('{}')
    with pytest.raises((ValueError,RuntimeError)):verify(p)


@pytest.mark.parametrize('goal,enabled,stop,host,allowed', [
    ('active',True,False,True,True),('usageLimited',True,False,True,False),
    ('paused',True,False,True,False),('active',False,False,True,False),
    ('active',True,True,True,False),('active',True,False,False,False)])
def test_host_goal_pause_quota_and_stop_are_fail_closed(tmp_path,monkeypatch,goal,enabled,stop,host,allowed):
    control=tmp_path/'control.json';monkeypatch.setattr(guard,'CONTROL',control)
    control.write_text(json.dumps(dict(enabled=enabled,execution_machine_id=guard.MACHINE,thread_id='fixture')))
    attempt=tmp_path/'attempt';attempt.mkdir();jobfolder=tmp_path/'job';jobfolder.mkdir()
    if stop:(attempt/'STOP').touch()
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**kw: (guard.MACHINE if host else 'other')
        if str(p)=='/etc/machine-id' else original(p,*a,**kw))
    class DB:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def execute(self,*a):return self
        def fetchone(self):return (goal,)
    monkeypatch.setattr(guard.sqlite3,'connect',lambda *a,**kw:DB())
    job=dict(job_folder=str(jobfolder),control=str(control),thread_id='fixture')
    if allowed:guard.require_authorization(job,attempt)
    else:
        with pytest.raises(RuntimeError):guard.require_authorization(job,attempt)
