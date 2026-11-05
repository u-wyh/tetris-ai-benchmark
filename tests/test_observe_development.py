from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from observe_development import inspect_service, kernel_budget, last_sample, observe


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_stale_running_file_without_live_process_is_not_reported_as_training(tmp_path):
    save(tmp_path/'runs/autonomous_development_20261105/development_status.json',dict(stage='training'))
    def show(name):return dict(MainPID='0',ActiveState='inactive')
    result=observe(tmp_path,tmp_path/'missing_codex',show,lambda:[],cgroup_root=tmp_path/'cgroup')
    assert result['recorded_global_stage']=='training' and result['observed_activity']=='no_active_compute'
    assert result['goal_status']=='unknown' and not (tmp_path/'missing_codex').exists()


def test_live_guard_proves_training_and_reports_actual_cuda8_config(tmp_path):
    run=tmp_path/'runs/train';guard=tmp_path/'runs/guard';proc=tmp_path/'proc'
    save(run/'transaction_state.json',dict(status='running',committed_steps=8192))
    save(run/'config.json',dict(device='cuda',ppo=dict(n_envs=8)))
    save(guard/'launch.json',dict(profile=dict(sample_seconds=2)))
    (guard/'samples.jsonl').write_text(json.dumps(dict(at='2026-11-05T09:00:00+00:00',cpu_c=76))+'\n')
    args=['python','guarded_process.py','--heartbeat',str(guard/'heartbeat.json'),'--',
          'python','-m','training.train_vector_transaction','--run-dir',str(run)]
    p=proc/'42/cmdline';p.parent.mkdir(parents=True);p.write_bytes('\0'.join(args).encode()+b'\0')
    s=inspect_service(tmp_path,'guard.service',dict(MainPID='42',ActiveState='active'),proc,
                      datetime(2026,11,5,9,0,3,tzinfo=timezone.utc))
    assert s['process_live'] and s['activity']=='training' and s['sample_fresh']
    assert s['device']=='cuda' and s['n_envs']==8 and s['committed_steps']==8192
    assert 'cmdline' not in s  # Never publish raw arguments.


def test_kernel_budget_uses_cgroup_values(tmp_path):
    folder=tmp_path/'compute';folder.mkdir()
    for name,value in [('cpu.max','50000 20000'),('memory.current','1048576'),
                       ('memory.max','6442450944'),('memory.swap.max','0')]:
        (folder/name).write_text(value)
    assert kernel_budget('/compute',tmp_path)==dict(available=True,cpu_quota_percent=250.0,
        memory_current_mib=1.0,memory_max_mib=6144.0,memory_swap_max_mib=0.0)
    assert not kernel_budget('/../compute',tmp_path)['available']


def test_partial_last_sample_is_ignored_and_old_telemetry_is_flagged(tmp_path):
    guard=tmp_path/'runs/guard';guard.mkdir(parents=True)
    path=guard/'samples.jsonl';path.write_bytes(b'{"at":"2026-11-05T09:00:00+00:00","cpu_c":76}\n{"at":')
    assert last_sample(path)['cpu_c']==76
    proc=tmp_path/'proc/42';proc.mkdir(parents=True)
    (proc/'cmdline').write_bytes(('python\0guarded_process.py\0--heartbeat\0'+str(guard/'heartbeat.json')+'\0').encode())
    s=inspect_service(tmp_path,'guard.service',dict(MainPID='42',ActiveState='active'),tmp_path/'proc',
                      datetime(2026,11,5,9,1,0,tzinfo=timezone.utc))
    assert s['process_live'] and not s['sample_fresh'] and s['sample_age_seconds']==60
