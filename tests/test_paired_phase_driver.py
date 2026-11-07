import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import continue_paired_branch_job as driver


def test_transient_unit_observation_repolls_same_handle_without_stop_or_relaunch(tmp_path,monkeypatch):
    job=tmp_path/'job';job.mkdir();(job/'design.json').write_text('{}')
    folder=tmp_path/'pipeline';folder.mkdir()
    start={'round':512};end={'round':513}
    (folder/'pipeline_design.json').write_text(json.dumps(dict(job=str(job),expected_start=start,frozen_files={})))
    launched=[];observed=[];stopped=[]
    monkeypatch.setattr(driver,'require_authorization',lambda *a:None)
    monkeypatch.setattr(driver,'stored_boundary',lambda *a:end if len(observed)>=3 else start)
    monkeypatch.setattr(driver.time,'sleep',lambda *a:None)
    def execute(args,**kwargs):
        if 'prepare_paired_branch_phase.py' in str(args[1]):
            attempt=Path(args[args.index('--attempt-dir')+1]);attempt.mkdir()
            (attempt/'summary.json').write_text(json.dumps(dict(status='passed')))
            (attempt/'receipt.json').write_text(json.dumps(dict(status='completed',pid=123,terminal=True,end=end,
                work=dict(environment_steps=8,gpu_forward_calls=1,optimizer_steps=0,model_loads=1))))
        elif args[0]=='systemd-run':launched.append(args)
        elif args[:3]==['systemctl','--user','stop']:stopped.append(args)
        else:raise AssertionError('Unexpected subprocess: '+str(args))
    monkeypatch.setattr(driver.subprocess,'run',execute)
    def observe(unit):
        observed.append(unit)
        if len(observed)==1:raise subprocess.TimeoutExpired('systemctl',10)
        return dict(ActiveState='active' if len(observed)==2 else 'inactive',MainPID='999',Result='success')
    monkeypatch.setattr(driver,'unit_state',observe)
    driver.run(folder)
    assert len(launched)==1 and not stopped
    assert len(observed)==3 and len(set(observed))==1
    result=json.loads((folder/'pipeline_result.json').read_text())
    assert result['status']=='completed' and len(result['observation_errors'])==1
