"""Finite serial phase driver; every compute phase retains its own live GPU guard."""
import argparse
import json
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from paired_branch_capture import sha,stored_boundary
from paired_branch_guard import require_authorization
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def unit_state(unit):
    raw=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID','-p','Result'],
        check=True,capture_output=True,text=True,timeout=10).stdout
    state=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    if not {'ActiveState','MainPID','Result'}<=state.keys():
        raise ValueError('Incomplete systemd observation; retain the same unit handle')
    return state


def phase_configuration(design):
    first, last = design.get('first_phase', 3), design.get('last_phase', 11)
    if type(first) is not int or type(last) is not int:
        raise ValueError('Invalid finite phase range')
    maximum = design.get('max_phases', last-first+1)
    prefix, suffix = design.get('unit_prefix', 'tetris-paired-r2'), design.get('unit_suffix', '20261107')
    if (type(maximum) is not int or
            not 1 <= first <= last <= 11 or not 1 <= last-first+1 == maximum <= 10 or
            type(prefix) is not str or not re.fullmatch(r'tetris-[a-z0-9-]+', prefix) or
            type(suffix) is not str or not re.fullmatch(r'[0-9]{8}', suffix)):
        raise ValueError('Invalid finite phase range or independent unit namespace')
    return range(first, last+1), prefix, suffix


def run(folder):
    design=json.loads((folder/'pipeline_design.json').read_text());job=Path(design['job'])
    ordinals, unit_prefix, unit_suffix = phase_configuration(design)
    jd=json.loads((job/'design.json').read_text());active_unit=None;completed=[];observation_errors=[]
    if (folder/'pipeline_result.json').exists():raise FileExistsError('Finite driver already terminal; inspect its result')
    def check():
        require_authorization(jd,folder)
        for path,want in design['frozen_files'].items():
            if sha(path)!=want:raise ValueError('Frozen finite driver input changed: '+path)
    try:
        check()
        if stored_boundary(job)!=design['expected_start']:
            raise ValueError('Finite driver start differs; inspect actual progress rather than relaunch')
        for ordinal in ordinals:
            check();attempt=folder.parent/f'phase_{ordinal:02d}'
            if attempt.exists():raise FileExistsError('Phase directory exists; do not duplicate or restart automatically')
            subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/prepare_paired_branch_phase.py'),
                '--job-folder',str(job),'--attempt-dir',str(attempt),'--max-chunks','2'],check=True,timeout=30)
            active_unit=f'{unit_prefix}-phase{ordinal:02d}-{unit_suffix}.service'
            subprocess.run(['systemd-run','--user','--unit='+active_unit,
                '--property=CPUQuota=50%','--property=MemoryMax=1G','--property=MemorySwapMax=0',
                '--property=Slice=tetris-compute.slice','--property=KillMode=mixed',
                '--property=TimeoutStopSec=40','--property=RuntimeMaxSec=680',
                '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',
                '--setenv=PATH=/home/wyh/.local/bin:/home/wyh/.nvm/versions/node/v24.18.0/bin:/usr/local/bin:/usr/bin:/bin',
                '--working-directory='+str(ROOT),str(ROOT/'.venv/bin/python'),
                str(ROOT/'scripts/verify_context_guards_when_idle.py'),'--run-dir',str(attempt)],check=True,timeout=15)
            while True:
                check()
                try:
                    state=unit_state(active_unit)
                except (OSError,subprocess.SubprocessError,ValueError) as exc:
                    observation_errors.append(dict(at=now(),unit=active_unit,error=repr(exc)))
                    durable_json(folder/'pipeline_status.json',dict(at=now(),stage='retrying_same_unit_observation',
                        unit=active_unit,error=repr(exc),completed_phases=completed))
                    time.sleep(2)
                    continue
                durable_json(folder/'pipeline_status.json',dict(at=now(),stage='waiting_live_phase',
                    unit=active_unit,unit_state=state,completed_phases=completed))
                if state['ActiveState'] not in ('active','activating','deactivating'):
                    if state['Result']!='success':raise RuntimeError('Phase controller failed: '+active_unit)
                    break
                time.sleep(2)
            summary=json.loads((attempt/'summary.json').read_text());receipt=json.loads((attempt/'receipt.json').read_text())
            if (summary['status']!='passed' or receipt['status']!='completed' or
                    receipt['work']['environment_steps']>4096 or receipt['work']['optimizer_steps']!=0 or
                    stored_boundary(job)!=receipt['end']):
                raise ValueError('Actual bounded phase result or checkpoint differs')
            completed.append(dict(attempt=str(attempt),pid=receipt['pid'],work=receipt['work'],end=receipt['end']))
            active_unit=None
            if receipt['terminal']:break
        terminal=bool(completed and receipt['terminal'])
        durable_json(folder/'pipeline_result.json',dict(at=now(),status='completed' if terminal else 'preserved_phase_budget',
            terminal=terminal,completed_phases=completed,end=stored_boundary(job),observation_errors=observation_errors,
            optimizer_steps=0,scope=f'At most{len(ordinals)} serial explicit phases,2chunks each. Source receipts are actual work; driver performs no policy computation.'))
    except BaseException as exc:
        durable_json(folder/'pipeline_result.json',dict(at=now(),status='stopped',error=repr(exc),
            completed_phases=completed,observation_errors=observation_errors,live_unit_at_stop=active_unit,
            scope='Preserve committed cache, inspect actual unit/cache before any resume'))
        raise
    finally:
        if active_unit is not None:
            subprocess.run(['systemctl','--user','stop',active_unit],check=False,timeout=50)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True,type=Path)
    run(p.parse_args().run_dir.resolve())
