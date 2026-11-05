"""Verify corrected RNG/CUDA recovery and the resumable screen before rollout."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from evaluation_identity import RESUMABLE
from hourly_continue import inspect
from resource_guard import ROOT, now, operation, write_json
from run_parameter_followup import ensure_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(folder):
    if (folder/'pilot_summary.json').exists():
        return
    design=json.loads((folder/'design.json').read_text())
    lock=(folder/'pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+design['wall_seconds'];child=None

    def stopped(*args):raise KeyboardInterrupt('RNG pilot interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stopped)

    def check():
        if inspect(Path.home()/'.codex',design['thread_id'])[0] not in ('active','usageLimited','usage_limited') or (folder/'STOP').exists():
            raise RuntimeError('Goal stopped')
        if time.monotonic()>deadline:raise RuntimeError('Pilot deadline exhausted')
        for name,sha in design['frozen_files'].items():
            if digest(name)!=sha:raise RuntimeError('Frozen pilot input changed: '+name)

    def stage(label,command,seconds,profile):
        nonlocal child
        check();output=folder/label;attempt=0
        while output.exists() or output.with_suffix('.log').exists():
            attempt+=1;output=folder/f'{label}_retry{attempt:02d}'
        write_json(folder/'pilot_status.json',dict(at=now(),stage=label,guard=str(output)))
        with output.with_suffix('.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/resource_guard.py'),
                 '--profile',str(folder/profile),'--output',str(output),'--seconds',str(seconds),'--',*command],
                 cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:check();time.sleep(1)
        result=json.loads((output/'result.json').read_text())
        if child.returncode or result['outcome']!='completed':raise RuntimeError(f'{label} failed; inspect {output}')

    def evaluate(label,output,seeds,count,cap):
        stage(label,[sys.executable,'-u','-m',RESUMABLE,'--model',design['reference_model'],
             '--seeds-file',str(seeds),'--seed-set','validation','--seed-count',str(count),
             '--max-pieces',str(cap),'--workers','2','--protocol','rng_isolation_baseline',
             '--committed-steps','0','--output',str(output)],1200,'evaluation_resources.json')

    def protocol(baseline,cases,steps):
        return dict(at=now(),thread_id=design['thread_id'],purpose=design['purpose'],cases=cases,
            parameter_name='initial_optimizer',stage1_steps=steps,full_target_steps=2097152,
            wall_budget_seconds=10800,baseline=str(baseline),baseline_sha256=digest(baseline),
            evaluation_module=RESUMABLE,evaluation_filename='stage1_evaluation.json',
            evaluation_resource_profile='evaluation_resources.json',evaluation_workers=2,
            evaluation_protocol='rng_isolated_development_screen',
            training_source_sha256=design['training_source_sha256'],support_files_sha256=design['frozen_files'],
            fixed_parameters=design['fixed_parameters'])

    try:
        case=folder/'rng_isolated';training=case/'training'
        if not (folder/'smoke_verification.json').exists():
            for n in (1,2):
                state=json.loads((training/'transaction_state.json').read_text()) if (training/'transaction_state.json').exists() else {}
                if state.get('committed_task',0)>=n:continue
                if state.get('pid'):
                    try:
                        args=Path(f'/proc/{state["pid"]}/cmdline').read_bytes().split(b'\0')
                    except FileNotFoundError:
                        args=[]
                    if b'training.train_vector_transaction' in args and str(training).encode() in args:
                        raise RuntimeError('Existing trainer is live; refusing duplicate launch')
                if state.get('committed_task',0)!=n-1:raise RuntimeError('Unexpected pilot transaction')
                command=[sys.executable,'-u','-m','training.train_vector_transaction','--run-dir',str(training),
                         '--config-file',str(case/'preset.json'),'--max-tasks','1']
                if state:command.append('--resume')
                stage(f'smoke_task_{n:02d}',command,240,'resource_profile.json')
            stage('pilot_verification_guard',[sys.executable,str(folder/'verify_smoke.py')],180,'evaluation_resources.json')
        smoke=Path(design['controller_smoke']);smoke.mkdir(exist_ok=True)
        for filename in ('resource_profile.json','evaluation_resources.json','smoke_verification.json'):
            destination=smoke/filename
            if not destination.exists():shutil.copyfile(folder/filename,destination)
        ensure_json(smoke/'seeds.json',dict(validation=[100000,100001]))
        small=smoke/'baseline.json'
        if not small.exists():evaluate('controller_baseline',small,smoke/'seeds.json',2,64)
        smoke_case=smoke/'rng_isolated';smoke_case.mkdir(exist_ok=True)
        if not (smoke_case/'preset.json').exists():shutil.copyfile(case/'preset.json',smoke_case/'preset.json')
        if not (smoke_case/'training').exists():(smoke_case/'training').symlink_to(training,target_is_directory=True)
        small_protocol=protocol(small,['rng_isolated'],8192);small_protocol['seeds_file']=str(smoke/'seeds.json')
        ensure_json(smoke/'protocol.json',small_protocol)
        check();subprocess.run(['systemctl','--user','start',design['smoke_service']],check=True,timeout=20)
        while True:
            check()
            pid=subprocess.check_output(['systemctl','--user','show',design['smoke_service'],'-p','MainPID','--value'],text=True,timeout=10).strip()
            if pid=='0':break
            time.sleep(2)
        state=json.loads((smoke/'status.json').read_text())
        summary=json.loads((smoke/'summary.json').read_text())
        if state['stage']!='completed_screen' or not summary['results'][0]['comparison']['comparable']:
            raise RuntimeError('Resumable controller smoke failed')
        write_json(folder/'controller_smoke_verification.json',dict(at=now(),status='passed',
            smoke=str(smoke),evaluation_module=RESUMABLE,training_reused_without_new_steps=True))
        baseline=folder/'baseline.json'
        if not baseline.exists():evaluate('baseline_guard',baseline,ROOT/'training/evaluation/seeds.json',8,20000)
        data=json.loads(baseline.read_text());legacy=json.loads(Path(design['legacy_baseline']).read_text())
        if data['per_seed']!=legacy['per_seed'] or data['aggregate']!=legacy['aggregate']:
            raise RuntimeError('Full baseline outcomes differ from the unchanged original evaluator')
        write_json(folder/'baseline_equivalence.json',dict(at=now(),status='passed',seeds=8,max_pieces=20000,
                   old_sha256=design['frozen_files'][design['legacy_baseline']],new_sha256=digest(baseline)))
        ensure_json(folder/'protocol.json',protocol(baseline,['rng_isolated'],262144))
        followup=Path(design['followup_dir']);followup.mkdir(exist_ok=True)
        extra=Path(design['extra_dir'])
        frozen={**design['frozen_files'],str(folder/'protocol.json'):digest(folder/'protocol.json'),
                str(baseline):digest(baseline),str(training/'config.json'):digest(training/'config.json')}
        followup_protocol=dict(at=now(),thread_id=design['thread_id'],wall_seconds=14400,
            parameter_screen=str(folder),upstream_service=design['control_service'],heldout=str(extra),
            extension=str(extra/'extension'),heldout_service=f'tetris-development-pair@{extra.name}.service',
            extension_service=f'tetris-development-extension@{extra.name}.service',extra_seeds=list(range(100008,100016)),
            training_resources=str(folder/'resource_profile.json'),evaluation_resources=str(folder/'evaluation_resources.json'),
            short_baseline=design['short_baseline'],extension_gate=design['extension_gate'],frozen_files=frozen,
            interpretation='Exploratory development, correct RNG initialization and resumable evaluation; no final test or promotion')
        ensure_json(followup/'protocol.json',followup_protocol)
        check()
        subprocess.run(['systemctl','--user','enable',design['control_service'],design['followup_service']],check=True,timeout=20)
        for service in (design['control_service'],design['followup_service']):
            subprocess.run(['systemctl','--user','start',service],check=True,timeout=20)
        write_json(folder/'pilot_summary.json',dict(at=now(),status='passed',outcome='screen_started',
            controller_smoke=True,full_baseline_equivalent=True,model_promotion='none'))
        write_json(folder/'pilot_status.json',dict(at=now(),stage='handed_off'))
        operation('正确run RNG的CUDA8恢复、可恢复控制器与完整基线一致性通过，自动继续262144步开发筛选',folder/'pilot_summary.json')
    except BaseException as exc:
        write_json(folder/'pilot_status.json',dict(at=now(),stage='stopped',error=repr(exc)))
        operation(f'RNG隔离接力停止：{exc!r}，保留已提交任务与评测缓存',folder)
        raise
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=25)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    run(parser.parse_args().run_dir.resolve())
