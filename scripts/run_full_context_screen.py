"""Rebuild a matched resumable baseline, then hand off the bounded context screen."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from development_allowed import allowed
from resource_guard import ROOT,now,operation,write_json


def run(folder):
    if (folder/'handoff.json').exists():return
    lock=(folder/'bootstrap.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    design=json.loads((folder/'bootstrap_design.json').read_text());child=None
    deadline=time.monotonic()+design['wall_seconds']
    def interrupted(*args):raise KeyboardInterrupt('Full context baseline handoff interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def check():
        if not allowed(design['thread_id'],folder):raise RuntimeError('Goal stopped')
        if time.monotonic()>deadline:raise RuntimeError('Baseline handoff budget exhausted')
        for name,expected in design['frozen_files'].items():
            if digest(name)!=expected:raise RuntimeError('Frozen handoff input changed: '+name)
    try:
        check();baseline=folder/'baseline.json'
        if not baseline.exists():
            attempts=len(list((folder/'baseline_guards').glob('*'))) if (folder/'baseline_guards').exists() else 0
            guard=folder/'baseline_guards'/f'attempt_{attempts:02d}';guard.parent.mkdir(exist_ok=True)
            write_json(folder/'bootstrap_status.json',dict(at=now(),stage='matched_baseline',guard=str(guard)))
            with guard.with_suffix('.log').open('w') as stream:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/resource_guard.py'),
                    '--profile',str(folder/'evaluation_resources.json'),'--output',str(guard),
                    '--seconds','1200','--',sys.executable,'-u','-m','training.evaluation.evaluate_resumable',
                    '--model',design['frozen_model'],'--seeds-file',str(ROOT/'training/evaluation/seeds.json'),
                    '--seed-set','validation','--seed-count','8','--max-pieces','20000','--workers','2',
                    '--protocol','full_context_matched_baseline','--committed-steps','0','--output',str(baseline)],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                while child.poll() is None:check();time.sleep(2)
            result=json.loads((guard/'result.json').read_text())
            if child.returncode or result['outcome']!='completed':raise RuntimeError('Matched baseline failed')
        check();current=json.loads(baseline.read_text());legacy=json.loads(Path(design['legacy_baseline']).read_text())
        if current['per_seed']!=legacy['per_seed'] or current['aggregate']!=legacy['aggregate']:
            raise RuntimeError('Unchanged frozen model baseline differs from original evaluator outcomes')
        write_json(folder/'baseline_equivalence.json',dict(at=now(),status='passed',seeds=8,max_pieces=20000,
            baseline_sha256=digest(baseline),legacy_sha256=digest(design['legacy_baseline'])))
        protocol=dict(at=now(),thread_id=design['thread_id'],purpose=design['purpose'],cases=['full_context'],
            parameter_name='learning_rate',stage1_steps=262144,full_target_steps=262144,
            wall_budget_seconds=7200,baseline=str(baseline),baseline_sha256=digest(baseline),
            evaluation_module='training.evaluation.evaluate_resumable',evaluation_filename='stage1_evaluation.json',
            evaluation_resource_profile='evaluation_resources.json',evaluation_workers=2,
            evaluation_protocol='full_context_development_screen',
            training_source_sha256=design['training_source_sha256'],support_files_sha256=design['frozen_files'])
        protocol_path=folder/'protocol.json'
        if protocol_path.exists():
            existing=json.loads(protocol_path.read_text());protocol['at']=existing['at']
            if existing!=protocol:raise RuntimeError('Existing full context screen protocol differs')
        else:write_json(protocol_path,protocol)
        check()
        subprocess.run(['systemctl','--user','enable',design['controller_service']],check=True,timeout=20)
        subprocess.run(['systemctl','--user','start',design['controller_service']],check=True,timeout=20)
        write_json(folder/'handoff.json',dict(at=now(),status='handed_off',service=design['controller_service'],
            added_step_budget=262144,source_steps_offset=30003200,automatic_extension_above_budget=False))
        operation('完整上下文匹配基线逐局一致，已自动接力262144新增步数及8×20000门槛',folder/'handoff.json')
    except BaseException as exc:
        write_json(folder/'bootstrap_status.json',dict(at=now(),stage='stopped',error=repr(exc)))
        operation('完整上下文基线接力停止并保留缓存：'+repr(exc),folder)
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
