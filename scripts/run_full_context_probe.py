"""Persistently run a frozen, bounded CUDA8 full-context continuation probe."""

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
    if (folder/'probe_summary.json').exists():return
    lock=(folder/'probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    design=json.loads((folder/'design.json').read_text());child=None
    deadline=time.monotonic()+design['wall_seconds']
    def stop(*args):raise KeyboardInterrupt('Full context probe interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    def check():
        if not allowed(design['thread_id'],folder):raise RuntimeError('Goal stopped or STOP file')
        if time.monotonic()>deadline:raise RuntimeError('Full context probe time budget exhausted')
        for path,expected in design['frozen_files'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Frozen probe input changed: '+path)
    def stage(label,command,profile,seconds):
        nonlocal child
        check();output=folder/'guards'/label;output.parent.mkdir(exist_ok=True)
        attempt=0
        while output.exists():
            attempt+=1;output=folder/'guards'/f'{label}_retry{attempt:02d}'
        write_json(folder/'status.json',dict(at=now(),stage=label,guard=str(output)))
        with output.with_suffix('.log').open('w') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/resource_guard.py'),
                '--profile',str(folder/profile),'--output',str(output),'--seconds',str(seconds),'--',*command],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            while child.poll() is None:check();time.sleep(1)
        result=json.loads((output/'result.json').read_text())
        if child.returncode or result['outcome']!='completed':
            raise RuntimeError(label+' failed: '+str(output))
    try:
        if not (folder/'direct/task_000002/vector_env_state.pkl').exists():
            # Incomplete reference output is never treated as a recoverable checkpoint.
            if (folder/'direct').exists():raise RuntimeError('Incomplete reference; retain and inspect before retry')
            stage('direct',[sys.executable,str(ROOT/'scripts/probe_full_context_step.py'),'direct',
                           '--run-dir',str(folder)],'training_resources.json',300)
        for number in (1,2):
            state_path=folder/'training/transaction_state.json'
            state=json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get('committed_task',0)>=number:continue
            assert state.get('committed_task',0)==number-1
            command=[sys.executable,'-u','-m','training.train_vector_transaction','--run-dir',str(folder/'training'),
                     '--config-file',str(folder/'preset.json'),'--max-tasks','1']
            if state:command.append('--resume')
            stage(f'rebased_task_{number:02d}',command,'training_resources.json',300)
        stage('verify',[sys.executable,str(ROOT/'scripts/probe_full_context_step.py'),'verify',
                        '--run-dir',str(folder)],'evaluation_resources.json',120)
        write_json(folder/'status.json',dict(at=now(),stage='completed'))
        operation('完整上下文直接续接与新步数位图恢复探针通过；未使用最终测试或发布模型',folder/'probe_summary.json')
    except BaseException as exc:
        write_json(folder/'status.json',dict(at=now(),stage='stopped',error=repr(exc)))
        operation('完整上下文探针停止并保留证据：'+repr(exc),folder)
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
