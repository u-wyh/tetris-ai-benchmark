"""Run predeclared AB/BA CUDA8 transaction pairs, with normal safeguards."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from hourly_continue import inspect
from keep_codex_thread import permitted
from resource_guard import ROOT,now,operation,write_json


def run(folder):
    if (folder/'gpu_summary.json').exists():return
    design=json.loads((folder/'gpu_design.json').read_text())
    design_hash=hashlib.sha256((folder/'gpu_design.json').read_bytes()).hexdigest()
    lock=(folder/'gpu_benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None;deadline=time.monotonic()+design['wall_seconds']
    def stopped(*args):raise KeyboardInterrupt('GPU feature benchmark interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stopped)
    def check():
        if design.get('control'):
            control_path=Path(design['control'])
            if not permitted(control_path, design['thread_id']):
                raise RuntimeError('Unattended control stopped or host/workspace changed')
        if hashlib.sha256((folder/'gpu_design.json').read_bytes()).hexdigest()!=design_hash:
            raise RuntimeError('Benchmark design changed')
        if inspect(Path.home()/'.codex',design['thread_id'])[0] not in ('active','usageLimited','usage_limited') or (folder/'STOP').exists():
            raise RuntimeError('Goal stopped')
        if time.monotonic()>deadline:raise RuntimeError('Benchmark time budget exhausted')
        for name,sha in design['frozen_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen input changed: '+name)
    def stage(label,command,profile):
        nonlocal child
        check();output=folder/'gpu_guards'/label;output.parent.mkdir(exist_ok=True);attempt=0
        while output.exists() or output.with_suffix('.log').exists():
            attempt+=1;output=folder/'gpu_guards'/f'{label}_retry{attempt:02d}'
        write_json(folder/'gpu_status.json',dict(at=now(),stage=label,guard=str(output)))
        with output.with_suffix('.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/resource_guard.py'),
                '--profile',str(folder/profile),'--output',str(output),'--seconds','300','--',*command],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:check();time.sleep(1)
        result=json.loads((output/'result.json').read_text())
        if child.returncode or result['outcome']!='completed':raise RuntimeError(label+' failed; inspect '+str(output))
    try:
        if design.get('preflight_tests'):
            receipt=folder/'preflight_passed.json'
            if receipt.exists():
                cached=json.loads(receipt.read_text())
                if cached.get('design_sha256')!=design_hash or cached.get('tests')!=design['preflight_tests']:
                    raise RuntimeError('Preflight identity changed')
            else:
                stage('preflight',[sys.executable,'-m','pytest','-q',*design['preflight_tests']],
                      'preflight_resources.json')
                write_json(receipt,dict(at=now(),design_sha256=design_hash,tests=design['preflight_tests']))
        for name in design['case_order']:
            case=folder/'gpu_pairs'/name;training=case/'training'
            for task in (1,2):
                state=json.loads((training/'transaction_state.json').read_text()) if (training/'transaction_state.json').exists() else {}
                if state.get('committed_task',0)>=task:continue
                if state.get('committed_task',0)!=task-1:raise RuntimeError('Unexpected benchmark task count')
                if state.get('pid'):
                    try:args=Path(f'/proc/{state["pid"]}/cmdline').read_bytes().split(b'\0')
                    except FileNotFoundError:args=[]
                    if b'training.train_vector_transaction' in args and str(training).encode() in args:
                        raise RuntimeError('Existing trainer is live')
                command=[sys.executable,'-u','-m','training.train_vector_transaction','--run-dir',str(training),
                         '--config-file',str(case/'preset.json'),'--max-tasks','1']
                if state:command.append('--resume')
                stage(f'{name}_task_{task:02d}',command,'training_resources.json')
        stage('verification',[sys.executable,str(ROOT/'scripts/verify_feature_backend_benchmark.py'),
                              '--run-dir',str(folder)],'evaluation_resources.json')
        if not (folder/'gpu_summary.json').exists():raise RuntimeError('Verifier produced no summary')
        write_json(folder/'gpu_status.json',dict(at=now(),stage='completed'))
        operation('候选环境后端两组AB/BA真实CUDA8事务比较完成；未发布模型',folder/'gpu_summary.json')
    except BaseException as exc:
        write_json(folder/'gpu_status.json',dict(at=now(),stage='stopped',error=repr(exc)))
        operation(f'特征后端GPU对照停止：{exc!r}，已保留事务与证据',folder)
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
