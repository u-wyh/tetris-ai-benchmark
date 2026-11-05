"""Verify staged context guards after the frozen live pipeline releases compute."""

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
from resource_guard import ROOT,now,write_json,operation


def run(folder):
    if (folder/'summary.json').exists():return
    lock=(folder/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    design=json.loads((folder/'design.json').read_text());deadline=time.monotonic()+design['wall_seconds'];child=None
    def stopped(*args):raise KeyboardInterrupt('Staged context verification interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stopped)
    def check():
        if not allowed(design['thread_id'],folder):raise RuntimeError('Goal stopped')
        if design.get('control'):
            control_path=Path(design['control']);control=json.loads(control_path.read_text())
            if (control.get('enabled') is not True or (control_path.parent/'STOP').exists()
                    or control.get('execution_machine_id')!=Path('/etc/machine-id').read_text().strip()):
                raise RuntimeError('Unattended control stopped or execution host changed')
        if time.monotonic()>deadline:raise RuntimeError('Staged validation wait budget exhausted')
        for name,expected in design['frozen_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Staged validation input changed: '+name)
    def pid(unit):
        result=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','--value'],
                              capture_output=True,text=True,check=True,timeout=10).stdout.strip()
        if not result.isdigit():raise RuntimeError('Cannot determine upstream process')
        return int(result)
    try:
        write_json(folder/'status.json',dict(at=now(),stage='waiting_for_frozen_pipeline'))
        while True:
            check()
            busy=any(pid(unit)>0 for unit in design['upstream_units'])
            guards=subprocess.run(['systemctl','--user','list-units','--type=service','--state=running',
                '--no-legend','--plain','tetris-guard-*.service'],capture_output=True,text=True,check=True,timeout=10)
            if not busy and not guards.stdout.strip():break
            time.sleep(5)
        check();guard=folder/'guard'
        if guard.exists():raise RuntimeError('Previous guard output exists; inspect before retry')
        write_json(folder/'status.json',dict(at=now(),stage='protected_gpu_validation',guard=str(guard)))
        tests=design.get('test_targets',['tests/test_full_context_validation_guards.py',
            'tests/test_full_context_runtime_guards.py','tests/test_full_context_initialization.py',
            'tests/test_initialization.py'])
        command=design.get('validation_command')
        if command is None:
            command=['/usr/bin/env','TETRIS_ORIGINAL_CONTEXT_DIR='+design['original_context'],
                     sys.executable,'-m','pytest','-q',*tests]
        with (folder/'guard.log').open('w') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/resource_guard.py'),
                '--profile',str(folder/'profile.json'),'--output',str(guard),'--seconds',str(design.get('command_seconds',180)),'--',
                *command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            while child.poll() is None:check();time.sleep(1)
        result=json.loads((guard/'result.json').read_text())
        if child.returncode or result['outcome']!='completed':raise RuntimeError('Staged runtime verification failed')
        write_json(folder/'summary.json',dict(at=now(),status='passed',source_worktree=str(ROOT),
            activation=design.get('activation','staged only; main frozen source unchanged'),guard=str(guard)))
        write_json(folder/'status.json',dict(at=now(),stage='completed'))
        operation(design.get('description','隔离副本上下文边界及实际CUDA模型配方校验通过；主实验来源未改变'),folder/'summary.json')
    except BaseException as exc:
        write_json(folder/'status.json',dict(at=now(),stage='stopped',error=repr(exc)))
        operation('隔离上下文验证停止并保留进度：'+repr(exc),folder)
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
