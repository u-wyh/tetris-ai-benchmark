"""Build a resumable frozen-model long baseline with the verified evaluation backend."""

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
    def interrupted(*args):raise KeyboardInterrupt('Bitboard baseline interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def check():
        if not allowed(design['thread_id'],folder):raise RuntimeError('Goal stopped')
        if time.monotonic()>deadline:raise RuntimeError('Baseline wall budget exhausted')
        for name,sha in design['frozen_files'].items():
            if digest(name)!=sha:raise RuntimeError('Frozen baseline input changed: '+name)
    try:
        check();output=folder/'baseline.json';guards=folder/'guards';guards.mkdir(exist_ok=True)
        if not output.exists():
            attempt=0;guard=guards/f'attempt_{attempt:02d}'
            while guard.exists():attempt+=1;guard=guards/f'attempt_{attempt:02d}'
            write_json(folder/'status.json',dict(at=now(),stage='matched_bitboard_baseline',guard=str(guard)))
            with guard.with_suffix('.log').open('w') as stream:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/resource_guard.py'),
                    '--profile',str(folder/'profile.json'),'--output',str(guard),'--seconds','1200','--',
                    sys.executable,'-u','-m',design['evaluation_module'],'--model',design['model'],
                    '--seeds-file',str(ROOT/'training/evaluation/seeds.json'),'--seed-set','validation',
                    '--seed-count','8','--max-pieces','20000','--workers',str(design['evaluation_workers']),
                    '--candidate-feature-backend',design['candidate_feature_backend'],
                    '--protocol','verified_bitboard_long_baseline','--committed-steps','0','--output',str(output)],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                while child.poll() is None:check();time.sleep(2)
            outcome=json.loads((guard/'result.json').read_text())
            if child.returncode or outcome['outcome']!='completed':raise RuntimeError('Bitboard baseline evaluation failed')
        check();actual=json.loads(output.read_text());legacy=json.loads(Path(design['legacy_baseline']).read_text())
        from evaluation_identity import validate_evaluator
        validate_evaluator(design,actual)
        if (actual['model_sha256']!=digest(design['model']) or actual['model_sha256']!=legacy['model_sha256']
                or actual['seeds']!=legacy['seeds'] or actual['max_pieces']!=20000
                or actual['per_seed']!=legacy['per_seed'] or actual['aggregate']!=legacy['aggregate']):
            raise RuntimeError('Full frozen-model baseline outcomes differ')
        write_json(folder/'summary.json',dict(at=now(),status='passed',baseline=str(output),
            baseline_sha256=digest(output),legacy_outcomes_exact=True,seeds=8,max_pieces=20000,
            candidate_feature_backend=design['candidate_feature_backend'],final_test_used=False,training_steps_added=0))
        write_json(folder/'status.json',dict(at=now(),stage='completed'))
        operation('已验证位图8×20000匹配基线，逐局及聚合与历史冻结模型精确一致',folder/'summary.json')
    except BaseException as exc:
        write_json(folder/'status.json',dict(at=now(),stage='stopped',error=repr(exc)))
        operation('位图基线停止并保留逐seed缓存：'+repr(exc),folder)
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
