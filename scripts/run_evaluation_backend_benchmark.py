"""Frozen serial reference/parity/cache and AB/BA evaluation speed benchmark."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from development_allowed import allowed
from resource_guard import ROOT,now,operation,write_json


def run(folder):
    if (folder/'summary.json').exists():return
    lock=(folder/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    design=json.loads((folder/'design.json').read_text());deadline=time.monotonic()+design['wall_seconds'];child=None
    def stopped(*args):raise KeyboardInterrupt('Evaluation benchmark interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stopped)
    def check():
        if not allowed(design['thread_id'],folder):raise RuntimeError('Goal stopped')
        if time.monotonic()>deadline:raise RuntimeError('Evaluation benchmark budget exhausted')
        for name,sha in design['frozen_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen evaluation input changed: '+name)
    def stage(label,command):
        nonlocal child
        check();output=folder/'guards'/label;output.parent.mkdir(exist_ok=True)
        if output.exists():raise RuntimeError('Existing stage output requires inspection: '+str(output))
        write_json(folder/'status.json',dict(at=now(),stage=label,guard=str(output)))
        with output.with_suffix('.log').open('w') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/resource_guard.py'),
                '--profile',str(folder/'profile.json'),'--output',str(output),'--seconds','300','--',*command],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            while child.poll() is None:check();time.sleep(1)
        result=json.loads((output/'result.json').read_text())
        if child.returncode or result['outcome']!='completed':raise RuntimeError(label+' failed')
    def evaluation(label,backend,cap,workers=2,module='training.evaluation.evaluate_resumable'):
        output=folder/(label+'.json')
        if output.exists():return
        stage(label,[sys.executable,'-u','-m',module,'--model',design['model'],
            '--seeds-file',str(folder/'seeds.json'),'--seed-set','validation','--seed-count','2',
            '--max-pieces',str(cap),'--workers',str(workers),'--diagnostics',
            '--candidate-feature-backend',backend,'--protocol','evaluation_backend_equivalence',
            '--committed-steps','0','--output',str(output)])
    try:
        if not (folder/'legacy_reference.json').exists():
            stage('archived_reference',[sys.executable,str(ROOT/'scripts/evaluation_backend_reference.py'),
                                       'reference','--run-dir',str(folder)])
        reference=json.loads((folder/'legacy_reference.json').read_text())
        records={}
        for label in design['case_order']:
            backend='bitboard' if label.endswith('bitboard') else 'official'
            evaluation(label,backend,design['cap'])
            result=json.loads((folder/(label+'.json')).read_text())
            if result['per_seed']!=reference['per_seed'] or result['aggregate']!=reference['aggregate']:
                raise RuntimeError('Exact archived diagnostic row/aggregate equivalence failed: '+label)
            if (result['candidate_feature_backend']!=backend or result['newly_evaluated_seeds']!=design['seeds']
                    or result['resumed_completed_seeds']):raise RuntimeError('Benchmark reused or mislabeled work')
            guard=folder/'guards'/label;timing=json.loads((guard/'result.json').read_text())['wall_seconds']
            samples=[json.loads(line) for line in (guard/'samples.jsonl').read_text().splitlines()]
            records[label]=dict(guard_process_wall_seconds=timing,evaluator_wall_seconds=result['wall_seconds'],
                quota150_throughout=all(s['cpu_quota_percent']==150 for s in samples),
                cpu_max_c=max(s['cpu_c'] for s in samples),gpu_max_c=max(s['gpu_c'] for s in samples),
                memory_max_mib=max(s['memory_mib'] for s in samples))
            write_json(folder/'partial_results.json',records)
        for label,workers,module in [('serial',1,'training.evaluation.evaluate_resumable'),
                                      ('spawn',2,'training.evaluation.evaluate_resumable'),
                                      ('legacy_cli',2,'training.evaluation.evaluate')]:
            evaluation(label,'bitboard',64,workers,module)
        small=[json.loads((folder/(label+'.json')).read_text()) for label in ('serial','spawn','legacy_cli')]
        assert all(r['per_seed']==small[0]['per_seed'] and r['aggregate']==small[0]['aggregate'] for r in small)
        if not (folder/'cache_isolation.json').exists():
            stage('cache_isolation',[sys.executable,str(ROOT/'scripts/evaluation_backend_reference.py'),
                                    'cache_isolation','--run-dir',str(folder)])
        speedups=[records[f'r{r}_official']['guard_process_wall_seconds']/
                  records[f'r{r}_bitboard']['guard_process_wall_seconds'] for r in (0,1)]
        geometric=statistics.geometric_mean(speedups)
        eligible=(all(v['quota150_throughout'] for v in records.values())
                  and all(s>1 for s in speedups) and geometric>=design['minimum_speedup'])
        write_json(folder/'summary.json',dict(at=now(),status='passed',records=records,paired_speedups=speedups,
            geometric_mean_guard_process_speedup=geometric,eligible_for_default_recommendation=eligible,
            minimum_speedup=design['minimum_speedup'],archived_reference_exact=True,
            serial_spawn_legacy_cli_exact=True,backend_cache_isolation='passed',workers=2,
            measured_scope='entire guarded command, including imports/model load/spawn, guard startup and polling',
            training_device_and_envs='unchanged CUDA8',final_test_used=False,model_promotion='none'))
        write_json(folder/'status.json',dict(at=now(),stage='completed'))
        operation('位图评测AB/BA及归档逐局/诊断、单进程/spawn/旧CLI与缓存隔离验证完成',folder/'summary.json')
    except BaseException as exc:
        write_json(folder/'status.json',dict(at=now(),stage='stopped',error=repr(exc)))
        operation('位图评测基准停止并保留证据：'+repr(exc),folder)
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
