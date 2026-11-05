"""Run a frozen, bounded learning-rate or reference-KL screen; never promote a model."""

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

from resource_guard import ROOT,now,operation,write_json
from hourly_continue import inspect as inspect_goal
from compare_development_evaluations import compare
from evaluation_identity import validate_evaluator


def parameter_value(preset, name):
    if name == 'learning_rate':
        return preset['ppo_overrides']['learning_rate']
    if name == 'kl_beta':
        return preset['kl_reference']['beta']
    if name == 'initial_optimizer':
        return preset.get('initial_optimizer', 'fresh')
    raise ValueError('Unsupported screened parameter: ' + name)


def select_candidate(reports, prefer_higher=False):
    eligible = [r for r in reports if r['eligible_for_longer_development']]
    if not eligible:
        return None
    highest = max(r['comparison']['mean_observed_score_ratio'] for r in eligible)
    close = [r for r in eligible if r['comparison']['mean_observed_score_ratio'] >= highest-0.005]
    choose = max if prefer_higher else min
    return choose(close, key=lambda r: r['parameter_value'])['case']


def run(folder):
    if (folder/'summary.json').exists() and (folder/'status.json').exists():
        if json.loads((folder/'status.json').read_text())['stage'] == 'completed_screen':
            print(json.dumps(dict(already_completed=True)))
            return
    sys.path.insert(0,str(ROOT))
    from training.train_vector_transaction import verify_task, training_source_identity, verify_training_source_identity
    protocol=json.loads((folder/'protocol.json').read_text())
    baseline=json.loads(Path(protocol['baseline']).read_text())
    parameter = protocol.get('parameter_name', 'learning_rate')
    count, cap = baseline['num_seeds'], baseline['max_pieces']
    evaluation_filename = protocol.get('evaluation_filename', 'stage1_8x5k.json')
    if hashlib.sha256(Path(protocol['baseline']).read_bytes()).hexdigest()!=protocol['baseline_sha256']:
        raise RuntimeError('Frozen baseline result changed')
    lock=(folder/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+protocol['wall_budget_seconds']
    child=None
    reports=[]

    def interrupted(*args):
        raise KeyboardInterrupt('development controller interrupted')

    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(signum,interrupted)

    def stage(label,command,seconds):
        nonlocal child
        if (folder/'STOP').exists():
            raise RuntimeError('STOP file requested termination')
        remaining=int(deadline-time.monotonic())
        if remaining<30:
            raise RuntimeError('Experiment wall budget exhausted')
        guards=folder/'guards';guards.mkdir(exist_ok=True)
        attempt=len(list(guards.glob(label+'_*')))
        output=guards/f'{label}_{attempt:02d}'
        profile = (folder/protocol['evaluation_resource_profile']
                   if label.endswith('_evaluation') and protocol.get('evaluation_resource_profile')
                   else folder/'resource_profile.json')
        args=[sys.executable,'-u',str(ROOT/'scripts/resource_guard.py'),'--profile',str(profile),
              '--output',str(output),'--seconds',str(min(seconds,remaining)),'--',*command]
        write_json(folder/'status.json',dict(at=now(),stage=label,guard=str(output)))
        with output.with_suffix('.log').open('w') as log:
            child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                goal_status,_=inspect_goal(Path.home()/'.codex',protocol['thread_id'])
                if goal_status not in ('active','usageLimited','usage_limited') or (folder/'STOP').exists():
                    raise RuntimeError('Goal stopped or local STOP file set')
                time.sleep(3)
        result=json.loads((output/'result.json').read_text())
        if child.returncode or result['outcome']!='completed':
            raise RuntimeError(f'{label} failed; inspect {output}')
        return result

    operation(f'启动{parameter}受控实验，各{protocol["stage1_steps"]}步开发筛选，不自动替换正式模型',folder)
    try:
        for label in protocol['cases']:
            for name, expected in protocol.get('support_files_sha256', {}).items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Frozen controller/resource support changed: ' + name)
            if training_source_identity()!=protocol['training_source_sha256']:
                raise RuntimeError('Frozen training source changed during the controlled experiment')
            evaluation_module = validate_evaluator(protocol, baseline)
            case=folder/label;training=case/'training';preset=json.loads((case/'preset.json').read_text())
            state_file=training/'transaction_state.json';state=json.loads(state_file.read_text()) if state_file.exists() else {}
            pid=state.get('pid')
            if pid and Path(f'/proc/{pid}/cmdline').exists():
                args=Path(f'/proc/{pid}/cmdline').read_bytes()
                if str(training).encode() in args and b'training.train_vector_transaction' in args:
                    raise RuntimeError('Training already live; do not duplicate it')
            remaining=protocol['stage1_steps']//4096-state.get('committed_task',0)
            if remaining<0:
                raise RuntimeError('Run has advanced beyond this screening protocol')
            if remaining:
                command=[sys.executable,'-u','-m','training.train_vector_transaction','--run-dir',str(training),
                         '--config-file',str(case/'preset.json'),'--max-tasks',str(remaining)]
                if state:command.append('--resume')
                stage(label+'_training',command,3600)
            config=json.loads((training/'config.json').read_text());n=protocol['stage1_steps']//4096
            verify_training_source_identity(json.loads((training/'metadata.json').read_text()))
            source=training/'committed'/f'task_{n:06d}';verify_task(source,n,config)
            snapshot=case/'stage1_checkpoint'/source.name
            if not snapshot.exists():
                from training.train_transaction import fsync_dir, fsync_tree
                snapshot.parent.mkdir(exist_ok=True)
                pending = snapshot.with_suffix('.tmp')
                if pending.exists():shutil.rmtree(pending)
                shutil.copytree(source,pending);verify_task(pending,n,config)
                fsync_tree(pending);pending.rename(snapshot);fsync_dir(snapshot.parent)
            verify_task(snapshot,n,config)
            evaluation=case/evaluation_filename
            if not evaluation.exists():
                stage(label+'_evaluation',[sys.executable,'-u','-m',evaluation_module,
                      '--model',str(snapshot/'model.zip'),'--seeds-file',protocol.get('seeds_file',str(ROOT/'training/evaluation/seeds.json')),
                      '--seed-set','validation','--seed-count',str(count),'--max-pieces',str(cap),
                      '--workers',str(protocol.get('evaluation_workers',0)),
                      '--protocol',protocol.get('evaluation_protocol','learning_rate_development_screen'),'--committed-steps',str(protocol['stage1_steps']),
                      '--output',str(evaluation)],900)
            result=json.loads(evaluation.read_text());comparison=compare(baseline,result)
            if not comparison['comparable']:
                raise RuntimeError('Evaluation identity mismatch: ' + comparison['reason'])
            if (result['model_sha256']!=hashlib.sha256((snapshot/'model.zip').read_bytes()).hexdigest()
                    or result['committed_steps']!=protocol['stage1_steps']):
                raise RuntimeError('Cached evaluation does not match this checkpoint')
            write_json(case/'comparison.json',comparison)
            eligible=(comparison.get('comparable') and all(r['survived_cap'] for r in result['per_seed'])
                      and comparison['mean_observed_score_ratio']>=0.99)
            reports.append(dict(case=label,learning_rate=preset['ppo_overrides']['learning_rate'],
                                parameter_name=parameter,parameter_value=parameter_value(preset,parameter),
                                eligible_for_longer_development=bool(eligible),comparison=comparison))
            write_json(folder/'partial_results.json',reports)
        best=select_candidate(reports, protocol.get('prefer_higher_parameter',False))
        summary=dict(at=now(),results=reports,recommended_for_review=best,
                     action='Screen complete; configured followup applies predeclared gates; no promotion')
        write_json(folder/'summary.json',summary)
        write_json(folder/'status.json',dict(at=now(),stage='completed_screen',recommended_for_review=best))
        operation(f'{parameter}开发筛选完成，门槛合格候选{best}；由已配置接力处理后续，未替换正式模型',folder/'summary.json')
    except BaseException as exc:
        write_json(folder/'status.json',dict(at=now(),stage='stopped',error=repr(exc)))
        operation(f'{parameter}对照停止：{exc!r}，已保留提交与日志',folder)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                child.kill();child.wait()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    run(args.run_dir.resolve())
