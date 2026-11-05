"""Finish the approved development budget using frozen gates, without human prompts.

Waits for the longer-cap screen, resumes its selected run only if eligible, and
stops on regression, a protection failure, a user pause, or the bounded budget.
Does not promote models, change account limits, or invoke another AI session.
"""

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

from compare_development_evaluations import compare
from hourly_continue import inspect as inspect_goal
from resource_guard import ROOT, now, operation, write_json
from run_longer_development_screen import extension_eligible


def short_gate(baseline, candidate):
    paired = compare(baseline, candidate)
    return (paired['comparable'] and paired['mean_observed_score_ratio'] is not None
            and paired['mean_observed_score_ratio'] >= 0.99
            and all(row['survived_cap'] and not row['game_over'] for row in candidate['per_seed']))


def run(folder):
    if (folder/'summary.json').exists():
        previous = json.loads((folder/'summary.json').read_text())
        if previous['outcome'] in ('not_eligible', 'regression_stop', 'completed_budget'):
            print(json.dumps(dict(already_terminal=previous['outcome'])))
            return
    protocol = json.loads((folder/'protocol.json').read_text())
    lock = (folder/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + protocol['wall_seconds']
    child = None
    history = json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else []

    def interrupted(*args):
        raise KeyboardInterrupt('Unattended extension interrupted')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)

    def authorized():
        status, _ = inspect_goal(Path.home()/'.codex', protocol['thread_id'])
        if status not in ('active', 'usageLimited', 'usage_limited') or (folder/'STOP').exists():
            raise RuntimeError('Goal stopped or STOP file set')
        if time.monotonic() >= deadline:
            raise RuntimeError('Unattended development wall budget exhausted')

    def verify_inputs():
        for name, expected in protocol['frozen_files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Frozen input changed: ' + name)

    def stage(label, command, profile, seconds):
        nonlocal child
        authorized()
        verify_inputs()
        output = folder/'guards'/label
        attempt = 0
        while output.exists() or output.with_suffix('.log').exists():
            attempt += 1
            output = folder/'guards'/f'{label}_retry{attempt:02d}'
        output.parent.mkdir(exist_ok=True)
        write_json(folder/'status.json', dict(at=now(), stage=label, guard=str(output)))
        args = [sys.executable, '-u', str(ROOT/'scripts/resource_guard.py'), '--profile',
                str(profile), '--output', str(output), '--seconds',
                str(min(seconds, int(deadline-time.monotonic()))), '--', '/usr/bin/env',
                'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', *command]
        with output.with_suffix('.log').open('w') as log:
            child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                authorized()
                time.sleep(3)
        if not (output/'result.json').exists():
            raise RuntimeError(f'{label} guard exited {child.returncode}; inspect {output.with_suffix(".log")}')
        result = json.loads((output/'result.json').read_text())
        if child.returncode or result['outcome'] != 'completed':
            raise RuntimeError(f'{label}: {result.get("error", result["outcome"])}')
        verify_inputs()

    def finish(outcome, reason):
        result = dict(at=now(), outcome=outcome, reason=reason, history=history,
                      model_promotion='none', follow_up='Agent reviews evidence on next available authorized continuation')
        write_json(folder/'summary.json', result)
        write_json(folder/'status.json', dict(at=now(), stage=outcome, reason=reason))
        operation(f'无人值守开发结束：{outcome}，{reason}；未替换正式模型', folder/'summary.json')

    operation('启动无人值守续训决策：先等待长局门槛，通过后逐段训练至最多2097152步，退化即止损', folder)
    try:
        upstream = Path(protocol['longer_screen'])
        write_json(folder/'status.json', dict(at=now(), stage='waiting_for_longer_screen'))
        while not (upstream/'summary.json').exists():
            authorized()
            live = subprocess.run(['systemctl', '--user', 'show', protocol['upstream_service'],
                                   '-p', 'MainPID', '--value'], capture_output=True, text=True,
                                  check=True, timeout=10).stdout.strip()
            if (upstream/'summary.json').exists():
                break
            state = (json.loads((upstream/'status.json').read_text())
                     if (upstream/'status.json').exists() else dict(stage='starting'))
            if state['stage'] == 'stopped' or not live.isdigit() or int(live) == 0:
                raise RuntimeError('Upstream screen stopped; inspect its evidence before retrying')
            time.sleep(10)
        # Wait until the previous controller has released its final guard.
        while subprocess.run(['systemctl', '--user', 'is-active', '--quiet', protocol['upstream_service']]).returncode == 0:
            authorized()
            time.sleep(3)
        if json.loads((upstream/'status.json').read_text())['stage'] != 'completed':
            raise RuntimeError('Upstream summary exists but screen did not finish successfully')
        authorized()
        verify_inputs()
        screen = json.loads((upstream/'summary.json').read_text())
        if not screen['eligible_for_training_extension']:
            finish('not_eligible', 'Longer-cap screen failed the predeclared extension gate')
            return
        sys.path.insert(0, str(ROOT))
        from training.train_vector_transaction import verify_task, verify_training_source_identity
        from training.train_transaction import fsync_dir, fsync_tree

        training = Path(protocol['training'])
        config = json.loads((training/'config.json').read_text())
        if (config['device'] != 'cuda' or config['ppo']['n_envs'] != 8
                or config['task_steps'] != 4096 or config['target_total_steps'] != 2097152):
            raise RuntimeError('Training configuration differs from approved GPU8 budget')
        verify_training_source_identity(json.loads((training/'metadata.json').read_text()))
        baseline = json.loads(Path(protocol['short_baseline']).read_text())
        long_baseline = json.loads((upstream/'baseline.json').read_text())
        # Freeze completed upstream evidence as an extension input at the handoff.
        write_json(folder/'upstream_identity.json', {name: hashlib.sha256((upstream/name).read_bytes()).hexdigest()
                   for name in ['summary.json', 'baseline.json', 'candidate.json']})
        for steps in protocol['checkpoints']:
            authorized()
            verify_inputs()
            state = json.loads((training/'transaction_state.json').read_text())
            snapshot = folder/'checkpoints'/f'task_{steps//4096:06d}'
            old = next((row for row in history if row['steps'] == steps), None)
            if (old and old['passed_short_gate'] and state['committed_steps'] >= steps
                    and (steps not in (1048576, 2097152) or old.get('passed_longer_gate'))):
                verify_task(snapshot, steps//4096, config)
                continue
            if state.get('pid') and Path(f'/proc/{state["pid"]}/cmdline').exists():
                args = Path(f'/proc/{state["pid"]}/cmdline').read_bytes().split(b'\0')
                if b'training.train_vector_transaction' in args:
                    raise RuntimeError('Existing trainer is live; refusing duplicate launch')
            remaining = steps//4096-state['committed_task']
            if remaining < 0:
                raise RuntimeError('Run already advanced beyond this stage; review before resuming')
            if remaining:
                stage(f'train_{steps:09d}', [sys.executable, '-u', '-m', 'training.train_vector_transaction',
                      '--run-dir', str(training), '--config-file', protocol['preset'], '--resume',
                      '--max-tasks', str(remaining)], folder/'training_resources.json', 2400)
            state = json.loads((training/'transaction_state.json').read_text())
            if state['committed_steps'] != steps:
                raise RuntimeError('Trainer did not commit the requested stage')
            source = training/'committed'/f'task_{steps//4096:06d}'
            verify_task(source, steps//4096, config)
            snapshot = folder/'checkpoints'/source.name
            if not snapshot.exists():
                snapshot.parent.mkdir(exist_ok=True)
                pending = snapshot.with_suffix('.tmp')
                if pending.exists():
                    shutil.rmtree(pending)
                shutil.copytree(source, pending)
                verify_task(pending, steps//4096, config)
                fsync_tree(pending)
                pending.rename(snapshot)
                fsync_dir(snapshot.parent)
            verify_task(snapshot, steps//4096, config)
            periodic = json.loads((training/'evaluations/validation'/f'step_{steps:09d}_periodic.json').read_text())
            if periodic['model_sha256'] != hashlib.sha256((snapshot/'model.zip').read_bytes()).hexdigest():
                raise RuntimeError('Periodic evaluation model mismatch')
            record = dict(steps=steps, short_comparison=compare(baseline, periodic),
                          passed_short_gate=short_gate(baseline, periodic))
            history = [row for row in history if row['steps'] != steps]
            history.append(record)
            write_json(folder/'progress.json', history)
            if not record['passed_short_gate']:
                finish('regression_stop', f'Short development gate failed at {steps} steps')
                return
            if steps in (1048576, 2097152):
                output = folder/f'longer_{steps:09d}.json'
                if not output.exists():
                    stage(f'longer_{steps:09d}', [sys.executable, '-u', '-m', 'training.evaluation.evaluate',
                          '--model', str(snapshot/'model.zip'), '--seeds-file', protocol['seeds_file'],
                          '--seed-set', 'validation', '--seed-count', '8', '--max-pieces', '20000',
                          '--workers', '2', '--protocol', 'unattended_longer_development',
                          '--committed-steps', str(steps), '--output', str(output)], folder/'evaluation_resources.json', 1200)
                result = json.loads(output.read_text())
                if result['model_sha256'] != periodic['model_sha256'] or result['committed_steps'] != steps:
                    raise RuntimeError('Longer evaluation checkpoint mismatch')
                paired = compare(long_baseline, result)
                failures = dict(baseline=sum(r['game_over'] for r in long_baseline['per_seed']),
                                candidate=sum(r['game_over'] for r in result['per_seed']))
                passed = extension_eligible(paired, failures, protocol['extension_gate'])
                record.update(longer_comparison=paired, passed_longer_gate=passed)
                write_json(folder/'progress.json', history)
                if not passed:
                    finish('regression_stop', f'Longer development gate failed at {steps} steps')
                    return
        finish('completed_budget', 'Completed approved2097152-step budget with scheduled development reviews')
    except BaseException as exc:
        write_json(folder/'status.json', dict(at=now(), stage='stopped', error=repr(exc)))
        operation(f'无人值守续训停止：{exc!r}，检查点与证据保留', folder)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
