"""Run a frozen, guarded replay and longer-cap paired development screen."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from compare_development_evaluations import compare
from hourly_continue import inspect as inspect_goal
from resource_guard import ROOT, now, operation, write_json
from evaluation_identity import validate_evaluator, feature_backend


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extension_eligible(comparison, failures, gate):
    return (comparison['comparable']
            and comparison['mean_pieces_ratio'] >= gate['mean_pieces_ratio_at_least']
            and comparison['mean_observed_score_ratio'] >= gate['mean_observed_score_ratio_at_least']
            and failures['candidate'] <= failures['baseline']
            and all(p['candidate_pieces'] >= gate['no_seed_survival_ratio_below'] * p['baseline_pieces']
                    for p in comparison['per_seed']))


def run(folder):
    if (folder/'summary.json').exists() and (folder/'status.json').exists():
        if json.loads((folder/'status.json').read_text())['stage'] == 'completed':
            print(json.dumps(dict(already_completed=True)))
            return
    sys.path.insert(0, str(ROOT))
    from training.train_vector_transaction import training_source_identity

    protocol = json.loads((folder/'protocol.json').read_text())
    evaluation_module = validate_evaluator(protocol, protocol)
    lock = (folder/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + protocol['wall_seconds']
    child = None

    def interrupted(*args):
        raise KeyboardInterrupt('Longer development screen interrupted')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)

    def verify_sources():
        validate_evaluator(protocol, protocol)
        if training_source_identity() != protocol['training_source_sha256']:
            raise RuntimeError('Frozen training/environment source changed')
        for path, expected in protocol['frozen_files'].items():
            if digest(Path(path)) != expected:
                raise RuntimeError('Frozen experiment input changed: ' + path)

    def stage(label, command, seconds):
        nonlocal child
        verify_sources()
        status, _ = inspect_goal(Path.home()/'.codex', protocol['thread_id'])
        if status not in ('active', 'usageLimited', 'usage_limited') or (folder/'STOP').exists():
            raise RuntimeError('Goal stopped or STOP file set')
        remaining = int(deadline-time.monotonic())
        if remaining < 30:
            raise RuntimeError('Screen wall budget exhausted')
        guards = folder/'guards'
        guards.mkdir(exist_ok=True)
        attempt = 0
        while ((guards/f'{label}_{attempt:02d}').exists()
               or (guards/f'{label}_{attempt:02d}.log').exists()):
            attempt += 1
        output = guards/f'{label}_{attempt:02d}'
        write_json(folder/'status.json', dict(at=now(), stage=label, guard=str(output)))
        args = [sys.executable, '-u', str(ROOT/'scripts/resource_guard.py'), '--profile',
                str(folder/'resource_profile.json'), '--output', str(output), '--seconds',
                str(min(seconds, remaining)), '--', '/usr/bin/env', 'OMP_NUM_THREADS=1',
                'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', *command]
        with output.with_suffix('.log').open('w') as log:
            child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                status, _ = inspect_goal(Path.home()/'.codex', protocol['thread_id'])
                if status not in ('active', 'usageLimited', 'usage_limited') or (folder/'STOP').exists():
                    raise RuntimeError('Goal stopped or STOP file set')
                time.sleep(3)
        if not (output/'result.json').exists():
            raise RuntimeError(f'{label} guard exited {child.returncode} before a result; inspect {output.with_suffix(".log")}')
        result = json.loads((output/'result.json').read_text())
        if child.returncode or result['outcome'] != 'completed':
            raise RuntimeError(f'{label} failed: {result.get("error", result["outcome"])}; inspect {output}')
        verify_sources()

    operation('启动匹配种子的长局配对开发评测，后续按预声明门槛处理', folder)
    try:
        if protocol.get('diagnostic_required', True) and not (folder/'failure_replay.json').exists():
            stage('failure_replay', [sys.executable, '-u', str(ROOT/'scripts/diagnose_development_failure.py'),
                  '--baseline', protocol['short_baseline'], '--candidate', protocol['failed_short_candidate'],
                  '--output', str(folder/'failure_replay.json')], 180)
        results = {}
        for label, model in protocol['models'].items():
            verify_sources()
            output = folder/(label+'.json')
            if not output.exists():
                stage(label, [sys.executable, '-u', '-m', evaluation_module,
                      '--model', model['path'], '--seeds-file', protocol['seeds_file'],
                      '--seed-set', 'validation', '--seed-count', '8', '--max-pieces', '20000',
                      '--workers', '2', '--candidate-feature-backend',feature_backend(protocol),
                      '--protocol', 'learning_rate_longer_development_screen',
                      '--committed-steps', str(model['committed_steps']), '--output', str(output)], 1200)
            result = json.loads(output.read_text())
            if (result['model_sha256'] != model['sha256'] or result['max_pieces'] != 20000
                    or result['seeds'] != protocol['seeds']
                    or result['evaluation_source_sha256'] != protocol['evaluation_source_sha256']
                    or result['committed_steps'] != model['committed_steps']):
                raise RuntimeError('Cached evaluation identity mismatch')
            results[label] = result
        comparison = compare(results['baseline'], results['candidate'])
        if not comparison['comparable']:
            raise RuntimeError(comparison['reason'])
        failures = {name: sum(r['game_over'] for r in result['per_seed'])
                    for name, result in results.items()}
        gate = protocol['extension_gate']
        eligible = extension_eligible(comparison, failures, gate)
        summary = dict(at=now(), comparison=comparison, failures=failures,
                       eligible_for_training_extension=eligible,
                       action='Configured continuation applies the predeclared extension gate; no model promotion')
        write_json(folder/'summary.json', summary)
        write_json(folder/'status.json', dict(at=now(), stage='completed', eligible=eligible))
        operation(f'更长局开发评测完成，扩展门槛通过={eligible}；未替换正式模型', folder/'summary.json')
    except BaseException as exc:
        write_json(folder/'status.json', dict(at=now(), stage='stopped', error=repr(exc)))
        operation(f'更长局开发队列停止：{exc!r}，保留结果和断点', folder)
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
