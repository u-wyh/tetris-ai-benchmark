"""One bounded, guarded CUDA8 evaluation phase; refuses stale/repeated attempts."""
import argparse
from contextlib import redirect_stdout
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import capture_gpu_evaluation as capture
from gpu_phase_guard import require_authorization, verify_live_guard
from inspect_gpu_evaluation import inspect
from own_policy_capture import sha
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now

VERSION = 'bounded-gpu8-phase-v1'


def boundary(state):
    return {key: state[key] for key in ['last_manifest_sha256', 'committed_chunks', 'round',
                                        'actual_unique_environment_steps']}


def plan(state, requested):
    if type(requested) is not int or not 1 <= requested <= 20:
        raise ValueError('Requested chunks must be an integer from one to twenty')
    if state['status'] not in ['fresh', 'resumable_partial', 'terminal_verified']:
        return dict(action='blocked', chunks=0, reason='Invalid or blocked cache: '+state['status'])
    if state['status'] == 'terminal_verified':
        return dict(action='preserved', chunks=0, reason='Terminal evaluation already verified')
    if state['status'] == 'resumable_partial' and not state['active_lanes']:
        return dict(action='finalize', chunks=1, round_upper_bound=0, environment_step_upper_bound=0)
    if state['status'] == 'resumable_partial' and not state['can_resume']:
        return dict(action='preserved', chunks=0, reason='Finite phase budget reached; no automatic extension')
    active = 8 if state['status'] == 'fresh' else len(state['active_lanes'])
    start = current = state['round']; steps = state['actual_unique_environment_steps']; chunks = 0
    for _ in range(requested):
        rounds = min(256, 20000-current)
        if (rounds <= 0 or current+rounds > state['phase_round_limit'] or
                steps+active*rounds > state['phase_environment_step_limit']):
            break
        current += rounds; steps += active*rounds; chunks += 1
    if not chunks:
        return dict(action='preserved', chunks=0, reason='Remaining finite budget cannot fit one complete chunk')
    return dict(action='capture', chunks=chunks, round_upper_bound=current-start,
                environment_step_upper_bound=steps-state['actual_unique_environment_steps'])


def sealed_sources(job):
    if job.get('bounded_phase_version') != VERSION:
        raise ValueError('A new explicitly sealed bounded-phase job is required; old caches cannot migrate')
    for name in ['run_gpu_evaluation_phase.py', 'gpu_phase_guard.py', 'inspect_gpu_evaluation.py']:
        path = ROOT/'scripts'/name
        if job['frozen_files'].get(str(path)) != sha(path):
            raise ValueError('Job must freeze bounded phase/guard/inspector source: '+name)


def finalization_accounted(folder, state):
    processes = [json.loads(p.read_text()) for p in folder.glob('process_*.json')]
    if (sum(p['environment_steps_this_process'] for p in processes) != state['actual_unique_environment_steps'] or
            any(sum(p[key] for p in processes) != state['round'] for key in
                ['candidate_forward_calls_this_process', 'shadow_forward_calls_this_process']) or
            any(p['optimizer_steps'] != 0 for p in processes)):
        raise ValueError('Missing committed process work; cannot fabricate a finalization receipt')


def execute(folder, request, attempt, initial, selected, job):
    """Add rejection checks after original validation, including its in-lock read.

    The frozen run function remains responsible for its lock and all computation.
    Neither hook changes the identity, snapshot, trace or return value it validates.
    """
    original_chain, original_inputs = capture.read_chain, capture.check_inputs
    checks = 0
    def checked_chain(path, identity):
        nonlocal checks
        saved, last, count = original_chain(path, identity)
        current = dict(initial, status='fresh' if saved is None else 'resumable_partial',
            last_manifest_sha256=last, committed_chunks=count,
            round=0 if saved is None else saved['round'],
            actual_unique_environment_steps=0 if saved is None else saved['environment_steps'],
            active_lanes=[] if saved is None else [i for i, w in enumerate(saved['workers']) if w['active']])
        if Path(path).resolve() != folder or identity != initial['identity'] or boundary(current) != request['expected_start']:
            raise RuntimeError('Committed boundary changed before/in capture lock; refuse stale budget')
        if plan(current, request['max_chunks']) != selected:
            raise RuntimeError('In-lock work bound differs from the sealed phase plan')
        checks += 1
        return saved, last, count
    def checked_inputs(design):
        original_inputs(design)
        require_authorization(job, attempt)
        verify_live_guard(request, attempt, job)
    try:
        capture.read_chain, capture.check_inputs = checked_chain, checked_inputs
        with redirect_stdout(sys.stderr):
            capture.run(folder, selected['chunks'])
    finally:
        capture.read_chain, capture.check_inputs = original_chain, original_inputs
    if checks < 2:
        raise RuntimeError('Frozen runner did not validate both preflight and locked boundaries')
    return checks


def run(attempt):
    attempt = Path(attempt).resolve()
    request_path = attempt/'request.json'; request_sha = sha(request_path)
    request = json.loads(request_path.read_text())
    folder = Path(request['job_folder']).resolve()
    if (request.get('version') != VERSION or request.get('mode') not in ['capture', 'guard_probe'] or
            not folder.is_relative_to(ROOT/'runs') or attempt.is_relative_to(folder)):
        raise ValueError('Explicit bounded phase request outside the original job is required')
    with (attempt/'attempt.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination = attempt/'phase_receipt.json'
        if destination.exists():
            receipt = json.loads(destination.read_text())
            if receipt['request_sha256'] != request_sha:
                raise ValueError('Completed attempt request changed; preserve evidence')
            return dict(receipt, reused_receipt=True)
        result = dict(at=now(), pid=os.getpid(), request_sha256=request_sha, job_folder=str(folder),
            new_environment_steps=0, gpu_forward_calls=0, optimizer_steps=0, completed_evaluation=False)
        try:
            if (attempt/'started.json').exists():
                raise RuntimeError('Prior attempt started without receipt; inspect its commits before creating a new attempt')
            if sha(folder/'design.json') != request['job_design_sha256']:
                raise ValueError('Requested job identity changed')
            initial = inspect(folder, request.get('upstream_summary'))
            selected = plan(initial, request['max_chunks'])
            result.update(initial_state=initial, plan=selected)
            if initial['status'] in ['fresh', 'resumable_partial', 'terminal_verified'] and boundary(initial) != request['expected_start']:
                raise RuntimeError('Request refers to a stale committed boundary')
            if request['mode'] == 'capture' and selected['action'] in ['preserved', 'blocked']:
                result.update(status=selected['action'], reason=selected['reason'])
            else:
                job = dict(json.loads((folder/'design.json').read_text()), job_folder=str(folder))
                require_authorization(job, attempt)
                if request['mode'] == 'capture':
                    sealed_sources(job)
                    if selected['action'] == 'finalize':
                        finalization_accounted(folder, initial)
                elif initial['status'] in ['invalid', 'blocked']:
                    raise RuntimeError('Guard probe requires a valid unchanged job')
                result['live_guard'] = verify_live_guard(request, attempt, job)
                if request['mode'] == 'guard_probe':
                    result.update(status='guard_verified_no_compute', reason='Live protection verified without capture/model loading')
                else:
                    durable_json(attempt/'started.json', dict(at=now(), pid=os.getpid(),
                        request_sha256=request_sha, expected_start=request['expected_start'], plan=selected))
                    # A failure may occur after uncommitted environment work.
                    # Never turn missing process accounting into zero work.
                    result.update(new_environment_steps=None, gpu_forward_calls=None,
                                  work_accounting='unknown until the frozen runner process receipt is verified')
                    result['boundary_checks'] = execute(folder, request, attempt, initial, selected, job)
                    process = json.loads((folder/f'process_{os.getpid()}.json').read_text())
                    if (process['environment_steps_this_process'] > selected['environment_step_upper_bound'] or
                            process['candidate_forward_calls_this_process'] > selected['round_upper_bound'] or
                            process['shadow_forward_calls_this_process'] != process['candidate_forward_calls_this_process'] or
                            process['optimizer_steps'] != 0):
                        raise RuntimeError('Actual process work exceeded the conservative phase bound')
                    result.update(status='phase_command_completed', process=process,
                        new_environment_steps=process['environment_steps_this_process'],
                        gpu_forward_calls=process['candidate_forward_calls_this_process']+process['shadow_forward_calls_this_process'],
                        work_accounting='verified frozen runner process receipt',
                        final_state=inspect(folder, request.get('upstream_summary')),
                        reason='Phase command completed; terminal completion still requires full audit and passed controller receipt')
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            result.update(status='blocked', reason=str(exc))
        durable_json(destination, result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.attempt_dir)
    print(json.dumps(result))
    sys.exit(3 if result['status'] == 'blocked' else 0)
