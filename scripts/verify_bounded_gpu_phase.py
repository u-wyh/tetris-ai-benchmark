"""Audit a real bounded CUDA phase, exact frozen prefix, reuse and full-budget refusal."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture_gpu_evaluation import prepare
from restore_own_death_contexts import load_verified_chunk
from verify_feature_backend_benchmark import identical
from own_policy_capture import sha
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def files(folder):
    return {str(p): sha(p) for p in folder.rglob('*') if p.is_file()}


def run(folder):
    if (folder/'acceptance_receipt.json').exists():
        raise FileExistsError('Bounded positive acceptance already complete; preserve it')
    design = json.loads((folder/'design.json').read_text())
    protocol = json.loads(Path(design['protocol']).read_text())
    job, attempt = Path(protocol['job_folder']), Path(protocol['attempt_folder'])
    controller = json.loads(Path(protocol['compute_controller_summary']).read_text())
    phase = json.loads((attempt/'phase_receipt.json').read_text())
    request = json.loads((attempt/'request.json').read_text())
    watchdog = json.loads((Path(protocol['compute_guard_folder'])/'watchdog_result.json').read_text())
    if (len(protocol['reference_chunks']) != 2 or watchdog['reason'] != 'command_completed' or
            watchdog['exit_code'] != 0 or watchdog['child_pid'] != phase['pid']):
        raise ValueError('Frozen oracle scope or actual guarded source process differs')
    if (controller['status'] != 'passed' or phase['status'] != 'phase_command_completed' or
            request['max_chunks'] != 20 or phase['plan']['chunks'] != 1 or phase['boundary_checks'] < 2 or
            phase['new_environment_steps'] != 2048 or phase['gpu_forward_calls'] != 512 or phase['optimizer_steps'] != 0 or
            phase['plan']['round_upper_bound'] != 256 or phase['plan']['environment_step_upper_bound'] != 2048):
        raise ValueError('Actual CUDA phase did not match the frozen bounded protocol')
    job_design, identity, (snapshot, last_sha, count) = prepare(job)
    if (count != 2 or snapshot['round'] != 256 or snapshot['environment_steps'] != 2048 or
            job_design['phase_round_limit'] != 256 or job_design['phase_environment_step_limit'] != 2048 or
            phase['final_state']['status'] != 'resumable_partial' or phase['final_state']['can_resume'] or
            not phase['final_state']['phase_budget_reached'] or phase['final_state']['completed']):
        raise ValueError('Bounded job was extended or incorrectly reported as complete')
    expected_models = {name: spec['counts'] for name, spec in job_design['models'].items()}
    for number, reference in enumerate(protocol['reference_chunks']):
        expected, expected_rows, _ = load_verified_chunk(Path(reference['directory']),
            reference['manifest_sha256'], protocol['reference_identity'])
        directory = job/f'chunk_{number:04d}'
        actual, rows, manifest = load_verified_chunk(directory, sha(directory/'manifest.json'), identity)
        if not identical((actual, rows), (expected, expected_rows)):
            raise ValueError('Bounded CUDA trace or complete worker/RNG/views differ from frozen oracle')
        audit = manifest['audit']
        if (audit['policy_adam_counts_unchanged'] is not True or audit['global_sampling_rng_unchanged'] is not True or
                audit['optimizer_steps'] != 0 or audit['model_counts'] != expected_models or
                [w['state']['episode_seed'] for w in actual['workers']] != job_design['seeds']):
            raise ValueError('Actual seed order or frozen policy/Adam/counters/RNG audit differs')
    processes = list(job.glob('process_*.json'))
    if (len(processes) != 1 or processes[0].name != f'process_{phase["pid"]}.json' or
            json.loads(processes[0].read_text()) != phase['process']):
        raise ValueError('Actual source process accounting differs')
    protected = files(job)
    original_receipt_sha = sha(attempt/'phase_receipt.json')
    def invoke(path):
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts/run_gpu_evaluation_phase.py'),
            '--attempt-dir', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = child.communicate()
        if child.returncode:
            raise RuntimeError('Phase CLI acceptance failed: '+str(child.pid)+' '+stderr[-1000:])
        return child.pid, json.loads(stdout)
    reuse_pid, reused = invoke(attempt)
    if (reused.get('reused_receipt') is not True or reused['pid'] != phase['pid'] or
            sha(attempt/'phase_receipt.json') != original_receipt_sha or files(job) != protected):
        raise ValueError('Repeat attempt changed the source receipt or performed new work')
    # reused.new_environment_steps is the original attempt's historical total;
    # invariant cache/process files and original PID prove no second computation.
    refusal_folder = Path(design['refusal_attempt'])
    refusal_pid, refused = invoke(refusal_folder)
    if (refused['status'] != 'preserved' or refused['plan']['chunks'] != 0 or
            any(refused[key] != 0 for key in ['new_environment_steps', 'gpu_forward_calls', 'optimizer_steps']) or
            (refusal_folder/'started.json').exists() or files(job) != protected):
        raise ValueError('Full finite phase budget allowed another computation')
    durable_json(folder/'acceptance_receipt.json', dict(at=now(), status='passed', verifier_pid=os.getpid(),
        actual_compute_pid=phase['pid'], repeated_cli_pid=reuse_pid, refusal_cli_pid=refusal_pid,
        actual_environment_steps=2048, actual_gpu_forward_calls=512, optimizer_steps=0,
        requested_chunks=20, actual_chunks=1, original_boundary_checks=phase['boundary_checks'],
        complete_oracle_trace_worker_rng_views_exact=True, original_repeat_receipt_unchanged=True,
        repeated_attempt_additional_environment_steps=0, second_attempt_environment_steps=0,
        protected_job_files=protected, last_manifest_sha256=last_sha,
        eligible_for_future_bounded_phase_launch=True, policy_promotion='none',
        scope='Bounded phase production-path acceptance only, not a new full quality evaluation, speed gate or policy improvement. '
              'This 256-round job is complete as an acceptance; do not extend its declared phase budget.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
