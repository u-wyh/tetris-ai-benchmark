"""Bounded new-seed functional CUDA oracle versus a real two-process restart."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture_gpu_evaluation import prepare, validate_snapshot
from own_policy_capture import sha, read_chain
from restore_own_death_contexts import load_verified_chunk
from verify_feature_backend_benchmark import identical
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def run(folder):
    if (folder/'verification_started.json').exists():
        raise FileExistsError('Inspect existing acceptance before resuming; never repeat its computed work')
    jobs = [folder/'oracle', folder/'parallel']
    prepared = [prepare(job) for job in jobs]
    if any(count != 0 for _, _, (_, _, count) in prepared):
        raise FileExistsError('Functional acceptance requires fresh jobs; preserve existing commits')
    oracle_design, oracle_identity, _ = prepared[0]
    parallel_design, parallel_identity, _ = prepared[1]
    if (oracle_design['dispatch_mode'] != 'serial_oracle' or parallel_design['dispatch_mode'] != 'parallel' or
            any(oracle_design[key] != parallel_design[key] for key in
                ['seeds', 'models', 'evaluation_sources', 'training_sources', 'inference_sources',
                 'evaluation_runtime', 'native_feature_runtime'])):
        raise ValueError('Oracle and parallel must share the complete ordered numerical protocol')
    durable_json(folder/'verification_started.json', dict(at=now(), pid=os.getpid(),
        seeds=oracle_design['seeds'], environment_step_limit=8192, optimizer_steps=0,
        scope='512 rounds per arm; functional identity/restart only, no speed or quality qualification'))
    def execute(job, chunks):
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts/capture_gpu_evaluation.py'),
            '--run-dir', str(job), '--max-chunks', str(chunks)])
        if child.wait():
            raise RuntimeError('GPU evaluation child failed; preserve work: '+str(child.pid))
        return json.loads((job/f'process_{child.pid}.json').read_text())
    oracle = execute(jobs[0], 2)
    first = execute(jobs[1], 1)
    protected = {str(path): sha(path) for chunk in jobs[1].glob('chunk_[0-9][0-9][0-9][0-9]')
                 for path in chunk.iterdir() if path.is_file()}
    if first['computed_chunks'] != [1] or first['reused_committed_chunks'] != 0:
        raise ValueError('First process did not compute only the first new chunk')
    second = execute(jobs[1], 1)
    if (first['pid'] == second['pid'] or second['computed_chunks'] != [2] or
            second['reused_committed_chunks'] != 2 or
            any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('Real restart changed or repeated the committed prefix')
    actual_steps = sum(p['environment_steps_this_process'] for p in [oracle, first, second])
    if not 0 < actual_steps <= 8192:
        raise ValueError('Actual finite functional work exceeded its budget')
    chains = [read_chain(job, identity) for job, (_, identity, _) in zip(jobs, prepared)]
    if (any(count != 3 or snapshot['round'] != 512 for snapshot, _, count in chains) or
            sum(snapshot['environment_steps'] for snapshot, _, _ in chains) != actual_steps):
        raise ValueError('Functional scope not reached or unique work accounting differs; do not change seeds')
    for number in range(3):
        loaded = []
        for job, (design, identity, _) in zip(jobs, prepared):
            chunk = job/f'chunk_{number:04d}'
            state, rows, manifest = load_verified_chunk(chunk, sha(chunk/'manifest.json'), identity)
            validate_snapshot(state, design)
            if rows and any(int(seed) != design['seeds'][int(lane)]
                            for lane, seed in zip(rows['lane'], rows['episode_seed'])):
                raise ValueError('Trace seed order differs from design, including after model loading')
            if (manifest['audit'].get('optimizer_steps') != 0 or
                    manifest['audit'].get('policy_adam_counts_unchanged') is not True or
                    manifest['audit'].get('global_sampling_rng_unchanged') is not True):
                raise ValueError('Frozen model/optimizer/RNG audit failed')
            loaded.append((state, rows))
        if not identical(loaded[0], loaded[1]):
            raise ValueError('Oracle and real restart trace or complete worker/RNG/views differ')
    durable_json(folder/'restart_receipt.json', dict(at=now(), status='passed', verifier_pid=os.getpid(),
        oracle_process=oracle, first_process=first, second_process=second,
        actual_environment_steps=actual_steps,
        actual_candidate_forward_calls=sum(p['candidate_forward_calls_this_process'] for p in [oracle, first, second]),
        actual_shadow_forward_calls=sum(p['shadow_forward_calls_this_process'] for p in [oracle, first, second]),
        optimizer_steps=0, ordered_seeds=oracle_design['seeds'],
        initial_and_first_chunk_files_unchanged=True, protected_files=protected,
        complete_oracle_trace_worker_rng_views_exact=True,
        last_manifest_sha256=[last for _, last, _ in chains],
        eligible_for_future_ordered_gpu_evaluation=True, policy_promotion='none',
        scope='Normal process-boundary restart, not OS reboot or arbitrary power-loss testing. '
              '512-round functional acceptance is complete; no extension of these jobs to 20k, '
              'candidate qualification, held-out confirmation or new speed claim.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
