"""Two real parallel-capture processes, unchanged prefix, exact calibrated trace/state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture_parallel_trajectories import prepare
from own_policy_capture import sha, read_chain
from restore_own_death_contexts import load_verified_chunk
from verify_feature_backend_benchmark import identical
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def run(folder):
    if (folder/'restart_receipt.json').exists() or any(folder.glob('chunk_[0-9][0-9][0-9][0-9]')):
        raise FileExistsError('Recovery acceptance requires a fresh job; preserve existing work')
    design, identity = prepare(folder)
    if len(design['expected_chunks']) != 3:
        raise ValueError('Expected initial plus both calibrated chunks before starting recovery')
    def execute():
        child = subprocess.Popen([sys.executable, str(ROOT/'scripts/capture_parallel_trajectories.py'),
            '--run-dir', str(folder), '--max-chunks', '1'])
        if child.wait():
            raise RuntimeError('Parallel capture child failed: '+str(child.pid))
        return child.pid, json.loads((folder/f'process_{child.pid}.json').read_text())
    first_pid, first = execute()
    protected = {str(path): sha(path) for chunk in folder.glob('chunk_[0-9][0-9][0-9][0-9]')
                 for path in chunk.iterdir() if path.is_file()}
    if first['computed_chunks'] != [1] or first['reused_committed_chunks'] != 0:
        raise ValueError('First process did not compute only the first new chunk')
    second_pid, second = execute()
    if (first_pid == second_pid or second['computed_chunks'] != [2] or
            second['reused_committed_chunks'] != 2 or
            any(sha(path) != digest for path, digest in protected.items()) or
            first['environment_steps_this_process'] != 2048 or second['environment_steps_this_process'] != 2048):
        raise ValueError('Actual parallel restart repeated or changed the committed prefix')
    snapshot, final_sha, count = read_chain(folder, identity)
    if count != 3 or snapshot['round'] != 512 or snapshot['environment_steps'] != 4096:
        raise ValueError('Actual committed parallel recovery work differs')
    for number, spec in enumerate(design['expected_chunks']):
        expected, expected_rows, _ = load_verified_chunk(Path(spec['directory']), spec['manifest_sha256'],
                                                        design['expected_identity'])
        actual, actual_rows, _ = load_verified_chunk(folder/f'chunk_{number:04d}',
            sha(folder/f'chunk_{number:04d}'/'manifest.json'), identity)
        if not identical(actual, expected) or not identical(actual_rows, expected_rows):
            raise ValueError('Real restart differs from calibrated parallel trajectory or complete state')
    durable_json(folder/'restart_receipt.json', dict(at=now(), status='passed', verifier_pid=os.getpid(),
        first_pid=first_pid, second_pid=second_pid, first_process=first, second_process=second,
        actual_environment_steps=4096, actual_candidate_forward_calls=512, actual_shadow_forward_calls=512,
        optimizer_steps=0, initial_and_first_chunk_files_unchanged=True, protected_files=protected,
        complete_source_trace_worker_rng_views_exact=True, last_manifest_sha256=final_sha,
        eligible_for_future_evaluation_capture=True, policy_promotion='none',
        scope='Normal process-boundary capture recovery, not OS reboot or arbitrary power-loss testing. '
              'This 512-round acceptance job is complete; do not fill it to 20k as another quality run.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
