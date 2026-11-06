"""Run a bounded initial capture phase with an actual process boundary."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from own_policy_capture import sha
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def main(folder):
    if (folder/'restart_receipt.json').exists():
        raise RuntimeError('Initial restart verification already completed; continue capture directly')
    if any(folder.glob('chunk_[0-9][0-9][0-9][0-9]')):
        raise RuntimeError('Initial restart verifier requires a fresh capture; inspect existing chunks')
    def execute(chunks):
        command = [sys.executable, str(ROOT/'scripts/capture_own_policy_trajectories.py'),
                   '--run-dir', str(folder), '--max-chunks', str(chunks)]
        child = subprocess.Popen(command)
        if child.wait():
            raise RuntimeError('Bounded capture child failed: '+str(child.pid))
        return child.pid, json.loads((folder/f'process_{child.pid}.json').read_text())
    first_pid, first = execute(1)
    protected = [path for chunk in sorted(folder.glob('chunk_[0-9][0-9][0-9][0-9]'))
                 for path in chunk.iterdir() if path.is_file()]
    before = {str(path): sha(path) for path in protected}
    if first['computed_chunks'] != [1] or first['reused_committed_chunks'] != 0:
        raise ValueError('Initial capture process did not commit exactly one new chunk')
    second_pid, second = execute(19)
    if (first_pid == second_pid or second['reused_committed_chunks'] != 2 or
            not second['computed_chunks'] or second['computed_chunks'][0] != 2 or
            any(sha(path) != expected for path, expected in before.items())):
        raise ValueError('Capture process restart failed to preserve committed prefix')
    durable_json(folder/'restart_receipt.json', dict(at=now(), status='passed',
        first_pid=first_pid, second_pid=second_pid, verifier_pid=os.getpid(),
        initial_and_first_chunk_files_unchanged=True, protected_files=before,
        first_process=first, second_process=second,
        environment_steps_this_phase=first['environment_steps_this_process']+second['environment_steps_this_process'],
        optimizer_steps=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    main(parser.parse_args().run_dir.resolve())
