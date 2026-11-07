"""Check the real inspection CLI on frozen caches without advancing their jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inspect_gpu_evaluation import fingerprint
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def run(folder):
    if (folder/'inspection_receipt.json').exists():
        raise FileExistsError('Read-only acceptance already recorded; preserve it')
    design = json.loads((folder/'design.json').read_text())
    results = []
    for case in design['inspection_cases']:
        job = Path(case['folder']).resolve()
        before = fingerprint(job)
        command = [sys.executable, str(ROOT/'scripts/inspect_gpu_evaluation.py'), '--run-dir', str(job)]
        if case.get('upstream_summary'):
            command.extend(['--upstream-summary', case['upstream_summary']])
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = child.communicate()
        if child.returncode != case['expected_exit_code']:
            raise RuntimeError('Inspection CLI failed: '+str(child.pid)+' '+stderr[-1000:])
        actual = json.loads(stdout)
        if (any(actual.get(key) != value for key, value in case['expected'].items()) or
                any(actual[key] != 0 for key in ['new_environment_steps', 'gpu_forward_calls', 'optimizer_steps']) or
                before != fingerprint(job) or actual['protected_input_files'] != before):
            raise ValueError('Actual status, finite budget or read-only input invariant differs')
        results.append(dict(pid=child.pid, result=actual))
    durable_json(folder/'inspection_receipt.json', dict(at=now(), status='passed', pid=os.getpid(),
        cases=results, new_environment_steps=0, gpu_forward_calls=0, optimizer_steps=0,
        original_job_inputs_unchanged=True, interpretation='Read-only status acceptance, not resumed evaluation or policy qualification'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
