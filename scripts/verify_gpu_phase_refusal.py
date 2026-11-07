"""Real CLI finite-budget refusals and receipt reuse, without advancing old jobs."""
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
    if (folder/'refusal_receipt.json').exists():
        raise FileExistsError('Refusal acceptance already recorded')
    design = json.loads((folder/'design.json').read_text())
    cases = []
    for case in design['refusal_cases']:
        attempt, job = Path(case['attempt']), Path(case['job_folder'])
        if (attempt/'phase_receipt.json').exists():
            raise FileExistsError('Inspect prior attempt before starting acceptance')
        protected = fingerprint(job)
        results = []
        for _ in range(2):
            child = subprocess.Popen([sys.executable, str(ROOT/'scripts/run_gpu_evaluation_phase.py'),
                '--attempt-dir', str(attempt)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = child.communicate()
            if child.returncode:
                raise RuntimeError('Expected budget refusal failed: '+str(child.pid)+' '+stderr[-1000:])
            actual = json.loads(stdout)
            if (actual['status'] != 'preserved' or actual['plan']['chunks'] != 0 or
                    any(actual[key] != 0 for key in ['new_environment_steps', 'gpu_forward_calls', 'optimizer_steps']) or
                    actual['initial_state']['completed'] or actual['initial_state']['can_resume'] or
                    (attempt/'started.json').exists() or fingerprint(job) != protected):
                raise ValueError('Budget refusal advanced or altered the original cache')
            results.append(dict(cli_pid=child.pid, result=actual))
        if results[0]['result'].get('reused_receipt') or results[1]['result'].get('reused_receipt') is not True:
            raise ValueError('Repeated attempt did not reuse its receipt')
        cases.append(dict(job_folder=str(job), protected_input_files=protected, invocations=results))
    durable_json(folder/'refusal_receipt.json', dict(at=now(), status='passed', verifier_pid=os.getpid(),
        cases=cases, new_environment_steps=0, gpu_forward_calls=0, optimizer_steps=0,
        original_job_inputs_unchanged=True, repeated_attempts_reused_receipts=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
