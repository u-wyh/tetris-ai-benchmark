"""Compare audited complete CUDA8 captures using an unchanged development gate."""
import argparse
import json
from pathlib import Path

from compare_development_evaluations import compare
from run_longer_development_screen import extension_eligible
from own_policy_capture import sha
from resource_guard import now
from training.evaluation.evaluate_resumable import durable_json


def matching_protocol(left, right):
    for key in ('training_sources', 'inference_sources', 'evaluation_runtime',
                'native_feature_runtime', 'seeds', 'max_pieces', 'chunk_rounds',
                'device', 'n_envs', 'optimizer_steps'):
        if left[key] != right[key]:
            raise ValueError('Different paired GPU protocol: ' + key)
    if (left['device'] != 'cuda' or left['n_envs'] != 8 or left['optimizer_steps'] != 0 or
            left['seeds'] != list(range(100000, 100008)) or left['max_pieces'] != 20000):
        raise ValueError('Complete development GPU8 protocol differs')
    if (left['models']['candidate'] != right['models']['shadow'] or
            left['models']['shadow'] != right['models']['candidate'] or
            left['models']['candidate']['sha256'] == right['models']['candidate']['sha256']):
        raise ValueError('Paired models must exchange controller and observer roles')


def result(job, design):
    summary = json.loads((job/'capture_summary.json').read_text())
    audit = json.loads((job/'trajectory_audit.json').read_text())
    identity = summary['identity']
    expected = dict(design_sha256=sha(job/'design.json'), chunk_rounds=256,
        ordered_lanes=design['seeds'], device='cuda', precision='highest', tf32=False,
        batching='active lanes only, in original lane order; no auto-reset')
    if (summary['status'] != 'passed' or audit['status'] != 'passed' or
            identity != expected or audit['identity'] != expected or
            audit['complete_worker_trace_endpoints_exact'] is not True or
            audit['global_and_worker_reset_sampling_rng_unchanged'] is not True or
            audit['optimizer_steps'] != 0 or summary['optimizer_steps'] != 0 or
            audit['actual_unique_environment_steps'] != summary['environment_steps'] or
            audit['last_manifest_sha256'] != sha(job/f"chunk_{audit['chunks']-1:04d}"/'manifest.json')):
        raise ValueError('Missing or different complete trajectory audit')
    model = design['models']['candidate']
    if sha(model['path']) != model['sha256']:
        raise ValueError('Evaluated checkpoint changed')
    endpoints = summary['endpoints']
    if (len(endpoints) != 8 or [r['seed'] for r in endpoints] != design['seeds'] or
            [r['lane'] for r in endpoints] != list(range(8)) or
            sum(r['steps'] for r in endpoints) != summary['environment_steps'] or
            any(type(r['game_over']) is not bool or type(r['cap_censored']) is not bool or
                r['game_over'] == r['cap_censored'] or
                r['cap_censored'] != (r['steps'] == 20000 and not r['game_over']) for r in endpoints)):
        raise ValueError('Incomplete, unordered or ambiguous episode endpoints')
    return dict(model_sha256=model['sha256'], model_checkpoint=model['path'],
        actual_source_counts=model['counts'], committed_steps=model['counts'][0],
        evaluation_source_sha256=design['inference_sources'],
        candidate_feature_backend='bitboard_native', **design['evaluation_runtime'],
        seeds=design['seeds'], max_pieces=design['max_pieces'], num_seeds=8,
        device='cuda', evaluation_workers=8, precision='highest', tf32=False,
        per_seed=[dict(seed=r['seed'], pieces_survived=r['steps'], score=r['score'],
                      lines=r['lines'], game_over=r['game_over'], survived_cap=r['cap_censored'])
                  for r in endpoints],
        provenance='Current sources identify evaluation only; original checkpoint training sources retained.')


def run(folder):
    protocol = json.loads((folder/'protocol.json').read_text())
    for path, digest in protocol['frozen_files'].items():
        if sha(path) != digest:
            raise ValueError('Frozen quality input changed: ' + path)
    gate_source = Path(protocol['gate_source'])
    gate = json.loads(gate_source.read_text())['extension_gate']
    if sha(gate_source) != protocol['gate_source_sha256'] or gate != protocol['extension_gate']:
        raise ValueError('Original development gate changed')
    jobs = {key: Path(path) for key, path in protocol['jobs'].items()}
    designs = {key: json.loads((job/'design.json').read_text()) for key, job in jobs.items()}
    matching_protocol(designs['baseline'], designs['candidate'])
    outputs = {key: result(job, designs[key])
               for key, job in jobs.items()}
    comparison = compare(outputs['baseline'], outputs['candidate'])
    failures = {key: sum(row['game_over'] for row in data['per_seed']) for key, data in outputs.items()}
    if not comparison['comparable']:
        raise ValueError(comparison['reason'])
    passed = extension_eligible(comparison, failures, gate)
    for key, data in outputs.items():
        durable_json(folder/(key+'.json'), data)
    durable_json(folder/'quality_comparison.json', dict(at=now(), status='completed',
        comparison=comparison, failures=failures, extension_gate=gate,
        development_gate_passed=passed, eligible_for_training_extension=False,
        independent_development_validation_required=True, optimizer_steps=0,
        policy_promotion='none',
        interpretation='Matched GPU8 development results only. Passing requires additional independent '
                       'development seeds before further training; no model promotion.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
