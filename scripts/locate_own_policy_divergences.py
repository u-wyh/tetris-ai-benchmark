"""Select the last observed policy disagreement before EACH captured true death."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from own_policy_capture import read_chain, sha
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def latest_disagreements(chunks, deaths):
    """Steps are 1-based actions; round/preceding state use step-1."""
    selected = {}
    for number, data in chunks:
        for row in range(len(data['lane'])):
            seed = int(data['episode_seed'][row])
            if seed not in deaths or data['action'][row] == data['shadow_action'][row]:
                continue
            step = int(data['pieces'][row])
            if step > deaths[seed] or int(data['round'][row]) != step-1:
                raise ValueError('Disagreement outside its one-episode death trajectory')
            if seed in selected and step <= selected[seed]['step']:
                raise ValueError('Disagreement rows are not in strictly increasing episode order')
            target = dict(key=f'seed_{seed}_step_{step}', seed=seed, lane=int(data['lane'][row]),
                step=step, round=step-1, source_chunk=number, previous_chunk=number-1,
                death_step=deaths[seed], actions_to_observed_death_including_selected=deaths[seed]-step+1,
                candidate_action=int(data['action'][row]), shadow_action=int(data['shadow_action'][row]))
            for key in ['candidate_features', 'shadow_features']:
                target[key] = data[key][row].tolist()
            for key in ['candidate_probability', 'shadow_probability', 'shadow_probability_of_executed']:
                target[key] = float(data[key][row])
            target['observation_digest'] = dict(observation=str(data['observation_sha256'][row]),
                                               mask=str(data['mask_sha256'][row]))
            selected[seed] = target
    if set(selected) != set(deaths):
        raise ValueError('A true death has no preceding observed disagreement; do not silently drop it')
    return [selected[seed] for seed in sorted(deaths)]


def run(folder):
    design = json.loads((folder/'design.json').read_text())
    for path, expected in design['frozen_files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen divergence input changed: '+path)
    capture = Path(design['capture_dir'])
    summary = json.loads((capture/'capture_summary.json').read_text())
    audit = json.loads((capture/'trajectory_audit.json').read_text())
    if summary['status'] != 'passed' or audit['status'] != 'passed':
        raise ValueError('Complete capture and trajectory audit required')
    snapshot, last_hash, count = read_chain(capture, summary['identity'])
    if last_hash != audit['last_manifest_sha256'] or any(w['active'] for w in snapshot['workers']):
        raise ValueError('Capture chain is not the audited complete chain')
    deaths = {endpoint['seed']: endpoint['steps'] for endpoint in summary['endpoints'] if endpoint['game_over']}
    if sorted(deaths) != design['expected_death_seeds']:
        raise ValueError('All observed true deaths must be included')
    def chunks():
        for number in range(1, count):
            with np.load(capture/f'chunk_{number:04d}/rows.npz', allow_pickle=False) as archive:
                yield number, {key: archive[key] for key in archive.files}
    targets = latest_disagreements(chunks(), deaths)
    result = dict(at=now(), status='passed', capture_dir=str(capture), capture_identity=summary['identity'],
        source_last_manifest_sha256=last_hash, capture_audit_sha256=sha(capture/'trajectory_audit.json'),
        selection_rule='Last actual candidate action != original shadow action in each true-death own trajectory; all deaths included',
        targets=targets, environment_steps=0, policy_forward_calls=0, optimizer_steps=0,
        interpretation='Descriptive selection for full-state recovery; no intervention or causal/survival claim.')
    path = folder/'targets.json'
    if path.exists():
        old = json.loads(path.read_text())
        if {k:v for k,v in old.items() if k != 'at'} != {k:v for k,v in result.items() if k != 'at'}:
            raise ValueError('Committed divergence target selection differs')
    else:
        durable_json(path, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
