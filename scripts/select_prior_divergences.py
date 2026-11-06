"""Predeclared recent disagreements before a previously diagnosed commuting pair."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from locate_own_policy_divergences import latest_disagreements
from own_policy_capture import read_chain, sha
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def prior_disagreements(chunks, seed, before_step, count, death_step):
    if not 1 <= count <= 8 or not 1 < before_step <= death_step:
        raise ValueError('Invalid predeclared prior disagreement scope')
    recent = deque(maxlen=count); available = 0; previous_step = 0
    for number, data in chunks:
        selected = np.flatnonzero((data['episode_seed'] == seed) & (data['pieces'] < before_step) &
                                  (data['action'] != data['shadow_action']))
        for row in selected:
            # Reuse the verified one-based target schema without changing its frozen producer.
            singleton = {key:value[row:row+1] for key,value in data.items()}
            target = latest_disagreements([(number,singleton)],{seed:death_step})[0]
            if target['step'] <= previous_step:
                raise ValueError('Prior disagreement input is not strictly chronological')
            previous_step = target['step']; available += 1; recent.append(target)
    if len(recent) != count:
        raise ValueError('Not enough prior disagreements; do not silently reduce the frozen sample count')
    return list(recent), available


def run(folder):
    design = json.loads((folder/'design.json').read_text())
    if (design['seed'],design['before_step'],design['count'],design['death_step']) != (100005,6921,8,6942):
        raise ValueError('Frozen all-eight prior-disagreement selection protocol differs')
    for path, expected in design['frozen_files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen prior disagreement input changed: '+path)
    capture = Path(design['capture_dir'])
    summary = json.loads((capture/'capture_summary.json').read_text())
    audit = json.loads((capture/'trajectory_audit.json').read_text())
    if summary['status'] != 'passed' or audit['status'] != 'passed':
        raise ValueError('Complete audited own-policy capture required')
    _, last_hash, chunks_count = read_chain(capture,summary['identity'])
    if last_hash != audit['last_manifest_sha256']:
        raise ValueError('Captured source differs from the audited chain')
    def chunks():
        for number in range(1,chunks_count):
            with np.load(capture/f'chunk_{number:04d}/rows.npz',allow_pickle=False) as archive:
                yield number,{key:archive[key] for key in archive.files}
    targets, available = prior_disagreements(chunks(),design['seed'],design['before_step'],
                                            design['count'],design['death_step'])
    result = dict(at=now(),status='passed',capture_dir=str(capture),capture_identity=summary['identity'],
        source_last_manifest_sha256=last_hash,seed=design['seed'],before_step_exclusive=design['before_step'],
        selected_count=len(targets),available_prior_disagreements=available,targets=targets,
        source_chunks=sorted({target['source_chunk'] for target in targets}),
        maximum_actions_to_death=max(target['actions_to_observed_death_including_selected'] for target in targets),
        selection_rule='Most recent8 actual disagreements strictly before6921 in seed100005, chronological order; no feature/outcome filtering',
        environment_steps=0,policy_forward_calls=0,optimizer_steps=0,
        interpretation='Descriptive target selection only; no restored states, interventions or causal claims.')
    path = folder/'targets.json'
    if path.exists():
        old = json.loads(path.read_text())
        if {k:v for k,v in old.items() if k != 'at'} != {k:v for k,v in result.items() if k != 'at'}:
            raise ValueError('Committed prior target selection differs')
    else:
        durable_json(path,result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    run(parser.parse_args().run_dir.resolve())
