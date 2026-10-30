"""Read-only algebraic mechanism analysis of already frozen ablation artifacts.

No controller inference, Core stepping, parameter selection or new trajectories.
The reconstructed actions are conditional on Full's visited states, not paired
game outcome estimates. Functional artifacts provide an independent identity check.
"""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/lookahead_ablation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    envelope = json.loads(path.read_text())
    payload = json.dumps(envelope['data'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(payload).hexdigest() != envelope['sha256']:
        raise RuntimeError(f'JSON SHA differs: {path}')
    return envelope['data']


def action(trace, two=False, shallow=False):
    roots = [r for r in trace['alternatives'] if not two or r['action'] in (trace['student_action'], trace['bc_action'])]
    def value(r):
        if not shallow:
            return r['value']
        death = 1000000 if len(r['path']) == 1 and r['terminal'] == 'over' else 0
        return 1.25 * r['immediate_geometry'] - 2 * r['immediate_reward'] + death
    first, best = roots[0], min(roots, key=value)
    return best['action'] if value(best) + 12 < value(first) else first['action']


def main():
    functional = read(RUN / 'functional.json')
    checks = 0
    for row in functional['rows']:
        trace = row['traces']['E']
        for variant, two, shallow in [('B', True, True), ('C', True, False), ('D', False, True), ('E', False, False)]:
            if action(trace, two, shallow) != row['actions'][variant]:
                raise RuntimeError(f'Algebra differs from actual functional {row["seed"]}/{row["piece"]}/{variant}')
            checks += 1
    output = dict(functional_action_checks=checks,
        scope='Actions reconstructed on Full-visited states only; no efficacy or survival inference from action differences.',
        formula='depth1 = 1.25*immediate_geometry - 2*immediate_reward + immediate_true_death_penalty; same margin12 and root tie order',
        source_sha256=sha(Path(__file__)), stages={})
    protocol = read(RUN / 'protocol.json')
    fields = protocol['cost_fields']
    for stage in ('short', 'long'):
        counts = Counter(); sources = []; examples = []
        for path in sorted((RUN / 'evaluation' / stage / 'E').glob('seed_*.json')):
            meta = read(path); artifact = path.with_suffix('.npz')
            if sha(artifact) != meta['artifact_sha256']:
                raise RuntimeError(f'Artifact SHA differs: {artifact}')
            sources.append(dict(seed=meta['seed'], result_sha256=sha(path), artifact_sha256=meta['artifact_sha256']))
            with np.load(artifact) as data:
                codes, roots, costs = data['codes'], data['root_codes'], data['costs']
                ids = roots[:, :, 0]; valid = ids >= 0
                deep = costs[:, :, fields.index('value')]
                shallow = 1.25 * costs[:, :, fields.index('immediate_geometry')] - 2 * costs[:, :, fields.index('immediate_reward')]
                shallow += 1000000 * ((roots[:, :, 1] == 1) & (roots[:, :, 3] == 2))
                two = valid & ((ids == codes[:, 0, None]) | (ids == codes[:, 1, None]))
                actions = {'A': codes[:, 0]}
                for variant, values, allowed in [('B', shallow, two), ('C', deep, two), ('D', shallow, valid), ('E', deep, valid)]:
                    values = np.where(allowed, values, np.inf)
                    best = values.argmin(axis=1); index = np.arange(len(best))
                    actions[variant] = np.where(values[index, best] + 12 < values[:, 0], ids[index, best], codes[:, 0])
                if not np.array_equal(actions['E'], codes[:, 4]):
                    raise RuntimeError(f'Full reconstruction mismatch: {path}')
                counts['decisions'] += len(codes)
                counts['unequal_root_horizons'] += int(np.sum(np.max(np.where(valid, roots[:, :, 1], -1), axis=1) != np.min(np.where(valid, roots[:, :, 1], 99), axis=1)))
                for left, right in [('A', 'B'), ('B', 'C'), ('D', 'E'), ('C', 'E'), ('B', 'D'), ('A', 'E')]:
                    counts[f'{right}_differs_{left}'] += int(np.sum(actions[left] != actions[right]))
                chosen = np.argmax(ids == codes[:, 4, None], axis=1); index = np.arange(len(chosen))
                override = actions['E'] != actions['A']; counts['overrides'] += int(override.sum())
                # Additive terms explain the stored frozen objective, not real safety.
                for label, component in [('terminal_geometry', costs[:, :, fields.index('final_geometry')]), ('path_max_quarter', .25 * costs[:, :, fields.index('max_geometry')]), ('formal_reward_term', costs[:, :, fields.index('score_contribution')]), ('known_death_penalty', costs[:, :, fields.index('death_penalty')])]:
                    saving = component[:, 0] - component[index, chosen]
                    counts[f'override_{label}_improved'] += int(np.sum(override & (saving > 1e-9)))
                    counts[f'override_{label}_worsened'] += int(np.sum(override & (saving < -1e-9)))
                    counts[f'override_{label}_saving_sum'] += float(saving[override].sum())
            # Fixed chronological examples: first eight Full overrides, no outcome filter.
            for sample in meta['traces']:
                trace = sample['trace']
                if trace['override'] and len(examples) < 8:
                    examples.append(dict(seed=meta['seed'], piece=sample['piece'], public=sample['public'],
                        reconstructed_actions={v: (trace['student_action'] if v == 'A' else action(trace, v in ('B', 'C'), v in ('B', 'D'))) for v in 'ABCDE'}, trace=trace))
        if sources:
            output['stages'][stage] = dict(completed_Full_episodes=len(sources), counts=dict(counts), sources=sources, first_chronological_override_examples=examples)
    target = RUN / 'derived_mechanism.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, target)
    print(json.dumps({k: (v if k != 'stages' else {stage: {'episodes': s['completed_Full_episodes'], 'counts': s['counts']} for stage, s in v.items()}) for k, v in output.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
