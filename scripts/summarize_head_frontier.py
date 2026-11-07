"""Read frozen scalar geometry and report all training-defined event boundaries."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def mean_episode_derivative(pairs):
    seeds = sorted({p['seed'] for p in pairs})
    means = []
    for seed in seeds:
        rows = [p for p in pairs if p['seed'] == seed]
        values = []
        for row in rows:
            m = row['margin']
            weight = math.exp(-m)/(1+math.exp(-m)) if m >= 0 else 1/(1+math.exp(m))
            values.append(-weight*row['slope'])
        means.append(sum(values)/len(values))
    return sum(means)/len(means)


def frontier(result):
    pairs, normals = result['preference_pairs'], result['normal_states']
    beneficial = [p for p in pairs if p['label'] == 'beneficial']
    train = [p for p in beneficial if p['split'] == 'train']
    validation = [p for p in beneficial if p['split'] == 'validation']
    if (result['status'] != 'passed' or len(normals) != 632 or len(train) != 146 or len(validation) != 31 or
            result['source_policy_adam_counts_rng_unchanged'] is not True or
            result['eligible_for_training'] is not False or
            any(p['margin'] >= 0 for p in pairs) or any(p['numerical_ties'] for p in normals)):
        raise ValueError('Cached fixed geometry scope differs')
    derivatives = dict(train=mean_episode_derivative(train), validation=mean_episode_derivative(validation))
    if not math.isclose(derivatives['train'], -result['direction_norm']**2, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Episode-weighted derivative does not match original direction norm')
    # Validation outcomes/thresholds never choose the event grid. No best step is selected.
    events = sorted({0., *[float(p['crossing']) for p in train],
                     *[float(p['crossing']) for p in normals if p['split'] == 'train']}- {math.inf})
    rows = []
    for boundary in events:
        row = dict(training_defined_boundary=boundary, splits={})
        for split in ('train', 'validation'):
            normal = [p for p in normals if p['split'] == split]
            selected = [p for p in pairs if p['split'] == split]
            row['splits'][split] = dict(normal_states=len(normal),
                reference_actions_changed=sum(float(p['crossing']) <= boundary for p in normal),
                conservative_changes=sum(float(p['conservative_crossing']) <= boundary for p in normal),
                beneficial_pairs=len([p for p in selected if p['label'] == 'beneficial']),
                newly_outranked_by_label={label:sum(p['label'] == label and p['slope'] > 0 and
                    float(p['crossing']) <= boundary for p in selected)
                    for label in ('beneficial', 'harmful', 'unsupported_within_window')})
        rows.append(row)
    first_repair = min(float(p['crossing']) for p in train)
    return dict(status='passed', episode_equal_loss_derivative=derivatives,
        direction_norm_square=result['direction_norm']**2,
        first_training_repair_boundary=first_repair,
        first_training_repair_row=next(r for r in rows if r['training_defined_boundary'] == first_repair),
        all_training_boundaries=rows, event_count=len(rows),
        convention='Analytic right limit t+ at every finite train normal/pair crossing, plus0; no finite step or candidate policy selected. Validation only evaluated on train grid. Pairwise outranking is not globalTop1 or quality.',
        work=dict(new_environment_steps=0, gpu_forward_calls=0, optimizer_steps=0, parameter_updates=0),
        eligible_for_training=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = args.input.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.sha256:
        raise ValueError('Cached geometry checksum differs')
    if args.output.exists():
        raise FileExistsError('Frontier output already exists; inspect instead of overwriting')
    result = frontier(json.loads(data))
    result['source_geometry_sha256'] = args.sha256
    with args.output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
