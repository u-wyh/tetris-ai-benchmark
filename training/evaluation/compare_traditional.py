"""Pair completed V1/V2 experiments on identical seeds and piece caps."""
import argparse
import json
from pathlib import Path
import random
import statistics
from math import comb

from training.evaluation.traditional import atomic, now


def _read(folder):
    folder = Path(folder)
    config = json.loads((folder/'config.json').read_text())
    status = json.loads((folder/'status.json').read_text())
    if status['state'] != 'complete':
        raise ValueError(f"Incomplete experiment: {folder}")
    rows = [json.loads((folder/f'seed_{seed}.json').read_text()) for seed in config['seeds']]
    if [row['seed'] for row in rows] != config['seeds']:
        raise ValueError(f"Seed records do not match config: {folder}")
    return config, rows


def _bootstrap(differences, iterations=10000):
    generator = random.Random(20261008)
    n = len(differences)
    samples = sorted(sum(generator.choice(differences) for _ in range(n))/n for _ in range(iterations))
    return [samples[int(iterations*.025)],samples[int(iterations*.975)]]


def _sign_test(wins, losses):
    n = wins+losses
    if not n:
        return 1.0
    smaller = min(wins,losses)
    return min(1.0,2*sum(comb(n,k) for k in range(smaller+1))/2**n)


def compare(v1_dir, v2_dir):
    c1, rows1 = _read(v1_dir)
    c2, rows2 = _read(v2_dir)
    if c1['seeds'] != c2['seeds'] or c1['max_pieces'] != c2['max_pieces']:
        raise ValueError("Paired comparison needs identical seeds and cap")
    if c1['strategy'] != 'v1-adapter-1' or c2['strategy'] != 'v2-1':
        raise ValueError("Unexpected strategy version")
    pairs = []
    for a,b in zip(rows1,rows2):
        if a['seed'] != b['seed']:
            raise ValueError("Unpaired seed records")
        pairs.append(dict(seed=a['seed'],v1_pieces=a['pieces_survived'],v2_pieces=b['pieces_survived'],
                          pieces_difference=b['pieces_survived']-a['pieces_survived'],
                          v1_lines=a['lines'],v2_lines=b['lines'],lines_difference=b['lines']-a['lines'],
                          v1_score=a['score'],v2_score=b['score'],score_difference=b['score']-a['score'],
                          v1_game_over=a['game_over'],v2_game_over=b['game_over'],
                          v1_truncated=a['truncated'],v2_truncated=b['truncated']))
    summary = {}
    for label in ('pieces','lines','score'):
        differences = [row[f'{label}_difference'] for row in pairs]
        summary[f'mean_{label}_difference'] = statistics.mean(differences)
        summary[f'median_{label}_difference'] = statistics.median(differences)
        summary[f'mean_{label}_difference_bootstrap_95ci'] = _bootstrap(differences)
    summary['v1_cap_rate'] = statistics.mean(p['v1_truncated'] for p in pairs)
    summary['v2_cap_rate'] = statistics.mean(p['v2_truncated'] for p in pairs)
    summary['both_cap_count'] = sum(p['v1_truncated'] and p['v2_truncated'] for p in pairs)
    summary['v1_dead_v2_cap_count'] = sum(p['v1_game_over'] and p['v2_truncated'] for p in pairs)
    summary['v1_cap_v2_dead_count'] = sum(p['v1_truncated'] and p['v2_game_over'] for p in pairs)
    wins = sum(p['pieces_difference'] > 0 for p in pairs)
    losses = sum(p['pieces_difference'] < 0 for p in pairs)
    summary.update(observed_wins=wins,observed_losses=losses,
                   equal_or_both_capped=len(pairs)-wins-losses,
                   sign_test_two_sided_p=_sign_test(wins,losses))
    return dict(v1=str(v1_dir),v2=str(v2_dir),seeds=c1['seeds'],max_pieces=c1['max_pieces'],
                v1_config=c1,v2_config=c2,paired=pairs,summary=summary,created_at=now())


def markdown(result):
    summary = result['summary']
    lines = ["# V1 Adapter / V2 paired comparison", "",
             f"Same {len(result['seeds'])} seeds and {result['max_pieces']} piece cap. Truncation is not death.",
             "Differences are V2 minus V1. Bootstrap resamples seed pairs 10,000 times with fixed analysis seed; intervals describe the cap-limited sample mean, not uncensored lifetime.",
             "The sign test uses observed unequal survival counts; jointly capped pairs remain unresolved.",
             "Official score includes level multipliers and is not a substitute for survival.", "",
             "| Metric | Result |", "|---|---:|"]
    for key,value in summary.items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "| Seed | V1 pieces | V2 pieces | Δ pieces | V1 lines | V2 lines | Δ lines | V1 score | V2 score | Δ score | End V1/V2 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in result['paired']:
        end = ('cap' if row['v1_truncated'] else 'over')+'/'+('cap' if row['v2_truncated'] else 'over')
        lines.append(f"| {row['seed']} | {row['v1_pieces']} | {row['v2_pieces']} | {row['pieces_difference']} | {row['v1_lines']} | {row['v2_lines']} | {row['lines_difference']} | {row['v1_score']} | {row['v2_score']} | {row['score_difference']} | {end} |")
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1-dir',type=Path,required=True)
    parser.add_argument('--v2-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    result = compare(args.v1_dir,args.v2_dir)
    atomic(args.output/'paired.json',result)
    (args.output/'report.md').write_text(markdown(result))
    print(args.output/'report.md')


if __name__ == '__main__':
    main()
