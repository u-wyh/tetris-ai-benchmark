"""Measure V1/V2 decision components on the same public board states."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter_ns
from unittest.mock import patch

from training.evaluation.traditional import ROOT, atomic, now
from training.tetris_core import TetrisCore
from training.traditional import v1
from training.traditional.v2 import V2, V2Config

SAMPLE_STEPS = (0, 5, 20, 50, 100, 200, 400, 800, 1200, 1600, 2000, 3000, 4000)


def states(reference):
    row = json.loads(Path(reference).read_text())
    core = TetrisCore(row['seed'])
    result = []
    for index,action in enumerate(row['actions']):
        if index in SAMPLE_STEPS:
            result.append((index,core.get_public_observation()))
        core.step(action)
    return result


def measure(reference, repeats=3, beam_width=8):
    source_states = states(reference)
    rows = []
    original_legal = v1.get_legal_placements
    original_simulate = v1.simulate
    original_value = v1.board_value
    for index,observation in source_states:
        for repetition in range(repeats):
            start = perf_counter_ns()
            placements = original_legal(observation)
            root_ms = (perf_counter_ns()-start)/1e6
            counters = dict(next_legal_ms=0.,simulation_ms=0.,features_ms=0.)
            def wrap(name,fn):
                def timed(*args,**kwargs):
                    begin = perf_counter_ns()
                    result = fn(*args,**kwargs)
                    counters[name] += (perf_counter_ns()-begin)/1e6
                    return result
                return timed
            start = perf_counter_ns()
            with patch.object(v1,'get_legal_placements',wrap('next_legal_ms',original_legal)), \
                 patch.object(v1,'simulate',wrap('simulation_ms',original_simulate)), \
                 patch.object(v1,'board_value',wrap('features_ms',original_value)):
                a1 = v1.V1Adapter(0).choose(observation,placements)
            v1_ms = (perf_counter_ns()-start)/1e6
            policy = V2(V2Config(beam_width=beam_width))
            start = perf_counter_ns()
            a2 = policy.choose(observation,placements)
            v2_ms = (perf_counter_ns()-start)/1e6
            rows.append(dict(step=index,repetition=repetition,root_legal_ms=root_ms,
                             v1_action=a1,v2_action=a2,v1_policy_ms=v1_ms,v2_policy_ms=v2_ms,
                             v1_breakdown=counters,v2_breakdown=policy.last_timing_ms))
    return rows


def summarize(rows):
    result = dict(states=len(set(row['step'] for row in rows)),samples=len(rows))
    for key in ('root_legal_ms','v1_policy_ms','v2_policy_ms'):
        result[f'mean_{key}'] = statistics.mean(row[key] for row in rows)
    for prefix, key in (('v1','v1_breakdown'),('v2','v2_breakdown')):
        for field in rows[0][key]:
            result[f'mean_{prefix}_{field}'] = statistics.mean(row[key][field] for row in rows)
    result['mean_v1_total_ms'] = result['mean_root_legal_ms']+result['mean_v1_policy_ms']
    result['mean_v2_total_ms'] = result['mean_root_legal_ms']+result['mean_v2_policy_ms']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--beam-width',type=int,default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    data = dict(reference=str(args.reference),repeats=args.repeats,beam_width=args.beam_width,
                created_at=now(),platform=platform.platform(),python=platform.python_version(),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    data['samples'] = measure(args.reference,args.repeats,args.beam_width)
    data['summary'] = summarize(data['samples'])
    atomic(args.output/'profile.json',data)
    lines = ['# Matched-state V1/V2 CPU decision profile','',
             f"{data['summary']['states']} public states × {args.repeats} repeats, same root BFS candidates; V2 beam width {args.beam_width}.",
             'V1 Python wrapper profiling adds small overhead; both policies ran sequentially under the same host load.',
             'Root legal BFS is counted once for both. V1 future BFS is inside next_legal_ms; V2 future BFS is future_legal. Timings exclude Core.step and file I/O.','',
             '| Metric | Mean ms |','|---|---:|']
    lines += [f'| {key} | {value:.3f} |' for key,value in data['summary'].items() if key.startswith('mean_')]
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(args.output/'report.md')


if __name__=='__main__':
    main()
