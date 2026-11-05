"""Compare existing feature implementations on identical public training states.

    Run under resource_guard, after other heavy work has finished. This does not
    change the production environment or establish an end-to-end training gain.
"""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import platform
import resource
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from training.candidate_features import candidate_features as reference
    from training.evaluation.fast_candidate_features import candidate_features as accelerated
    from training.tetris_core import TetrisCore
    from training.tetris_core.pieces import make_piece
    from resource_guard import now,operation,write_json

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=30)
    args=parser.parse_args()
    if args.repeats<3:parser.error('Use at least three paired repeats')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    lock=args.output.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.output.exists():raise FileExistsError('Preserve previous benchmark evidence')
    inputs={};fixtures=[]
    for path in args.snapshot:
        payload=path.read_bytes();inputs[str(path.resolve())]=hashlib.sha256(payload).hexdigest()
        states=pickle.loads(payload)  # Own, committed local training snapshots only.
        for index,state in enumerate(states):
            core=state['core']
            if 'get_legal_placements' in core.__dict__:
                raise RuntimeError('Snapshot contains a temporary placement override')
            fixtures.append((f'{path.parent.name}_worker_{index}_{len(fixtures)}',core))
    for kind in ('single_clear','lock_out','partial_lock'):
        core=TetrisCore(42)
        if kind=='single_clear':
            core.current=make_piece('O')
            for x in range(10):
                if x not in (3,4):core.board[19][x]='Z'
        elif kind=='lock_out':
            core.current=make_piece('I');core.current['y']=-2
            for x in range(3,7):core.board[0][x]='Z'
        else:
            core.current=make_piece('O');core.current['x'],core.current['y']=0,-1;core.board[1][0]='Z'
        fixtures.append((kind,core))
    rows=[];prepared=[]
    for name,core in fixtures:
        public=core.get_public_observation();placements=core.get_legal_placements()
        slow=reference(public,placements);fast=accelerated(public,placements)
        if not all(np.array_equal(a,b) and a.dtype==b.dtype for a,b in zip(slow,fast)):
            raise RuntimeError('Features or mask differ: '+name)
        rows.append(dict(name=name,legal_actions=len(placements),
            public_sha256=hashlib.sha256(json.dumps(public,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            features_and_mask_sha256=hashlib.sha256(slow[0].tobytes()+slow[1].tobytes()).hexdigest()))
        prepared.append((public,placements))
    timings=[]
    for repeat in range(args.repeats):
        pair={}
        order=[('reference',reference),('bitboard',accelerated)]
        if repeat%2:order.reverse()
        for name,method in order:
            began=time.process_time_ns()
            for public,placements in prepared:method(public,placements)
            pair[name]=(time.process_time_ns()-began)/1e9
        timings.append(pair)
    ratios=[r['reference']/r['bitboard'] for r in timings]
    sources=['scripts/benchmark_candidate_features.py','training/candidate_features.py',
             'training/evaluation/fast_candidate_features.py']
    sources += ['training/tetris_core/'+name for name in
                ['__init__.py','core.py','actions.py','pieces.py','rng.py','legal_placements.py']]
    result=dict(at=now(),parity='all_features_and_masks_exact',states=len(rows),repeats=args.repeats,
        numpy_version=np.__version__,python_version=platform.python_version(),
        scope='Feature construction with identical precomputed formal placements; excludes legal search, IPC, inference and optimizer',
        interpretation='Existing public-state implementation microbenchmark; no production adoption or end-to-end gain claim',
        input_snapshot_sha256=inputs,source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},
        fixtures=rows,paired_process_cpu_seconds=timings,median_paired_feature_speedup=statistics.median(ratios),
        reference_median_batch_cpu_seconds=statistics.median(r['reference'] for r in timings),
        bitboard_median_batch_cpu_seconds=statistics.median(r['bitboard'] for r in timings),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        next_gate='Before adoption: paired GPU8 transactions, exact policy tensors/non-timing metrics and deterministic resume')
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,result)
    operation('公开训练状态的既有候选特征实现配对微基准完成，当前训练未改实现',args.output)
    print(json.dumps({k:result[k] for k in ['states','parity','median_paired_feature_speedup','peak_rss_kib']}))


if __name__=='__main__':main()
