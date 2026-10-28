"""Pre-registered, resumable finite-window sequence transfer study (no PPO updates)."""
import argparse
import hashlib
import json
import os
import pickle
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, atomic_pickle, fast_action
from training.evaluation.hybrid_queue import BC, PPO, digest, load_models, verify_models
from training.train_ppo import ROOT, atomic_json, now

RUN = ROOT / 'runs/bc_sequence_transfer'
SEEDS = tuple(range(3102801000, 3102801006))
LENGTHS = (16, 32, 64, 128)
STARTS = (256, 1024, 4096, 8192)
CAP = 12000
FOLLOW = 2048
CHECKPOINT_INTERVAL = 512
MODELS = None
IDENTITY = None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def record(path, value):
    atomic_json(path, {'data': value, 'sha256': hashlib.sha256(canonical(value)).hexdigest()})


def read_record(path):
    item = json.loads(path.read_text())
    if hashlib.sha256(canonical(item['data'])).hexdigest() != item['sha256']:
        raise RuntimeError(f'JSON checksum mismatch: {path}')
    return item['data']


def save_blob(path, value):
    payload = pickle.dumps(value, protocol=5)
    atomic_pickle(path, {'payload': payload, 'sha256': hashlib.sha256(payload).hexdigest()})


def read_blob(path):
    path = Path(path)
    item = pickle.loads(path.read_bytes())
    if hashlib.sha256(item['payload']).hexdigest() != item['sha256']:
        raise RuntimeError(f'Snapshot checksum mismatch: {path}')
    return pickle.loads(item['payload'])


def operation(message):
    with (ROOT / 'OPERATIONS.md').open('a') as stream:
        stream.write(f'{now()} | BC sequence transfer: {message}\n')


def status(stage, **values):
    atomic_json(RUN / 'status.json', dict(at=now(), stage=stage, pid=os.getpid(), **values))


def guard():
    protocol = read_record(RUN / 'protocol.json')
    if time.time() - protocol['started_epoch'] > 12 * 3600:
        raise RuntimeError('12 hour wall budget exhausted')
    if shutil.disk_usage(RUN).free < 8 * 1024**3:
        raise RuntimeError('Less than 8 GiB free')
    if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()) > 4 * 1024**3:
        raise RuntimeError('4 GiB data budget exhausted')


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    verify_models()
    source_files = [Path('training/bc_sequence_transfer.py'), Path('training/bc_sequence_report.py'),
                    Path('training/candidate_policy.py'), Path('training/candidate_features.py'),
                    Path('training/hybrid_policy.py'), Path('training/train_ppo.py'),
                    Path('training/evaluation/hybrid_queue.py'),
                    Path('training/evaluation/failure_curriculum_eval.py'),
                    Path('training/evaluation/fast_legal.py'),
                    Path('training/evaluation/fast_candidate_features.py'),
                    Path('reports/experiments/bc_sequence_transfer_design.md')]
    source_files += sorted(Path('training/env').glob('*.py'))
    source_files += sorted(Path('training/tetris_core').rglob('*.py'))
    sources = {str(p): digest(ROOT / p) for p in source_files}
    models = {'bc': digest(BC), 'ppo': digest(PPO)}
    path = RUN / 'protocol.json'
    if path.exists():
        protocol = read_record(path)
        if protocol['sources'] != sources or protocol['models'] != models:
            raise RuntimeError('Frozen source/model identity changed; refusing resume')
        return protocol
    if subprocess.run(['git', 'diff', '--quiet', '--', *map(str, source_files)], cwd=ROOT).returncode:
        raise RuntimeError('Commit implementation/design before protocol freeze')
    # Conservative exact-token scan includes any use, not just a seed-key match.
    raw = subprocess.check_output(['git', 'worktree', 'list', '--porcelain'], cwd=ROOT, text=True)
    roots = sorted({str(Path(line[9:]) / folder) for line in raw.splitlines()
                    if line.startswith('worktree ') for folder in ('runs', 'reports', 'training', 'ops')
                    if (Path(line[9:]) / folder).exists()})
    audit = subprocess.run(['rg', '-l', '--hidden', '--glob', '*.json', '--glob', '*.jsonl',
                            '--glob', '*.md', '--glob', '*.py', '--glob', '*.sh', '--glob', '*.csv',
                            '--glob', '!bc_sequence_transfer/**',
                            r'\b310280[1-4][0-9]{3}\b', *roots], capture_output=True, text=True)
    # The new design and implementation declare reservations but have not consumed seeds.
    own = {str(ROOT / p) for p in source_files}
    matches = [p for p in audit.stdout.splitlines()
               if p not in own and '/bc_sequence_transfer/' not in p]
    if audit.returncode not in (0, 1) or matches:
        raise RuntimeError(f'Seed audit failed: matches={matches}, error={audit.stderr[:1000]}')
    record(RUN / 'seed_audit.json', {'at': now(), 'roots': roots, 'matches': matches,
           'method': 'conservative exact integer token scan of all registered worktrees',
           'range': '3102801000..3102804999', 'excluded': sorted(own)})
    protocol = dict(version='bc-sequence-transfer-v1', started_epoch=time.time(), at=now(),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    sources=sources, models=models, study_seeds=SEEDS,
                    training_seeds=list(range(3102802000, 3102802032)),
                    development_seeds=list(range(3102803000, 3102803016)),
                    confirmation_seeds=list(range(3102804000, 3102804032)),
                    starts=STARTS, lengths=LENGTHS, follow=FOLLOW, cap=CAP,
                    device_detected='cuda' if torch.cuda.is_available() else 'cpu', inference_device='cpu',
                    optimizer='not applicable: stage I frozen policies',
                    seed_audit_sha256=digest(RUN / 'seed_audit.json'))
    record(path, protocol)
    operation('stage-I protocol frozen before sampling; six independent seeds, four states/seed, four lengths')
    return protocol


def worker_init(identity):
    global MODELS, IDENTITY
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    torch.set_num_threads(1)
    MODELS = load_models()
    IDENTITY = identity


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def layout(board):
    heights = [next((20-y for y, row in enumerate(board) if row[x] is not None), 0) for x in range(10)]
    depths = [max(0, min(20 if x == 0 else heights[x-1],
                        20 if x == 9 else heights[x+1]) - heights[x]) for x in range(10)]
    return heights, depths


def move(env, source):
    public = env.core.get_public_observation()
    obs, mask = env.get_observation(), env.action_masks()
    if env.pieces % 512 == 0 and env._placement_cache != env.core.get_legal_placements():
        raise RuntimeError('Accelerated placement generator differs from frozen Core')
    begin = time.perf_counter()
    action = fast_action(MODELS[source], obs, mask)
    elapsed = time.perf_counter() - begin
    if not mask[action]:
        raise RuntimeError('Illegal teacher action')
    placement = next(p for p in env._placement_cache if p['actionId'] == action)
    hold = bool(placement['hold'])
    placed = (public['hold'] or public['next'][0]) if hold else public['currentPiece']['type']
    _, _, over, capped, info = env.step(action)
    heights, wells = layout(env.core.board)
    row = dict(piece=env.pieces, action=action, source=source, hold=hold, placed=placed,
               current=public['currentPiece']['type'], preview=public['next'], held=public['hold'],
               score=info['score'], clear=info['cleared_lines'], height=info['height_after'],
               holes=info['holes_after'], new_holes=info['new_holes'], heights=heights, wells=wells,
               inference_seconds=elapsed)
    return row, bool(over), bool(capped)


def job_path(kind, seed, start=0, length=0):
    folder = RUN / ('native' if kind in ('bc', 'ppo') else 'branches')
    return folder / f'{kind}_{seed}_{start}_{length}.pkl'


def run_job(kind, seed, start=0, length=0):
    path = job_path(kind, seed, start, length)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = (kind, seed, start, length)
    if path.exists():
        result = read_blob(path)
        if result['identity'] != IDENTITY or result['key'] != key:
            raise RuntimeError('Completed job identity mismatch')
        return str(path)
    guard()
    env = LongCandidateTetrisEnv(max_pieces=CAP)
    env.reset(seed=seed)
    snapshot_hash = None
    if kind == 'branch':
        snapshot = RUN / 'snapshots' / f'bc_{seed}_{start}.pkl'
        if not snapshot.exists():
            return None  # Prespecified state missing due to actual BC death; no replacement.
        saved = read_blob(snapshot)
        if saved['identity'] != IDENTITY:
            raise RuntimeError('Snapshot protocol mismatch')
        snapshot_hash = digest(snapshot)
        env.set_state(saved['env'])
    initial_score = env.core.score
    limit = CAP if kind in ('bc', 'ppo') else start + (length or max(LENGTHS)) + FOLLOW
    result = dict(identity=IDENTITY, key=key, initial_score=initial_score, initial_piece=start,
                  limit=limit, snapshot_sha256=snapshot_hash, rows=[], game_over=False, capped=False)
    cp = path.with_suffix('.checkpoint.pkl')
    if cp.exists():
        saved = read_blob(cp)
        if saved['result']['identity'] != IDENTITY or tuple(saved['result']['key']) != key:
            raise RuntimeError('Checkpoint identity mismatch')
        result = saved['result']
        env.set_state(saved['env'])
        restore_rng(saved['rng'])
        if result['snapshot_sha256'] != snapshot_hash:
            raise RuntimeError('Starting snapshot changed')
    try:
        while env.pieces < limit:
            source = kind if kind in ('bc', 'ppo') else ('ppo' if env.pieces-start < length else 'bc')
            row, over, capped = move(env, source)
            result['rows'].append(row)
            if kind == 'bc' and env.pieces in STARTS and not (over or capped):
                (RUN / 'snapshots').mkdir(exist_ok=True)
                save_blob(RUN / 'snapshots' / f'bc_{seed}_{env.pieces}.pkl',
                          dict(identity=IDENTITY, env=env.get_state(), rng=rng_state()))
            if over or capped:
                result['game_over'] = over
                break
            if env.pieces % CHECKPOINT_INTERVAL == 0:
                guard()
                save_blob(cp, dict(result=result, env=env.get_state(), rng=rng_state()))
        result['capped'] = not result['game_over']  # finite follow-up is right censoring
        result['final_state'] = env.get_state()
        result['rng'] = rng_state()
        result['pieces_observed'] = len(result['rows'])
        save_blob(path, result)
        if cp.exists():
            cp.unlink()
        return str(path)
    finally:
        env.close()


def pair(base, intervention, length):
    horizon = length + FOLLOW
    n = min(len(base['rows']), len(intervention['rows']), horizon)
    if not n:
        raise RuntimeError('Empty common window')
    a, b = base['rows'][:n], intervention['rows'][:n]
    def metrics(rows, run):
        observed = min(len(run['rows']), horizon)
        exposed = run['rows'][:observed]
        return dict(score=rows[-1]['score']-run['initial_score'],
                    pieces=observed, death=int(run['game_over'] and len(run['rows']) <= horizon),
                    high=sum(r['height'] >= 17 for r in rows), new_holes=sum(r['new_holes'] for r in rows),
                    four=sum(r['clear'] == 4 for r in rows),
                    full_four=sum(r['clear'] == 4 for r in exposed))
    divergence = next((i+1 for i, (x, y) in enumerate(zip(a, b))
                       if (x['current'], x['preview'], x['held']) != (y['current'], y['preview'], y['held'])), None)
    return dict(seed=intervention['key'][1], start=intervention['key'][2], length=length,
                common_pieces=n, baseline=metrics(a, base), intervention=metrics(b, intervention),
                first_public_sequence_divergence=divergence,
                hold_counts=[sum(r['hold'] for r in rows) for rows in (a, b)],
                placed_mismatches=sum(x['placed'] != y['placed'] for x, y in zip(a, b)),
                snapshot_sha256=intervention['snapshot_sha256'])


def decide(pairs):
    output = {}
    for length in LENGTHS:
        rows = [p for p in pairs if p['length'] == length]
        grouped = []
        for seed in SEEDS:
            sample = [p for p in rows if p['seed'] == seed]
            grouped.append([sum(p['baseline']['score'] for p in sample),
                            sum(p['intervention']['score'] for p in sample)])
        data = np.asarray(grouped, dtype=float)
        valid = bool(np.all(data[:, 0] > 0))
        ratio = float(data[:, 1].sum()/data[:, 0].sum()) if valid else None
        rng = np.random.default_rng(20261028)
        bootstrap = data[rng.integers(0, len(SEEDS), (10000, len(SEEDS)))].sum(axis=1)
        ci = np.percentile(bootstrap[:, 1]/np.maximum(1, bootstrap[:, 0]), [0.625, 99.375]).tolist()
        common = max(1, sum(p['common_pieces'] for p in rows))
        stats = {}
        for side in ('baseline', 'intervention'):
            stats[side] = dict(deaths=sum(p[side]['death'] for p in rows),
                observed=sum(p[side]['pieces'] for p in rows),
                high_fraction=sum(p[side]['high'] for p in rows)/common,
                new_holes_per100=100*sum(p[side]['new_holes'] for p in rows)/common,
                fours=sum(p[side]['four'] for p in rows))
        a, b = stats['baseline'], stats['intervention']
        checks = dict(complete=len(rows) == len(SEEDS)*len(STARTS),
                      score=valid and ratio >= 1.03,
                      cross_seed=int(np.sum(data[:, 1] > data[:, 0])) >= 5,
                      adjusted_ci=ci[0] > 1,
                      deaths=b['deaths'] <= a['deaths'],
                      observed_survival=b['observed'] >= .99*a['observed'],
                      high=b['high_fraction'] <= a['high_fraction']+.02,
                      holes=b['new_holes_per100'] <= max(a['new_holes_per100']*1.5, a['new_holes_per100']+.5))
        output[str(length)] = dict(passed=all(checks.values()), checks=checks, common_score_ratio=ratio,
            ratio_ci98_75=ci, positive_seeds=int(np.sum(data[:, 1] > data[:, 0])),
            per_seed=[dict(seed=s, baseline=x[0], intervention=x[1]) for s, x in zip(SEEDS, grouped)],
            **stats)
    passed = [h for h in LENGTHS if output[str(h)]['passed']]
    return dict(passed=bool(passed), selected_length=min(passed) if passed else None, lengths=output)


def native_analysis(result):
    rows = result['rows']
    windows = {}
    for length in LENGTHS:
        groups = {'survived_2048': [], 'death_within_2048': [], 'right_censored': []}
        # All four-clears with complete preceding windows, no successful-case selection.
        for i, row in enumerate(rows):
            if row['clear'] != 4 or i+1 < length:
                continue
            segment = rows[i+1-length:i+1]
            future = min(FOLLOW, len(rows)-1-i)
            category = ('death_within_2048' if result['game_over'] and len(rows)-1-i <= FOLLOW
                        else 'survived_2048' if future == FOLLOW else 'right_censored')
            initial = rows[i-length]['score'] if i >= length else result['initial_score']
            groups[category].append(dict(start=segment[0]['piece'], end=row['piece'],
                score_gain=row['score']-initial, hold=sum(r['hold'] for r in segment),
                first_height=segment[0]['height'], final_height=row['height'],
                max_height=max(r['height'] for r in segment),
                mean_holes=float(np.mean([r['holes'] for r in segment])),
                max_well=max(max(r['wells']) for r in segment),
                fours=sum(r['clear'] == 4 for r in segment), future_observed=future,
                high_steps=sum(r['height'] >= 17 for r in segment)))
        windows[str(length)] = groups
    return dict(seed=result['key'][1], source=result['key'][0], pieces=len(rows),
        score=rows[-1]['score'], death=result['game_over'], right_censored=result['capped'],
        clears=[sum(r['clear'] == k for r in rows) for k in range(5)],
        mean_height=float(np.mean([r['height'] for r in rows])),
        mean_holes=float(np.mean([r['holes'] for r in rows])),
        high_fraction=float(np.mean([r['height'] >= 17 for r in rows])),
        hold_fraction=float(np.mean([r['hold'] for r in rows])),
        well_ge4_fraction=float(np.mean([max(r['wells']) >= 4 for r in rows])),
        inference_seconds=sum(r['inference_seconds'] for r in rows), windows=windows)


def analyze():
    pairs, native = [], []
    for seed in SEEDS:
        for source in ('bc', 'ppo'):
            result = read_blob(job_path(source, seed))
            native.append(native_analysis(result))
        for start in STARTS:
            path = job_path('branch', seed, start, 0)
            if not path.exists():
                continue
            base = read_blob(path)
            # Restored BC branch must reproduce natural BC exactly (excluding runtime).
            original = read_blob(job_path('bc', seed))['rows'][start:start+len(base['rows'])]
            strip = lambda r: {k: v for k, v in r.items() if k != 'inference_seconds'}
            if [strip(r) for r in original] != [strip(r) for r in base['rows']]:
                raise RuntimeError('Restored BC branch failed natural-trajectory parity')
            for length in (1, *LENGTHS):
                pairs.append(pair(base, read_blob(job_path('branch', seed, start, length)), length))
    decision = decide(pairs)
    record(RUN / 'pairs.json', pairs)
    record(RUN / 'native_analysis.json', native)
    record(RUN / 'decision.json', decision)
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    protocol = freeze()
    if args.freeze_only:
        return
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, affinity[-2:])
    identity = digest(RUN / 'protocol.json')
    status('native_sampling', identity=identity)
    operation('tmux stage-I sampling started/resumed; two CPU workers; finite budgets enforced')
    with ProcessPoolExecutor(2, mp_context=get_context('spawn'), initializer=worker_init,
                             initargs=(identity,)) as pool:
        native = [pool.submit(run_job, source, seed) for seed in SEEDS for source in ('bc', 'ppo')]
        for count, future in enumerate(as_completed(native), 1):
            future.result()
            status('native_sampling', completed=count, total=len(native))
        status('finite_takeovers')
        branches = [pool.submit(run_job, 'branch', seed, start, length)
                    for seed in SEEDS for start in STARTS for length in (0, 1, *LENGTHS)]
        for count, future in enumerate(as_completed(branches), 1):
            future.result()
            status('finite_takeovers', completed=count, total=len(branches))
    decision = analyze()
    status('stage1_positive' if decision['passed'] else 'stopped_stage1_gate', decision=decision)
    operation(f'stage-I completed; gate passed={decision["passed"]}; selected length={decision["selected_length"]}; no gradient updates')
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    from training.bc_sequence_report import close_out
    close_out()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        RUN.mkdir(parents=True, exist_ok=True)
        status('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        operation(f'failed/interrupted: {type(exc).__name__}: {exc}; committed checkpoints retained')
        traceback.print_exc()
        raise
