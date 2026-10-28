"""Exactly one deterministic failure replay and a fixed 512-piece BC rescue branch."""
import os
import resource
import subprocess
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

import training.bc_sequence_transfer as study
from training.bc_sequence_transfer import RUN, ROOT, read_record, read_blob, record, save_blob, digest, guard, operation
from training.bc_sequence_supervised import WORK
from training.bc_sequence_policy_audit import LABEL

OUT = RUN/'policy_audit'/'failure'


def main():
    OUT.mkdir(exist_ok=True)
    model_path = Path(read_record(WORK/'training_result.json')['last_model'])
    cases = [read_record(p) for p in sorted((WORK/'evaluation'/LABEL/'candidate').glob('seed_*.json'))]
    failed = [r for r in cases if r['game_over']]
    if len(cases) != 16 or len(failed) != 1:
        raise RuntimeError('This fixed diagnostic requires the single observed death, not case selection')
    case = failed[0]
    protocol = dict(seed=case['seed'],death_piece=case['pieces'],lookback=128,bc_follow=512,
        model_sha256=digest(model_path),source_sha256=digest(Path(__file__)),
        origin_sha256=digest(WORK/'evaluation'/LABEL/'candidate'/f'seed_{case["seed"]}.json'),
        mode='single postmortem, not training/selection/confirmation')
    path = OUT/'protocol.json'
    if path.exists() and read_record(path) != protocol:
        raise RuntimeError('Failure diagnostic identity changed')
    if not path.exists():
        record(path,protocol)
        operation('one fixed failure replay started: full-score parity, snapshot128 pieces before death, BC512-piece rescue; no tuning')
    if (OUT/'result.json').exists():
        return
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
    torch.set_num_threads(1)
    study.MODELS = dict(candidate=MaskablePPO.load(model_path,device='cpu'),
                        bc=MaskablePPO.load(study.BC,device='cpu'))
    env = study.LongCandidateTetrisEnv(max_pieces=20000)
    env.reset(seed=case['seed'])
    tail,boards = [],[]
    cp = OUT/'replay_checkpoint.pkl'
    if cp.exists():
        saved = read_blob(cp)
        if saved['protocol'] != protocol:
            raise RuntimeError('Replay checkpoint identity changed')
        env.set_state(saved['env'])
        tail,boards = saved['tail'],saved['boards']
    if env.core.score != case['scores'][env.pieces]:
        raise RuntimeError('Replay resume score mismatch')
    point = case['pieces']-128
    death_kind = None
    while env.pieces < case['pieces']:
        if env.pieces == point:
            save_blob(OUT/'same_core_rng.pkl',dict(protocol=protocol,env=env.get_state(),rng=study.rng_state()))
        env.get_observation();env.action_masks()
        placements = env._placement_cache
        row,over,capped = study.move(env,'candidate')
        if row['score'] != case['scores'][env.pieces]:
            raise RuntimeError(f'Failed original score parity at {env.pieces}')
        if env.pieces > point:
            tail.append(row)
            boards.append([''.join(cell or '.' for cell in line) for line in env.core.board])
        if over:
            placed = next(p for p in placements if p['actionId'] == row['action'])
            death_kind = 'Lock Out' if not any(c['y'] >= 0 for c in placed['occupiedCells']) else 'Block Out on next spawn'
            break
        if capped:
            raise RuntimeError('Unexpected replay truncation')
        if env.pieces % 2048 == 0:
            guard()
            save_blob(cp,dict(protocol=protocol,env=env.get_state(),tail=tail,boards=boards))
    if env.pieces != case['pieces'] or env.core.phase != 'over':
        raise RuntimeError('Original death was not reproduced')
    save_blob(OUT/'candidate_terminal_core_rng.pkl',dict(env=env.get_state(),rng=study.rng_state()))
    branch = study.LongCandidateTetrisEnv(max_pieces=20000)
    branch.reset(seed=case['seed'])
    start = read_blob(OUT/'same_core_rng.pkl')
    branch.set_state(start['env'])
    rescue = []
    rescue_cp = OUT/'rescue_checkpoint.pkl'
    if rescue_cp.exists():
        saved = read_blob(rescue_cp)
        if saved['protocol'] != protocol:
            raise RuntimeError('Rescue checkpoint identity changed')
        branch.set_state(saved['env']);rescue=saved['rows']
    over = False
    while len(rescue) < 512:
        row,over,capped = study.move(branch,'bc')
        rescue.append(row)
        if over or capped:
            break
        if len(rescue) % 64 == 0:
            save_blob(rescue_cp,dict(protocol=protocol,env=branch.get_state(),rows=rescue))
    result = dict(protocol=protocol,reproduced_score_prefix=env.pieces,
        death_kind=death_kind,terminal_height=tail[-1]['height'],terminal_holes=tail[-1]['holes'],
        terminal_wells=tail[-1]['wells'],terminal_high_streak=case['high_streak'],
        snapshot_sha256=digest(OUT/'same_core_rng.pkl'),start_piece=point,
        candidate_tail=tail,candidate_tail_boards=boards,bc_branch=rescue,
        bc_observed=len(rescue),bc_death=over,bc_right_censored=not over,
        bc_final_height=rescue[-1]['height'],bc_final_holes=rescue[-1]['holes'])
    save_blob(OUT/'bc_terminal_core_rng.pkl',dict(env=branch.get_state(),rng=study.rng_state()))
    record(OUT/'result.json',result)
    operation(f'one failure reproduced through {env.pieces} official scores: {death_kind}, height={tail[-1]["height"]}, holes={tail[-1]["holes"]}; BC rescue observed={len(rescue)}, death={over}, right_censored={not over}')
    print({k:v for k,v in result.items() if k not in ('candidate_tail','candidate_tail_boards','bc_branch')},flush=True)
    env.close();branch.close()


if __name__ == '__main__':
    main()
