"""Bounded fresh-seed PPO counterfactual dataset; resume at episode/branch boundaries."""
import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from training.candidate_features import board_features
from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.evaluation.hybrid_queue import HASHES, PPO, digest, load_models, verify_models
from training.evaluation.ppo_multiaction_counterfactual import compare
from training.tetris_core.legal_placements import get_legal_placements
from training.traditional.v2 import simulate_public
from training.train_ppo import ROOT, atomic_bytes, atomic_json, now

RUN = ROOT / 'runs/ppo30m_fresh_counterfactual'
SEEDS = list(range(1517000, 1517008))
VERSION = 'fresh_v1'
CAP = 3000
HORIZON = 2048
MAX_SECONDS = 5400
FIXED_STEPS = (400, 1400, 2400)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def protocol():
    verify_models()
    spec = {'version': VERSION, 'ppo_sha256': HASHES['ppo'], 'bc_sha256': HASHES['bc'],
            'seeds': SEEDS, 'episode_cap': CAP, 'fixed_steps': list(FIXED_STEPS),
            'danger': 'first two steps >=200 apart with board max height >=14 or central top four occupied >=2, after step 100',
            'max_states_per_seed': 5, 'max_candidates_per_state': 5,
            'candidates': 'PPO top1/top2, BC top1, public lowest (game-over,height,holes), maximum immediate four-clear; deduplicate in this order',
            'continuation': 'same Core/RNG snapshot; one candidate action then deterministic frozen PPO',
            'horizon': HORIZON, 'branch_budget': 8*5*5, 'wall_seconds': MAX_SECONDS,
            'censoring': 'survival claims require observed death and a 256-piece gap; common-window score/fours only',
            'negative': 'candidate dies >=256 pieces before observed PPO, or at least 2 fewer four-clears and <=80% PPO four-clears over >=512 common pieces',
            'seed_grouping': 'episode seed is analysis unit; states and branches within seed are dependent'}
    path = RUN/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != spec: raise RuntimeError('Frozen protocol mismatch')
    else:
        names = {f'seed_{s}' for s in SEEDS}
        collisions = [str(p) for base in (ROOT/'runs', ROOT/'reports') for p in base.rglob('seed_*.*')
                      if p.stem.split('.')[0] in names and not p.is_relative_to(RUN)]
        if collisions: raise RuntimeError(f'Seed collision: {collisions[:4]}')
        atomic_json(path, spec)
        atomic_json(RUN/'seed_audit.json', {'seeds': SEEDS, 'collision_count': 0, 'audited_at': now()})
    return spec


def state_path(seed, step):
    return RUN/'states'/f'seed_{seed}_step_{step}.pkl.gz'


def capture(seed, model):
    manifest = RUN/'episodes'/f'seed_{seed}.json'
    if manifest.exists():
        row = json.loads(manifest.read_text())
        if row['version'] != VERSION or row['ppo_sha256'] != HASHES['ppo']: raise RuntimeError('Episode identity mismatch')
        for item in row['states']:
            if digest(state_path(seed,item['step'])) != item['snapshot_sha256']: raise RuntimeError('Snapshot checksum mismatch')
        return row
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    states, danger_steps = [], []
    try:
        obs,_ = env.reset(seed=seed)
        while env.pieces < CAP:
            step = env.pieces
            public = env.core.get_public_observation()
            board = public['board']
            heights = [next((20-y for y in range(20) if board[y][x] is not None),0) for x in range(10)]
            central = sum(board[y][x] is not None for y in range(4) for x in range(3,7))
            danger = step >= 100 and (max(heights)>=14 or central>=2)
            selected = step in FIXED_STEPS or (danger and len(danger_steps)<2 and all(step-x>=200 for x in danger_steps))
            if selected:
                snapshot = env.get_state()
                payload = gzip.compress(pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=3)
                path = state_path(seed,step)
                atomic_bytes(path,payload)
                states.append({'step':step,'stratum':'fixed' if step in FIXED_STEPS else 'danger',
                               'height':max(heights),'central_top4':central,
                               'public_sha256':sha(json.dumps(public,sort_keys=True).encode()),
                               'snapshot_sha256':sha(payload)})
                if step not in FIXED_STEPS: danger_steps.append(step)
            action = int(model.predict(obs,deterministic=True,action_masks=env.action_masks())[0])
            obs,_,done,capped,info = env.step(action)
            if done or capped: break
        row = {'version':VERSION,'ppo_sha256':HASHES['ppo'],'seed':seed,
               'pieces':env.pieces,'game_over':bool(done),'states':states,'completed_at':now()}
        atomic_json(manifest,row)
        return row
    finally: env.close()


def options(snapshot, models):
    env = FastCandidateTetrisEnv(max_pieces=CAP)
    try:
        env.reset(seed=snapshot['episode_seed']); env.set_state(snapshot)
        obs,mask = env.get_observation(),env.action_masks()
        public = env.core.get_public_observation()
        placements = {p['actionId']:p for p in get_legal_placements(public)}
        tensor_obs = {key: torch.as_tensor(value,device='cpu').unsqueeze(0) for key,value in obs.items()}
        with torch.no_grad():
            probs = models['ppo'].policy.get_distribution(
                tensor_obs,action_masks=torch.as_tensor(mask).unsqueeze(0)).distribution.probs[0].cpu().numpy()
        ranked = sorted(placements,key=lambda a:(-float(probs[a]),a))
        ppo = int(models['ppo'].predict(obs,deterministic=True,action_masks=mask)[0])
        if ranked[0] != ppo: raise RuntimeError('PPO top1 mismatch')
        bc = int(models['bc'].predict(obs,deterministic=True,action_masks=mask)[0])
        safety, fours = {}, {}
        for action, placement in placements.items():
            child, clear = simulate_public(public,placement)
            holes,_,height,_,_ = board_features(child['board'])
            safety[action] = (child['phase']=='over',height,holes,action)
            fours[action] = int(clear==4)
        source = [(ppo,'ppo'),(ranked[1],'ppo_top2'),(bc,'bc'),
                  (min(placements,key=lambda a:safety[a]),'public_safe'),
                  (min(placements,key=lambda a:(-fours[a],safety[a])),'immediate_four')]
        unique = {}
        for action,name in source:
            if action not in unique: unique[action] = {'action':action,'sources':[],'ppo_probability':float(probs[action]),
                'ppo_rank':ranked.index(action)+1,'hold':bool(placements[action]['hold']),
                'immediate_four':fours[action],'public_safety':list(safety[action][:3])}
            unique[action]['sources'].append(name)
        return list(unique.values())
    finally: env.close()


def classify(base, other):
    metric = compare(base,other)
    survival_harm = bool(other['game_over'] and base['pieces']-other['pieces']>=256)
    four_harm = bool(metric['common_pieces']>=512 and metric['ppo_fours']>=3 and
                     metric['ppo_fours']-metric['candidate_fours']>=2 and
                     metric['candidate_fours']<=.8*metric['ppo_fours'])
    return {**metric,'survival_harm':survival_harm,'four_harm':four_harm,
            'negative':survival_harm or four_harm,
            'ppo_censored':base['censored'],'candidate_censored':other['censored']}


def branch(snapshot, action, models, horizon=HORIZON):
    """Run one changed action, then the frozen PPO, from an identical RNG state."""
    cap = max(snapshot['max_pieces'], snapshot['pieces'] + horizon + 1)
    state = dict(snapshot, max_pieces=cap)
    env = FastCandidateTetrisEnv(max_pieces=cap)
    try:
        env.reset(seed=state['episode_seed']); env.set_state(state)
        start, initial_score = env.pieces, env.core.score
        if not env.action_masks()[action]: raise RuntimeError('Illegal first action')
        obs = env.get_observation()
        scores, fours = [], []
        done = capped = False
        while env.pieces-start < horizon:
            chosen = action if env.pieces == start else int(models['ppo'].predict(
                obs,deterministic=True,action_masks=env.action_masks())[0])
            obs,_,done,capped,info = env.step(chosen)
            scores.append(info['score']-initial_score)
            fours.append((fours[-1] if fours else 0)+int(info['cleared_lines']==4))
            if done or capped: break
        return {'action':action,'pieces':len(scores),'game_over':bool(done),
                'censored':not(done or capped),'horizon':horizon,'scores':scores,'fours':fours}
    finally: env.close()


def process(item, seed, models):
    step = item['step']; path = state_path(seed,step)
    if digest(path)!=item['snapshot_sha256']: raise RuntimeError('Snapshot changed')
    snapshot = pickle.loads(gzip.decompress(path.read_bytes())) # locally generated, checksum verified
    choices = options(snapshot,models)
    choice_path = RUN/'candidates'/f'seed_{seed}_step_{step}.json'
    identity = {'version':VERSION,'snapshot_sha256':item['snapshot_sha256'],'choices':choices}
    if choice_path.exists():
        if json.loads(choice_path.read_text())!=identity: raise RuntimeError('Candidate drift')
    else: atomic_json(choice_path,identity)
    results = {}
    for choice in choices:
        action = choice['action']
        result_path = RUN/'branches'/f'seed_{seed}_step_{step}_action_{action}.json'
        branch_id = {'version':VERSION,'snapshot_sha256':item['snapshot_sha256'],
                     'candidate_sha256':digest(choice_path),'action':action,'ppo_sha256':HASHES['ppo']}
        if result_path.exists():
            record = json.loads(result_path.read_text())
            if any(record.get(k)!=v for k,v in branch_id.items()): raise RuntimeError('Branch identity mismatch')
            outcome = record['outcome']
        else:
            outcome = branch(snapshot,action,models,HORIZON)
            atomic_json(result_path,{**branch_id,'outcome':outcome,'completed_at':now()})
        results[action] = outcome
    base = results[choices[0]['action']]
    summary = {'version':VERSION,'seed':seed,'step':step,'stratum':item['stratum'],
               'candidate_sha256':digest(choice_path),'ppo_pieces':base['pieces'],
               'comparisons':{str(c['action']):classify(base,results[c['action']]) for c in choices[1:]}}
    out = RUN/'comparisons'/f'seed_{seed}_step_{step}.json'
    if out.exists() and json.loads(out.read_text())!=summary: raise RuntimeError('Comparison drift')
    if not out.exists(): atomic_json(out,summary)
    return summary


def main():
    protocol(); started=time.monotonic(); models=load_models(); done=0
    for seed in SEEDS:
        episode=capture(seed,models['ppo'])
        for item in episode['states']:
            if time.monotonic()-started>MAX_SECONDS:
                atomic_json(RUN/'status.json',{'status':'budget_stop','states':done,'updated_at':now()})
                return
            process(item,seed,models); done+=1
            atomic_json(RUN/'status.json',{'status':'running','states':done,'last_seed':seed,'last_step':item['step'],'updated_at':now()})
            print(f'{done} seed={seed} step={item["step"]}',flush=True)
    atomic_json(RUN/'status.json',{'status':'complete','states':done,'updated_at':now()})


if __name__=='__main__': main()
