"""The single conditional supervised route; refuses to start without stage-I evidence."""
import hashlib
import math
import os
import random
import resource
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

import training.bc_sequence_transfer as study
from training.bc_sequence_transfer import (RUN, ROOT, BC, PPO, CAP, FOLLOW, guard, digest,
    save_blob, read_blob, record, read_record, rng_state, restore_rng, operation, status)
from training.imitation_bc import batch, metrics
import training.bc_score_rl as evaluation

WORK = RUN / 'supervised'
DATA_SEEDS = tuple(range(3102802000, 3102802008))
WORK_ID = None
HORIZON = None


def freeze():
    first = read_record(RUN / 'decision.json')
    if not first['passed']:
        raise RuntimeError('Stage-I gate did not pass; supervised training is forbidden')
    study.freeze()  # also verifies unchanged original models and Core
    WORK.mkdir(parents=True, exist_ok=True)
    names = ['training/bc_sequence_supervised.py', 'training/bc_sequence_supervised_report.py',
             'training/imitation_bc.py', 'training/bc_score_rl.py', 'training/bc_sequence_diagnostics.py',
             'reports/experiments/bc_sequence_transfer_supervised_design.md']
    sources = {name: digest(ROOT/name) for name in names}
    path = WORK / 'protocol.json'
    if path.exists():
        protocol = read_record(path)
        if protocol['sources'] != sources or protocol['stage1'] != digest(RUN/'protocol.json'):
            raise RuntimeError('Supervised source identity changed')
        return protocol
    if subprocess.run(['git', 'diff', '--quiet', '--', *names], cwd=ROOT).returncode:
        raise RuntimeError('Commit supervised implementation before collecting data')
    protocol = dict(at=study.now(), stage1=digest(RUN/'protocol.json'),
        decision_sha256=digest(RUN/'decision.json'), sources=sources,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        length=first['selected_length'], seeds=DATA_SEEDS, train_seeds=DATA_SEEDS[:6], val_seeds=DATA_SEEDS[6:],
        max_new_episodes=24, cap=CAP, follow=FOLLOW, epochs=3, lr=3e-5,
        loss_weights=[.5, .4, .1, .1], frames_per_batch_per_source=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        frozen_models={'bc':digest(BC), 'ppo':digest(PPO)})
    record(path, protocol)
    operation('conditional supervised protocol frozen; 24-episode cap, one loss, max three epochs')
    return protocol


def init_worker(identity, horizon):
    global WORK_ID, HORIZON
    study.worker_init(identity)
    WORK_ID, HORIZON = identity, horizon


def data_path(source, seed):
    return WORK / 'data' / f'{source}_{seed}.pkl'


def public_sample(env):
    obs, mask = env.get_observation(), env.action_masks()
    ids = np.flatnonzero(mask)
    return dict(state=obs['state'].copy(), legal_ids=ids.astype(np.uint16),
                legal_features=obs['candidate'][ids].astype(np.uint8), piece=env.pieces)


def select_native_frames(rows, frames, horizon, over):
    selected = []
    for end in range(horizon, len(rows)-FOLLOW+1, horizon):
        if over and len(rows)-end <= FOLLOW:
            continue
        if not any(r['clear'] == 4 for r in rows[end-horizon:end]):
            continue
        if len(selected)+horizon > 4096:
            break
        for frame in frames[end-horizon:end]:
            selected.append(dict(frame, fragment=end//horizon))
    return selected


def collect(source, seed):
    path = data_path(source, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        result = read_blob(path)
        if result['identity'] != WORK_ID:
            raise RuntimeError('Dataset identity mismatch')
        return str(path)
    guard()
    env = study.LongCandidateTetrisEnv(max_pieces=CAP)
    env.reset(seed=seed)
    start = 1024 if source == 'transfer' else 0
    if source == 'transfer':
        saved = read_blob(WORK/'snapshots'/f'bc_{seed}_1024.pkl')
        if saved['identity'] != WORK_ID:
            raise RuntimeError('Dataset snapshot identity mismatch')
        env.set_state(saved['env'])
    limit = start+HORIZON+FOLLOW if source == 'transfer' else CAP
    result = dict(identity=WORK_ID, seed=seed, source=source, initial_score=env.core.score,
                  rows=[], frames=[], game_over=False, right_censored=False)
    cp = path.with_suffix('.checkpoint.pkl')
    if cp.exists():
        saved = read_blob(cp)
        if saved['result']['identity'] != WORK_ID:
            raise RuntimeError('Dataset checkpoint identity mismatch')
        result = saved['result']
        env.set_state(saved['env'])
        restore_rng(saved['rng'])
    try:
        while env.pieces < limit:
            teacher = source == 'ppo' or (source == 'transfer' and env.pieces-start < HORIZON)
            keep = source == 'ppo' or (source == 'bc' and env.pieces % 4 == 0) or teacher
            sample = public_sample(env) if keep else None
            row, over, capped = study.move(env, 'ppo' if teacher else 'bc')
            result['rows'].append(row)
            if sample is not None:
                sample.update(action=row['action'], episode=f'{source}:{seed}', fragment=0)
                result['frames'].append(sample)
            if source == 'bc' and env.pieces == 1024 and not over:
                (WORK/'snapshots').mkdir(exist_ok=True)
                save_blob(WORK/'snapshots'/f'bc_{seed}_1024.pkl',
                          dict(identity=WORK_ID, env=env.get_state(), rng=rng_state()))
            if over or capped:
                result['game_over'] = over
                break
            if env.pieces % 512 == 0:
                guard()
                save_blob(cp, dict(result=result, env=env.get_state(), rng=rng_state()))
        result['right_censored'] = not result['game_over']
        result['observed'] = len(result['rows'])
        result['raw_frames'] = len(result['frames'])
        if source == 'ppo':
            result['frames'] = select_native_frames(result['rows'], result['frames'], HORIZON, result['game_over'])
        if source == 'transfer':
            baseline = read_blob(data_path('bc', seed))['rows']
            bc_gain = baseline[start+HORIZON-1]['score']-baseline[start-1]['score']
            teacher_gain = result['rows'][min(HORIZON,len(result['rows']))-1]['score']-result['initial_score']
            eligible = (len(result['rows']) == HORIZON+FOLLOW and not result['game_over']
                        and any(r['clear'] == 4 for r in result['rows'][:HORIZON])
                        and teacher_gain >= 1.05*bc_gain)
            result['eligible'] = eligible
            result['teacher_gain'], result['baseline_gain'] = teacher_gain, bc_gain
            if not eligible:
                result['frames'] = []
        result.update(env=env.get_state(), rng=rng_state())
        save_blob(path, result)
        if cp.exists():
            cp.unlink()
        return str(path)
    finally:
        env.close()


def groups():
    partitions = {'train':{}, 'validation':{}}
    for partition, seeds in (('train', DATA_SEEDS[:6]), ('validation', DATA_SEEDS[6:])):
        for source in ('bc', 'ppo', 'transfer'):
            partitions[partition][source] = [frame for seed in seeds
                for frame in read_blob(data_path(source, seed))['frames']]
    train_ids = {r['episode'] for rows in partitions['train'].values() for r in rows}
    val_ids = {r['episode'] for rows in partitions['validation'].values() for r in rows}
    if train_ids & val_ids:
        raise RuntimeError('Episode leakage')
    record(WORK/'dataset_manifest.json', dict(counts={p:{s:len(v) for s,v in groups.items()}
        for p,groups in partitions.items()}, train_episodes=sorted(train_ids), validation_episodes=sorted(val_ids),
        sha256={str(p.relative_to(WORK)):digest(p) for p in sorted((WORK/'data').glob('*.pkl'))}))
    return partitions


def measured(policy, rows, device):
    indices = np.linspace(0, len(rows)-1, min(4096,len(rows)), dtype=int)
    return metrics(policy, [rows[i] for i in indices], device)


def chunks(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault((r['episode'], r['fragment']), []).append(r)
    return list(grouped.values())


def draw_segment(fragments):
    segment = random.choice(fragments)
    begin = random.randrange(max(1, len(segment)-15))
    return segment[begin:begin+16]


def learning_checks(baseline, current):
    return dict(bc_agreement=current['bc']['top1'] >= .9,
        native_signal=current['ppo']['loss'] <= .99*baseline['ppo']['loss'],
        target_signal=current['transfer']['loss'] <= .99*baseline['transfer']['loss'])


def save_update(model, optimizer, state, identity, label):
    parent = WORK/'checkpoints'
    parent.mkdir(exist_ok=True)
    temporary = parent/'.pending'
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    model.save(temporary/'model.zip')
    torch.save(optimizer.state_dict(), temporary/'optimizer.pt')
    save_blob(temporary/'state.pkl', dict(state=state, rng=rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    manifest = dict(identity=identity, at=study.now(), label=label,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sha256={p.name:digest(p) for p in temporary.iterdir()})
    record(temporary/'manifest.json', manifest)
    destination = parent/label
    if destination.exists():
        raise RuntimeError(f'Refusing to overwrite checkpoint {label}')
    os.replace(temporary, destination)
    record(WORK/'latest.json', dict(path=str(destination.relative_to(WORK)),
                                   manifest_sha256=digest(destination/'manifest.json')))
    return destination


def load_update(path, device, identity):
    manifest = read_record(path/'manifest.json')
    if manifest['identity'] != identity or any(digest(path/name) != sha for name,sha in manifest['sha256'].items()):
        raise RuntimeError('Supervised checkpoint identity/checksum mismatch')
    model = MaskablePPO.load(path/'model.zip', device=device)
    actor = [p for name,p in model.policy.named_parameters() if name.startswith(('state_encoder.', 'candidate_encoder.', 'score_head.'))]
    optimizer = torch.optim.Adam(actor, lr=3e-5)
    optimizer.load_state_dict(torch.load(path/'optimizer.pt', map_location=device, weights_only=False))
    saved = read_blob(path/'state.pkl')
    restore_rng(saved['rng'])
    if saved['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return model, optimizer, saved['state']


def train(partitions, protocol, identity):
    if any(len(rows) < 16 for group in partitions.values() for rows in group.values()):
        record(WORK/'training_result.json', dict(reason='insufficient_target_demonstrations', updates=0, history=[]))
        return None
    torch.set_num_threads(1)
    device = protocol['device']
    random.seed(20261028); np.random.seed(20261028); torch.manual_seed(20261028)
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.5)
    reference = MaskablePPO.load(BC, device=device)
    reference.policy.set_training_mode(False)
    for parameter in reference.policy.parameters():
        parameter.requires_grad_(False)
    latest = WORK/'latest.json'
    if latest.exists():
        pointer = read_record(latest)
        folder = WORK/pointer['path']
        if digest(folder/'manifest.json') != pointer['manifest_sha256']:
            raise RuntimeError('Checkpoint pointer mismatch')
        model, optimizer, state = load_update(folder, device, identity)
    else:
        model = MaskablePPO.load(BC, device=device)
        actor = [p for name,p in model.policy.named_parameters() if name.startswith(('state_encoder.', 'candidate_encoder.', 'score_head.'))]
        optimizer = torch.optim.Adam(actor, lr=3e-5)
        state = dict(epoch=1, next_batch=0, updates=0, history=[dict(epoch=0,
            validation={s:measured(model.policy,rows,device) for s,rows in partitions['validation'].items()})])
        save_update(model,optimizer,state,identity,'initial')
    actor = [p for name,p in model.policy.named_parameters() if name.startswith(('state_encoder.', 'candidate_encoder.', 'score_head.'))]
    teacher_chunks = {s:chunks(partitions['train'][s]) for s in ('ppo','transfer')}
    steps = math.ceil(max(len(partitions['train']['bc']),len(partitions['train']['ppo']))/16)
    baseline = state['history'][0]['validation']
    if len(state['history']) > 1 and not all(state['history'][-1]['checks'].values()):
        return None  # A semantic stop is terminal, not a request for more epochs.
    while state['epoch'] <= 3:
        epoch = state['epoch']
        model.policy.set_training_mode(True)
        for index in range(state['next_batch'], steps):
            bc_rows = random.sample(partitions['train']['bc'],16)
            obs, mask, actions = batch(bc_rows, device)
            current = model.policy.get_distribution(obs, action_masks=mask)
            bc_loss = -current.log_prob(actions).mean()
            with torch.no_grad():
                target = reference.policy.get_distribution(obs, action_masks=mask)
                target_prob, target_log = target.distribution.probs, target.distribution.logits
            kl = (target_prob*(target_log-current.distribution.logits)).sum(dim=1).mean()
            loss = .5*bc_loss+.1*kl
            for source, weight in (('ppo',.4),('transfer',.1)):
                obs,mask,actions = batch(draw_segment(teacher_chunks[source]),device)
                loss = loss-weight*model.policy.get_distribution(obs,action_masks=mask).log_prob(actions).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite supervised loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor,.5)
            optimizer.step()
            state['updates'] += 1
            state['next_batch'] = index+1
            if state['updates'] == 2 or state['updates'] % 128 == 0:
                if any(not torch.equal(v,reference.policy.value_net.state_dict()[k])
                       for k,v in model.policy.value_net.state_dict().items()):
                    raise RuntimeError('Supervision modified frozen Critic')
                guard()
                path = save_update(model,optimizer,state,identity,f'update_{state["updates"]:06d}')
                if state['updates'] == 2:
                    expected = {k:v.detach().clone() for k,v in model.policy.state_dict().items()}
                    model,optimizer,state = load_update(path,device,identity)
                    if any(not torch.equal(v,expected[k]) for k,v in model.policy.state_dict().items()):
                        raise RuntimeError('Real-gradient Save/Load changed model')
                    actor = [p for name,p in model.policy.named_parameters() if name.startswith(('state_encoder.', 'candidate_encoder.', 'score_head.'))]
                    record(WORK/'gradient_save_load.json',dict(passed=True,real_updates=2,
                        optimizer_steps=[int(v['step']) for v in optimizer.state.values()]))
        validation = {s:measured(model.policy,rows,device) for s,rows in partitions['validation'].items()}
        checks = learning_checks(baseline,validation)
        state['history'].append(dict(epoch=epoch,validation=validation,checks=checks))
        state['epoch'],state['next_batch'] = epoch+1,0
        folder = save_update(model,optimizer,state,identity,f'epoch_{epoch:02d}')
        record(WORK/'training_result.json',dict(reason='complete' if all(checks.values()) else 'no_safe_learning_signal',
            updates=state['updates'],history=state['history'],last_model=str(folder/'model.zip')))
        operation(f'supervised epoch {epoch}: updates={state["updates"]}; validation checks={checks}')
        if not all(checks.values()):
            return None
    return folder/'model.zip'


def development_gate(result, long=False, confirm=False):
    a,b = result['summary']['bc'],result['summary']['candidate']
    paired = result['paired_to_bc']['candidate']
    tests = dict(mean=b['mean_pieces'] >= (.95 if long else .99)*a['mean_pieces'],
        p10=b['p10'] >= .9*a['p10'], early=b['early'] <= a['early'] if long else b['early']==0,
        score=paired['common_score_ratio'] >= (1.05 if long else 1.03),
        seeds=paired['positive_seeds'] >= math.ceil(.75*a['games']),
        high=b['high_fraction'] <= a['high_fraction']+.02,
        holes=b['new_holes_per_100'] <= max(1.5*a['new_holes_per_100'], a['new_holes_per_100']+.5))
    if not long:
        tests['capped'] = b['capped'] >= 15
    if confirm:
        tests['ci'] = paired['ratio_ci95'][0] > 1
    return dict(passed=all(tests.values()), checks=tests)


def evaluate(model, seeds, cap, label, identity):
    paths = dict(bc=str(BC),ppo=str(PPO),candidate=str(model))
    names = ('bc','ppo','B12','candidate')
    eval_id = dict(protocol=identity,model_sha256={n:digest(Path(p)) for n,p in paths.items()})
    folder = WORK/'evaluation'/label
    rows = {name:[] for name in names}
    pending = []
    for name in names:
        for seed in seeds:
            path = folder/name/f'seed_{seed}.json'
            if path.exists():
                row = read_record(path)
                evaluation.validate_row(row,name,seed,cap,eval_id)
                rows[name].append(row)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                pending.append((name,seed,path))
    with ProcessPoolExecutor(2,mp_context=get_context('spawn'),initializer=eval_worker_init,initargs=(paths,)) as pool:
        tasks = {pool.submit(evaluation.eval_game,name,seed,cap,eval_id,path.with_suffix('.checkpoint.pkl')):(name,seed,path)
                 for name,seed,path in pending}
        for future in as_completed(tasks):
            name,seed,path = tasks[future]
            row = future.result()
            evaluation.validate_row(row,name,seed,cap,eval_id)
            record(path,row)
            rows[name].append(row)
            status('supervised_evaluation',label=label,completed=sum(map(len,rows.values())),total=len(names)*len(seeds))
            guard()
    rows = {n:sorted(r,key=lambda v:v['seed']) for n,r in rows.items()}
    result = dict(label=label,cap=cap,summary={n:evaluation.summary(r) for n,r in rows.items()},
        paired_to_bc={n:evaluation.paired(rows['bc'],rows[n]) for n in names if n!='bc'})
    record(folder/'summary.json',result)
    return result


def eval_worker_init(paths):
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    evaluation.eval_init(paths)


def main():
    protocol = freeze()
    if (WORK/'outcome.json').exists():
        from training.bc_sequence_supervised_report import close_out
        close_out()
        return
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,affinity[-2:])
    identity,horizon = digest(WORK/'protocol.json'),protocol['length']
    status('supervised_sampling')
    operation('supervised collection started/resumed; 24 episode maximum')
    with ProcessPoolExecutor(2,mp_context=get_context('spawn'),initializer=init_worker,initargs=(identity,horizon)) as pool:
        native = [pool.submit(collect,s,seed) for seed in DATA_SEEDS for s in ('bc','ppo')]
        for i,future in enumerate(as_completed(native),1):
            future.result()
            status('supervised_sampling',completed=i,total=24)
        branches = [pool.submit(collect,'transfer',seed) for seed in DATA_SEEDS]
        for i,future in enumerate(as_completed(branches),17):
            future.result()
            status('supervised_sampling',completed=i,total=24)
    partitions = groups()
    status('supervised_training')
    if protocol['device'] == 'cpu':
        resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    candidate = train(partitions,protocol,identity)
    reason = 'stopped_supervised_learning_signal'
    confirmation = False
    if candidate is not None:
        original = read_record(RUN/'protocol.json')
        for label,seeds,cap,long,confirm in (
            ('development_20k',original['development_seeds'],20000,False,False),
            ('development_100k',original['development_seeds'],100000,True,False),
            ('confirmation_100k',original['confirmation_seeds'],100000,True,True)):
            result = evaluate(candidate,seeds,cap,label,identity)
            gate = development_gate(result,long,confirm)
            record(WORK/f'{label}_gate.json',gate)
            operation(f'{label}: {gate}')
            reason = 'stopped_'+label
            if not gate['passed']:
                break
            if confirm:
                confirmation,reason = True,'confirmed_candidate'
    record(WORK/'outcome.json',dict(reason=reason,confirmation=confirmation,
        candidate=str(candidate) if candidate else None,
        candidate_sha256=digest(candidate) if candidate else None))
    from training.bc_sequence_supervised_report import close_out
    close_out()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('supervised_failed',error=repr(exc))
        operation(f'supervised failure/interruption: {type(exc).__name__}: {exc}')
        raise
