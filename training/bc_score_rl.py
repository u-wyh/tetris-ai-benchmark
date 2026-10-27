"""One frozen BC score-learning route, bounded transactions and paired evaluation."""
import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action, atomic_pickle, _DeterministicAdapter
from training.evaluation.hybrid_queue import BC, PPO, digest, verify_models
from training.hybrid_policy import HybridConfig, HybridController
from training.train_ppo import ROOT, atomic_json, now
from training.train_transaction import read_state
from training.train_vector_transaction import run, verify_task

RUN = ROOT / 'runs/bc_score_rl'
TRAIN = RUN / 'training'
OLD = ROOT / 'runs/candidate_bc_klppo_2m/final/model.zip'
DEV = tuple(range(1561000, 1561016))
CONFIRM = tuple(range(1562000, 1562032))
TRAIN_SEEDS = tuple(range(1560000, 1560004))
MODELS = {}
NAMES = ('bc', 'historical_kl', 'ppo', 'B12', 'candidate')
MAX_TASKS = 732  # 2,998,272 steps, including critic-only warmup.


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def save_record(path, data):
    atomic_json(path, {'data': data, 'sha256': hashlib.sha256(canonical(data)).hexdigest()})


def load_record(path):
    envelope = json.loads(path.read_text())
    if hashlib.sha256(canonical(envelope['data'])).hexdigest() != envelope['sha256']:
        raise RuntimeError(f'Record checksum mismatch: {path}')
    return envelope['data']


def status(stage, **values):
    atomic_json(RUN / 'status.json', dict(stage=stage, at=now(), pid=os.getpid(), **values))


def operation(message):
    with (ROOT / 'OPERATIONS.md').open('a') as stream:
        stream.write(f'{now()} | BC score RL: {message}\n')


def resource_guard():
    if shutil.disk_usage(RUN).free < 8 * 1024**3:
        raise RuntimeError('Less than 8 GiB free; stop safely')
    if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()) > 6 * 1024**3:
        raise RuntimeError('Research disk budget exceeded 6 GiB')
    if time.time() - json.loads((RUN / 'protocol.json').read_text())['started_epoch'] > 48 * 3600:
        raise RuntimeError('Research wall budget exceeded 48 hours')


def preset(device):
    return {'target_total_steps': MAX_TASKS * 4096, 'task_steps':4096, 'device':device,
            'n_envs':4, 'n_steps':1024, 'base_seed':TRAIN_SEEDS[0], 'candidate_policy':True,
            'fast_candidate_env':True, 'max_pieces':100000, 'overnight_monitor':True,
            'step0_baseline':False, 'transaction_retention':3,
            'milestone_interval':100000000, 'validation_interval':100000000,
            'final_test_seeds':100, 'final_test_max_pieces':50000,
            'reward_profile':'bc_score_v1',
            'ppo_overrides':{'learning_rate':3e-5, 'n_epochs':3, 'clip_range':0.1,
                             'ent_coef':0.0, 'gamma':0.995, 'gae_lambda':0.95, 'target_kl':0.01},
            'initial_model':{'path':str(BC), 'sha256':digest(BC)},
            'kl_reference':{'path':str(BC), 'sha256':digest(BC), 'beta':0.1}}


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    verify_models()
    path = RUN / 'protocol.json'
    sources = ('training/bc_score_rl.py', 'training/kl_maskable_ppo.py',
               'training/train_vector_transaction.py', 'training/env/tetris_env.py',
               'training/evaluation/failure_curriculum_eval.py', 'training/evaluation/fast_legal.py',
               'training/evaluation/fast_candidate_features.py', 'training/hybrid_policy.py')
    hashes = {name:digest(ROOT / name) for name in sources}
    if path.exists():
        protocol = json.loads(path.read_text())
        if protocol['sources'] != hashes or protocol['model_hashes'] != {n:digest(p) for n,p in
                (('bc',BC), ('ppo',PPO), ('historical_kl',OLD))}:
            raise RuntimeError('Frozen source or model changed; refusing resume')
        if json.loads((RUN / 'preset.json').read_text()) != protocol['preset'] or digest(RUN / 'seed_audit.json') != protocol['seed_audit_sha256']:
            raise RuntimeError('Frozen preset or seed audit changed')
        return protocol
    if not (RUN / 'seed_audit.json').exists():
        raise RuntimeError('Seed history audit is required before freeze')
    torch.set_num_threads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Loading the historical artifact checks compatibility, not old optimizer recovery.
    for model_path in (BC, OLD, PPO):
        model = MaskablePPO.load(model_path, device='cpu')
        if model.policy.__class__.__name__ != 'CandidateScoringPolicy' or model.action_space.n != 1840:
            raise RuntimeError('Policy protocol mismatch')
        if model.observation_space['state'].shape != (237,) or model.observation_space['candidate'].shape != (1840,16):
            raise RuntimeError('Observation protocol mismatch')
    protocol = {'version':'bc-score-rl-v1', 'started_epoch':time.time(), 'at':now(),
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'sources':hashes, 'model_hashes':{n:digest(p) for n,p in
                    (('bc',BC), ('ppo',PPO), ('historical_kl',OLD))},
                'training_seeds':TRAIN_SEEDS, 'development_seeds':DEV,
                'confirmation_seeds':CONFIRM, 'stage_tasks':[32,244,732],
                'max_steps':MAX_TASKS*4096, 'preset':preset(device),
                'seed_audit_sha256':digest(RUN / 'seed_audit.json')}
    atomic_json(path, protocol)
    atomic_json(RUN / 'preset.json', protocol['preset'])
    operation('protocol frozen; CPU/GPU device='+device+'; no old experiment restarted.')
    return protocol


def train_to(target):
    resource_guard()
    if target > MAX_TASKS:
        raise RuntimeError('Training budget exceeded')
    current = read_state(TRAIN)['committed_task'] if (TRAIN / 'transaction_state.json').exists() else 0
    if current < target:
        status('training', target_tasks=target, committed_tasks=current)
        operation(f'training transactions {current+1}..{target} started')
        run(TRAIN, resume=current>0, max_tasks=target-current, config_file=str(RUN / 'preset.json'))
    state = read_state(TRAIN)
    if state['committed_task'] != target:
        raise RuntimeError('Training did not reach a complete target transaction')
    folder = TRAIN / f'committed/task_{target:06d}'
    verify_task(folder, target, json.loads((TRAIN / 'config.json').read_text()))
    archive = RUN / 'checkpoints' / f'task_{target:06d}'
    if not archive.exists():
        temporary = archive.with_name(archive.name+'.tmp')
        if temporary.exists():
            raise RuntimeError('Incomplete checkpoint archive requires inspection')
        shutil.copytree(folder, temporary)
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, archive)
    verify_task(archive, target, json.loads((TRAIN / 'config.json').read_text()))
    return archive / 'model.zip'


def eval_init(paths):
    global MODELS
    torch.set_num_threads(1)
    MODELS = {n:MaskablePPO.load(p, device='cpu') for n,p in paths.items()}


def eval_game(name, seed, cap, identity, checkpoint):
    env = LongCandidateTetrisEnv(max_pieces=cap)
    control = HybridController(_DeterministicAdapter(MODELS['ppo']), _DeterministicAdapter(MODELS['bc']),
                               HybridConfig('B',12,8)) if name == 'B12' else None
    start = time.monotonic()
    state = {'seed':seed,'variant':name,'cap':cap,'identity':identity, 'pieces':0, 'score':0,
             'scores':[0], 'clears_0_to_4':[0]*5, 'new_holes':0, 'high_steps':0,
             'high_streak':0, 'max_high_streak':0, 'inference_seconds':0.0, 'wall_seconds':0.0}
    try:
        env.reset(seed=seed)
        if checkpoint.exists():
            envelope = pickle.loads(checkpoint.read_bytes())
            if hashlib.sha256(envelope['payload']).hexdigest() != envelope['sha256']:
                raise RuntimeError('Evaluation checkpoint checksum mismatch')
            saved = pickle.loads(envelope['payload'])
            state = saved['state']
            if state['identity'] != identity or (state['seed'],state['variant'],state['cap']) != (seed,name,cap):
                raise RuntimeError('Evaluation checkpoint identity mismatch')
            env.set_state(saved['env'])
            if env.pieces != state['pieces'] or len(state['scores']) != env.pieces+1:
                raise RuntimeError('Evaluation checkpoint score length mismatch')
            if control:
                control.active, control.last_switch, control.step = saved['controller']
            start -= state['wall_seconds']
        while True:
            obs, mask = env.get_observation(), env.action_masks()
            if env.pieces % 1000 == 0 and env._placement_cache != env.core.get_legal_placements():
                raise RuntimeError('Accelerated legal actions differ from Core')
            begin = time.perf_counter()
            action = (int(control.decide(obs,mask,env.core.get_public_observation(),env._placement_cache)[0])
                      if control else fast_action(MODELS[name],obs,mask))
            state['inference_seconds'] += time.perf_counter()-begin
            if not mask[action]:
                raise RuntimeError('Illegal evaluation action')
            _,_,over,truncated,info = env.step(action)
            state['pieces'], state['score'] = info['pieces'], info['score']
            state['scores'].append(info['score'])
            state['clears_0_to_4'][info['cleared_lines']] += 1
            state['new_holes'] += info['new_holes']
            high = info['height_after'] >= 17
            state['high_steps'] += int(high)
            state['high_streak'] = state['high_streak']+1 if high else 0
            state['max_high_streak'] = max(state['max_high_streak'],state['high_streak'])
            if over or truncated:
                state.update(game_over=bool(over), capped=bool(truncated), lines=info['lines'],
                             final_height=info['height_after'], final_holes=info['holes_after'],
                             wall_seconds=time.monotonic()-start)
                return state
            if env.pieces % 10000 == 0:
                state['wall_seconds'] = time.monotonic()-start
                payload = pickle.dumps({'state':state,'env':env.get_state(),
                    'controller':(control.active,control.last_switch,control.step) if control else None},protocol=5)
                atomic_pickle(checkpoint, {'payload':payload,'sha256':hashlib.sha256(payload).hexdigest()})
    finally:
        env.close()


def validate_row(row, name, seed, cap, identity):
    if (row['variant'],row['seed'],row['cap'],row['identity']) != (name,seed,cap,identity):
        raise RuntimeError('Evaluation result identity mismatch')
    n = row['pieces']
    if not (0<n<=cap) or len(row['scores']) != n+1 or row['scores'][-1] != row['score']:
        raise RuntimeError('Evaluation result score series mismatch')
    if sum(row['clears_0_to_4']) != n or row['game_over'] == row['capped'] or row['capped'] != (n==cap and not row['game_over']):
        raise RuntimeError('Evaluation termination semantics mismatch')


def summary(rows):
    return {'games':len(rows), 'mean_pieces':statistics.mean(r['pieces'] for r in rows),
            'p10':float(np.percentile([r['pieces'] for r in rows],10)),
            'minimum':min(r['pieces'] for r in rows), 'capped':sum(r['capped'] for r in rows),
            'deaths':sum(r['game_over'] for r in rows), 'early':sum(r['pieces']<10000 for r in rows),
            'mean_score':statistics.mean(r['score'] for r in rows),
            'score_per_1000':1000*sum(r['score'] for r in rows)/sum(r['pieces'] for r in rows),
            'four_fraction':sum(r['clears_0_to_4'][4] for r in rows)/max(1,sum(sum(r['clears_0_to_4'][1:]) for r in rows)),
            'new_holes_per_100':100*sum(r['new_holes'] for r in rows)/sum(r['pieces'] for r in rows),
            'high_fraction':sum(r['high_steps'] for r in rows)/sum(r['pieces'] for r in rows),
            'mean_max_high_streak':statistics.mean(r['max_high_streak'] for r in rows),
            'mean_inference_ms':1000*sum(r['inference_seconds'] for r in rows)/sum(r['pieces'] for r in rows)}


def paired(a,b):
    differences, pairs = [], []
    for left,right in zip(a,b):
        assert left['seed'] == right['seed']
        n = min(left['pieces'],right['pieces'])
        x,y = left['scores'][n],right['scores'][n]
        pairs.append({'seed':left['seed'],'common_pieces':n,'baseline_score':x,'candidate_score':y,
                      'ratio':y/x if x else None,'survival_delta':right['pieces']-left['pieces']})
        differences.append((x,y,right['pieces']-left['pieces']))
    d = np.asarray(differences,float)
    rng = np.random.default_rng(20261027)
    boot = d[rng.integers(0,len(d),(2000,len(d)))].sum(axis=1)
    return {'pairs':pairs, 'common_score_ratio':float(d[:,1].sum()/d[:,0].sum()),
            'positive_seeds':sum(p['ratio']>1 for p in pairs if p['ratio'] is not None),
            'ratio_ci95':np.percentile(boot[:,1]/boot[:,0],[2.5,97.5]).tolist(),
            'survival_difference_ci95':(np.percentile(boot[:,2]/len(d),[2.5,97.5])).tolist()}


def evaluate(model, seeds, cap, label):
    resource_guard()
    paths = {'bc':str(BC),'ppo':str(PPO),'historical_kl':str(OLD),'candidate':str(model)}
    protocol = json.loads((RUN / 'protocol.json').read_text())
    identity = {'model_sha256':{n:digest(Path(p)) for n,p in paths.items()},
                'protocol_sha256':digest(RUN / 'protocol.json'), 'sources':protocol['sources']}
    folder = RUN / 'evaluation' / label
    rows = {n:[] for n in NAMES}
    pending = []
    for name in NAMES:
        for seed in seeds:
            path = folder / name / f'seed_{seed}.json'
            if path.exists():
                row = load_record(path)
                validate_row(row,name,seed,cap,identity)
                rows[name].append(row)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                pending.append((name,seed,path))
    if pending:
        status('evaluation', label=label, pending=len(pending))
        operation(f'{label} paired evaluation started; {len(seeds)} seeds, cap {cap}, pending {len(pending)}')
        with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn'),initializer=eval_init,initargs=(paths,)) as pool:
            tasks = {pool.submit(eval_game,name,seed,cap,identity,path.with_suffix('.checkpoint.pkl')):(name,seed,path)
                     for name,seed,path in pending}
            for future in as_completed(tasks):
                name,seed,path = tasks[future]
                row = future.result()
                validate_row(row,name,seed,cap,identity)
                save_record(path,row)
                rows[name].append(row)
                status('evaluation',label=label,completed=sum(len(v) for v in rows.values()),total=len(seeds)*len(NAMES))
                resource_guard()
    rows = {n:sorted(v,key=lambda r:r['seed']) for n,v in rows.items()}
    result = {'label':label,'cap':cap,'summary':{n:summary(v) for n,v in rows.items()},
              'paired_to_bc':{n:paired(rows['bc'],rows[n]) for n in NAMES if n!='bc'}}
    save_record(folder / 'summary.json',result)
    return result


def gate(result, long=False, confirm=False):
    base,candidate = result['summary']['bc'],result['summary']['candidate']
    pair = result['paired_to_bc']['candidate']
    tests = {'mean_survival':candidate['mean_pieces']>=0.95*base['mean_pieces'],
             'p10':candidate['p10']>=0.9*base['p10'], 'early':candidate['early']<=base['early'],
             'common_score':pair['common_score_ratio']>=(1.05 if long else 1.01),
             'cross_seed':pair['positive_seeds']>=math.ceil(.75*base['games']),
             'high_board':candidate['high_fraction']<=base['high_fraction']+.02,
             'high_streak':candidate['mean_max_high_streak']<=max(1.5*base['mean_max_high_streak'],base['mean_max_high_streak']+50),
             'new_holes':candidate['new_holes_per_100']<=max(1.5*base['new_holes_per_100'],base['new_holes_per_100']+.5)}
    if not long:
        tests['short_survival'] = candidate['mean_pieces']>=.99*base['mean_pieces']
        tests['short_cap'] = candidate['capped']>=base['capped']-1
    if confirm:
        tests['score_ci'] = pair['ratio_ci95'][0]>1.0
    return {'passed':all(tests.values()),'checks':tests}


def finalize(reason, model=None):
    state = read_state(TRAIN) if (TRAIN / 'transaction_state.json').exists() else {'committed_steps':0}
    lines = ['# BC Champion 得分强化学习', '', f'更新时间：{now()}；停止/运行结果：`{reason}`。', '',
             f'实际新增训练步：**{state["committed_steps"]:,}**（包含 Critic 预热）。', '',
             '历史 Raw Reward 的消行部分与正式得分除以等级近似相同。本轮实质区别是重新初始化与新目标匹配的 Critic、前 16,384 步只更新 Critic、死亡惩罚 -20、较小学习率/clip/epoch 以及固定 BC KL 0.1。正式计分和游戏规则不变。', '',
             '历史 KL-PPO 2M checkpoint 已核对加载；旧 20K 平均分 331,629,591，BC 333,592,084；旧 100K 平均存活 85,658 对 BC 99,320.5。旧归档无逐块共同窗口分数，不能声称旧方案提高了效率；本轮重新配对测量。', '']
    if model:
        lines += [f'候选 checkpoint：`{model}`；SHA256：`{digest(model)}`。','']
    for path in sorted((RUN / 'evaluation').glob('*/summary.json')):
        result = load_record(path)
        lines += [f'## {result["label"]}', '', '|模型|平均存活|P10|触顶|<10K|平均正式分|四消比例|', '|---|---:|---:|---:|---:|---:|---:|']
        for n,v in result['summary'].items():
            lines.append(f'|{n}|{v["mean_pieces"]:.1f}|{v["p10"]:.1f}|{v["capped"]}/{v["games"]}|{v["early"]}|{v["mean_score"]:.0f}|{v["four_fraction"]:.2%}|')
        pair=result['paired_to_bc']['candidate']
        lines += ['',f'候选共同存活窗口正式得分比 BC：{pair["common_score_ratio"]:.4f}，seed 配对 Bootstrap 95% 区间 {pair["ratio_ci95"]}；正向 seed {pair["positive_seeds"]}。',
                  '每千块得分、推理时间及风险指标见 runs 中的同名 summary.json。达到上限为右截断。','',
                  '|Seed|共同块数|共同窗口得分比|候选−BC存活|','|---:|---:|---:|---:|']
        for p in pair['pairs']:
            lines.append(f'|{p["seed"]}|{p["common_pieces"]}|{p["ratio"]:.4f}|{p["survival_delta"]}|')
        lines += ['']
    confirmed = (RUN / 'decision_confirmation.json').exists() and json.loads((RUN / 'decision_confirmation.json').read_text())['passed']
    for decision_path in sorted(RUN.glob('decision_*.json')):
        lines += [f'门槛 `{decision_path.stem}`：`{decision_path.read_text().strip()}`', '']
    lines += [f'独立确认通过：**{bool(confirmed)}**。仅开发结果不足以登记新冠军；未启动的确认组继续保留。', '',
              '恢复：在冻结源码和模型哈希一致时运行 `python -m training.bc_score_rl`；已原子提交事务和对局只校验、不重复计算。语义门槛失败属于正常结束，不应恢复扩大预算。', '',
              '训练遥测：`runs/bc_score_rl/training/metric_history/`；协议、配置、完整逐 seed 配对、错误与测试日志均保存在 `runs/bc_score_rl/`。']
    from training.train_ppo import atomic_bytes
    atomic_bytes(ROOT / 'reports/experiments/bc_score_rl.md', ('\n'.join(lines)+'\n').encode())
    operation(f'finished reason={reason}; committed_steps={state["committed_steps"]}; confirmation={bool(confirmed)}')
    status(reason, committed_steps=state['committed_steps'],model=str(model) if model else None,
           model_sha256=digest(model) if model else None)


def regress_and_publish():
    tests = [[sys.executable,'-m','pytest','-q','tests/test_bc_score_rl.py','tests/test_kl_maskable_ppo.py',
              'tests/test_train_vector_transaction.py','tests/test_tetris_env.py','tests/parity'],
             ['node','--test',*[str(p) for p in (ROOT/'tests').glob('*.test.cjs')]]]
    results=[]
    for index,cmd in enumerate(tests):
        with (RUN / f'regression_{index}.log').open('w') as stream:
            proc=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
        results.append(proc.returncode)
    atomic_json(RUN/'regression.json',{'at':now(),'returncodes':results})
    operation(f'regression return codes {results}; report generated; no web model changes')
    subprocess.run(['git','add','reports/experiments/bc_score_rl.md','OPERATIONS.md','CHANGELOG.md'],cwd=ROOT,check=True)
    if subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m','Report bounded BC score reinforcement learning results'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','codex/bc-score-rl'],cwd=ROOT,check=True)
    if any(results):
        raise RuntimeError('Required regression failed; inspect regression logs')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--pilot-only',action='store_true')
    args=parser.parse_args()
    import fcntl
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN/"program.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze()
    if args.freeze_only:
        return
    if (RUN/'status.json').exists() and (RUN/'regression.json').exists():
        previous = json.loads((RUN/'status.json').read_text())['stage']
        if previous in ('confirmed','stopped_confirmation','stopped_no_learning_or_unstable','stopped_gate_32','stopped_gate_244','stopped_gate_732','failed'):
            print('Terminal experiment state preserved:', previous); return
    model=None
    try:
        # First and second transactions explicitly exercise creation and restore.
        if not (TRAIN/'transaction_state.json').exists():
            train_to(1)
        if read_state(TRAIN)['committed_task']<2:
            train_to(2)
        if args.pilot_only:
            return
        reason='completed'
        for tasks in (32,244,732):
            saved=RUN/f'decision_{tasks}.json'
            if saved.exists():
                decision=json.loads(saved.read_text())
                if not decision['passed']:
                    reason=f'stopped_gate_{tasks}'; break
                continue
            model=train_to(tasks)
            metrics = json.loads((TRAIN/'metric_history'/f'task_{tasks:06d}.json').read_text())
            if (metrics.get('actor_l2_to_bc',0)<=1e-7 or not math.isfinite(metrics['value_loss'])
                    or metrics['value_loss']>10000 or metrics.get('bc_kl',0)>1.0):
                reason='stopped_no_learning_or_unstable'; break
            short=evaluate(model,DEV,20000,f'task_{tasks}_20k')
            decision=gate(short)
            atomic_json(RUN/f'decision_{tasks}_short.json',decision)
            if decision['passed']:
                long=evaluate(model,DEV,100000,f'task_{tasks}_100k')
                decision=gate(long,long=True)
            atomic_json(saved,decision)
            if not decision['passed']:
                reason=f'stopped_gate_{tasks}'; break
        if reason=='completed':
            model=RUN/'checkpoints/task_000732/model.zip'
            confirmation=evaluate(model,CONFIRM,100000,'confirmation')
            decision=gate(confirmation,long=True,confirm=True)
            atomic_json(RUN/'decision_confirmation.json',decision)
            reason='confirmed' if decision['passed'] else 'stopped_confirmation'
        if model is None:
            n=read_state(TRAIN)['committed_task']
            model=TRAIN/f'committed/task_{n:06d}/model.zip'
        finalize(reason,model)
    except Exception:
        (RUN/'error.log').write_text(traceback.format_exc())
        finalize('failed',model if model and model.exists() else None)
        regress_and_publish()
        raise
    regress_and_publish()


if __name__=='__main__':
    main()
