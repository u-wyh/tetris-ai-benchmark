"""Frozen, resource-bounded paired ablations with atomic per-game recovery."""
import copy
import csv
import fcntl
import urllib.request
import json
import os
import pickle
import re
import resource
import shutil
import signal
import stat
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from training.train_ppo import ROOT, now
from training.bc_sequence_transfer import record, read_record, save_blob, read_blob, rng_state, restore_rng
from training.evaluation.hybrid_queue import BC, PPO, digest, verify_models
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv, fast_action
from training.public_lookahead import PublicLookaheadController, PublicSimulator
from training.lookahead_ablation_controller import AblationController, SPECS as ALL_SPECS, legacy_trace
SPECS={k:v for k,v in ALL_SPECS.items() if k in ("A","B","C","E")}
import training.bc_score_rl as evaluation

RUN=ROOT/'runs/lookahead_long_validation'
STUDENT=ROOT/'runs/bc_sequence_transfer/supervised/checkpoints/epoch_01/model.zip'
PLANNER_SHA='0c2dbea26708fed866c5089eb102672b0cfccafeaa9f130b58086c847e8ffb08'
STUDENT_SHA='e40bfad269566c052d880cb01857f5bd998e22a1730c6a71d75845387f605914'
GROUPS={'functional':[3102861000,3102861001], 'main':list(range(3102862000,3102862032)), 'extension':list(range(3102862000,3102862016))}
MODELS={}; IDENTITY=None
CHECKPOINT_INTERVAL=1000
MAIN_CAP=100000
EXT_CAP=200000
TERMINALS={'playing':0,'unknown_frontier':1,'over':2,'no_mechanical_action':3}
CODE_FIELDS=['student','bc','geometry','instant_clear','selected','nominal_depth','unique_roots','override','third']
ROOT_CODE_FIELDS=['action','actual_depth','known_next_consumed','stop_code']
COST_FIELDS=['immediate_geometry','immediate_reward','immediate_proxy','final_geometry','max_geometry','score_contribution','death_penalty','value']

class BudgetStop(RuntimeError):pass

def operation(message):
    with (ROOT/'OPERATIONS.md').open('a') as f:f.write(f'{now()} | Lookahead long validation: {message}\n')
def status(stage,**kw):record(RUN/'status.json',dict(at=now(),pid=os.getpid(),stage=stage,**kw))
def disk_bytes():
    total=0
    for p in RUN.rglob('*'):
        try:s=p.stat()
        except FileNotFoundError:continue
        if stat.S_ISREG(s.st_mode):total+=s.st_size
    return total

def check_services():
    with (RUN/'health.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        previous=read_record(RUN/'health.json') if (RUN/'health.json').exists() else {}
        if time.time()-previous.get('epoch',0)<300:return
        health={}
        for port,suffix in ((8080,'/'),(8081,'/api/health')):
            try:
                with urllib.request.urlopen(f'http://100.91.227.47:{port}{suffix}',timeout=3) as r:health[str(port)]=r.status
            except Exception as exc:health[str(port)]=repr(exc)
        failures=0 if all(v==200 for v in health.values()) else previous.get('consecutive_failures',0)+1
        entry=dict(at=now(),epoch=time.time(),services=health,consecutive_failures=failures)
        record(RUN/'health.json',entry)
        with (RUN/'health_history.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
        if failures>=2:raise BudgetStop('existing service health failed twice; release research resources')

def guard():
    p=read_record(RUN/'protocol.json')
    if time.time()-p['started']>=35.5*3600:raise BudgetStop('35.5h execution stop; reserve30min closeout')
    if disk_bytes()>=5.5*1024**3:raise BudgetStop('5.5GiB retained data stop; reserve0.5GiB closeout')
    if shutil.disk_usage(RUN).free<8*1024**3:raise BudgetStop('free disk below8GiB')
    if (RUN/'stop.json').exists():raise BudgetStop('coordinator stop marker')
    todo=[read_record(RUN/'runtime.json')['pid']];seen=set();rss=0
    while todo:
        pid=todo.pop()
        if pid in seen:continue
        seen.add(pid)
        try:
            rss+=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            todo+=list(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
        except FileNotFoundError:pass
    if rss>10*1024**3:raise BudgetStop('process tree RSS above10GiB')
    available=re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())
    if available and int(available[1])*1024<2*1024**3:raise BudgetStop('system available memory below2GiB')
    check_services()

def seed_audit():
    trees=subprocess.check_output(['git','worktree','list','--porcelain'],cwd=ROOT,text=True)
    roots=sorted({str((Path(line[9:])/d).resolve()) for line in trees.splitlines() if line.startswith('worktree ') for d in ('runs','reports','training','ops','OPERATIONS.md','README.md','CHANGELOG.md') if (Path(line[9:])/d).exists()})
    #Named seed fields/list metadata and seed filenames, not unrelated formal score integers.
    expr=r'(?i)"(?:[a-z_]*seed[s]?|seeds[a-z_]*)"\s*:\s*(?:\[[^\]]{0,20000}\]|[0-9]+)'
    target=RUN/'historical_seed_fields.txt'
    with target.open('w') as f:
        scan=subprocess.run(['rg','-U','-o','--hidden','--no-heading','-g','*.json','-g','*.jsonl',expr,*roots],stdout=f,stderr=subprocess.PIPE,text=True)
    if scan.returncode not in (0,1):raise RuntimeError('Historical metadata audit failed:'+scan.stderr[:300])
    found=set();text=target.read_text()
    for m in re.finditer(r'(?i)"(?:[a-z_]*seed[s]?|seeds[a-z_]*)"\s*:\s*(\[[^\]]*\]|[0-9]+)',text):
        found.update(int(n) for n in re.findall(r'\d+',m.group(1)) if int(n)<=0xffffffff)
    csv_files=set()
    for base in roots:
        for path in Path(base).rglob('*.csv'):
            resolved=path.resolve()
            if resolved in csv_files or path.is_relative_to(RUN):continue
            csv_files.add(resolved)
            with path.open(errors='replace',newline='') as stream:
                reader=csv.DictReader(stream)
                keys=[k for k in (reader.fieldnames or []) if k and 'seed' in k.lower()]
                if not keys:continue
                for row in reader:
                    for k in keys:
                        for n in re.findall(r'\d+',row.get(k) or ''):
                            if int(n)<=0xffffffff:found.add(int(n))
    names=[]
    for base in roots:
        if Path(base).name!='runs':continue
        for path in Path(base).rglob('*seed*'):
            if path.is_relative_to(RUN):continue
            for n in re.findall(r'seed[_-](\d+)',path.name):found.add(int(n));names.append(str(path))
    proposed={n for group in GROUPS.values() for n in group}
    #Also reject references to the new family in historical source/reports/reservations.
    own=[str(ROOT/'training/lookahead_long_validation.py'),str(ROOT/'reports/experiments/lookahead_long_validation_design.md')]
    refs=subprocess.run(['rg','-l','--hidden','-g','*.py','-g','*.md','-g','*.sh',r'310286[12][0-9]{3}',*roots],capture_output=True,text=True)
    if refs.returncode not in (0,1):raise RuntimeError('Historical source audit failed')
    conflicts=sorted(proposed&found);source_conflicts=[p for p in refs.stdout.splitlines() if p not in own and '/lookahead_long_validation/' not in p]
    audit=dict(at=now(),roots=roots,named_seed_count=len(found),named_seed_ids=sorted(found),filename_count=len(names),
               metadata_sha256=digest(target),csv_file_count=len(csv_files),proposed=GROUPS,conflicts=conflicts,source_conflicts=source_conflicts,
               method='all registered worktree JSON/JSONL named seed scalars/lists, seed filenames, source/report new-family reservations; binary contents rely on associated metadata/filenames')
    record(RUN/'seed_audit.json',audit)
    if conflicts or source_conflicts:raise RuntimeError('New seed collision:'+str((conflicts,source_conflicts)))
    return digest(RUN/'seed_audit.json')

def freeze():
    RUN.mkdir(parents=True,exist_ok=True);verify_models()
    if digest(STUDENT)!=STUDENT_SHA or digest(ROOT/'training/public_lookahead.py')!=PLANNER_SHA:raise RuntimeError('Frozen student/planner mismatch')
    own=['training/lookahead_long_validation.py','training/lookahead_ablation_controller.py','training/lookahead_ablation.py','training/lookahead_ablation_report.py','training/lookahead_long_validation_report.py','tests/test_lookahead_long_validation.py','ops/run_lookahead_long_validation.sh','reports/experiments/lookahead_long_validation_design.md']
    frozen=['training/public_lookahead.py','training/web_inference.py','training/candidate_features.py','training/evaluation/fast_legal.py','training/evaluation/fast_candidate_features.py','training/evaluation/failure_curriculum_eval.py','training/bc_score_rl.py','training/bc_sequence_transfer.py']
    frozen += [str(p) for p in Path('training/tetris_core').rglob('*.py')]+[str(p) for p in Path('training/env').glob('*.py')]+[str(p) for p in Path('tetris-game').glob('*') if p.is_file()]
    sources={f:digest(ROOT/f) for f in own+frozen};models={str(p.relative_to(ROOT)):digest(p) for p in (BC,STUDENT,PPO)}
    history={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'reports/experiments').glob('*') if p.is_file() and not p.name.startswith('lookahead_long_validation_')}
    if (RUN/'protocol.json').exists():
        p=read_record(RUN/'protocol.json')
        if (p['sources'],p['models'],p['history'])!=(sources,models,history):raise RuntimeError('Source/model/history freeze mismatch')
        return p
    if subprocess.run(['git','diff','--quiet','--',*own,*frozen],cwd=ROOT).returncode:raise RuntimeError('Commit source/design before freezing')
    audit=seed_audit();started=json.loads((RUN/'task_started.json').read_text())['epoch']
    p=dict(at=now(),started=started,git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=sources,models=models,history=history,groups=GROUPS,specs=SPECS,override_margin=12,bootstrap_seed=20261031,bootstrap_samples=5000,workers=4,device='cpu',cpu_affinity=sorted(os.sched_getaffinity(0))[-4:],seed_audit_sha256=audit,
           main_cap=MAIN_CAP,extension_cap=EXT_CAP,extension_parent='exact100K endpoint; never replay; dead parents retain observed death',admission='complete32main and16extension; C life>=E, deaths<=E, common score>=.99E, pooledP95<=E; no extension opposition',extension_opposition='C life<.95E OR C deaths>=E+2 OR C early>E OR paired survivalCIupper<0',confirmation_recommendation='admission plus survivalCIlower>0 OR (admission and seed-bootstrap mean episodeP95 ratioCIupper<1 and commonScore ratioCIlower>=.99); no automatic confirmation',code_fields=CODE_FIELDS,root_code_fields=ROOT_CODE_FIELDS,cost_fields=COST_FIELDS,terminal_codes=TERMINALS,
           rich_trace_sampling='piece index divisible128 plus first64 overrides/episode; all decisions numeric component arrays',long_priority=['main32_A_B_C_E_BC','extension16_BC_C_E'],budget_wall_seconds=129600,execution_stop_seconds=127800,retained_stop_bytes=int(5.5*1024**3))
    record(RUN/'protocol.json',p);operation('protocol frozen before results; A/B/C/E/BC100K32 seeds, presetBC/C/E200K16 continuations; immutable models/rules; no deployment');return p

def init(identity):
    global MODELS,IDENTITY
    # Stop orphan workers before any bounded coordinator restart.
    import ctypes
    parent=os.getppid();ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:os.kill(os.getpid(),signal.SIGTERM)
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));torch.set_num_threads(1)
    MODELS={'A':MaskablePPO.load(STUDENT,device='cpu'),'BC':MaskablePPO.load(BC,device='cpu')};IDENTITY=identity

def validate_rollouts(core,trace):
    checks=frontiers=0
    for branch in trace['alternatives']:
        real=copy.deepcopy(core);sim=PublicSimulator(core.get_public_observation())
        for action in branch['path']:
            real.step(action);pred=sim.step(action);actual=real.get_public_observation()
            for k in ('board','score','lines','level','hold'):
                if pred[k]!=actual[k]:raise RuntimeError('Simulation parity:'+k)
            if sim.phase!='unknown_frontier':
                for k in ('currentPiece','gameOver','phase','holdAvailable'):
                    if pred[k]!=actual[k]:raise RuntimeError('Known spawn parity:'+k)
            else:frontiers+=1
            if actual['next'][:len(pred['next'])]!=pred['next']:raise RuntimeError('Hold/Next prefix parity')
            checks+=1
    return checks,frontiers

def functional():
    init(digest(RUN/'protocol.json'));old=PublicLookaheadController(MODELS['A'],MODELS['BC'])
    ctrls={n:AblationController(MODELS['A'],MODELS['BC'],n) for n in SPECS};rows=[];checks=frontiers=hidden=equal=0
    for seed in GROUPS['functional']:
        env=LongCandidateTetrisEnv(max_pieces=128);env.reset(seed=seed)
        for step in range(128):
            obs,mask=env.get_observation(),env.action_masks();pub=env.core.get_public_observation();before=pickle.dumps(env.core)
            outputs={n:c.decide(obs,mask,pub) for n,c in ctrls.items()};a,t=old.decide(obs,mask,pub)
            if outputs['E'][0]!=a or legacy_trace(outputs['E'][1])!=t:raise RuntimeError('Full E differs from frozen original')
            if outputs['A'][0]!=t['student_action']:raise RuntimeError('A differs from frozen student')
            equal+=1
            for n,(action,trace) in outputs.items():
                if not mask[action]:raise RuntimeError('Functional illegal action')
                c,f=validate_rollouts(env.core,trace);checks+=c;frontiers+=f
                if step%16==0:
                    altered=copy.deepcopy(env.core);altered.rng.state^=0x12345678;altered.bag=list(reversed(altered.bag));altered.queue[3:]=list('II')
                    if (action,trace)!=ctrls[n].decide(obs,mask,altered.get_public_observation()):raise RuntimeError('Hidden-state dependence or nondeterminism')
                    hidden+=1
            if pickle.dumps(env.core)!=before:raise RuntimeError('Planner mutated real Core')
            rows.append(dict(seed=seed,piece=step,public=pub,actions={n:a for n,(a,_) in outputs.items()},traces={n:t for n,(_,t) in outputs.items()}))
            _,_,over,capped,_=env.step(outputs['A'][0])
            if over or capped:break
        env.close();guard()
    r=dict(passed=True,states=len(rows),full_equivalence_states=equal,simulation_checks=checks,unknown_frontiers=frontiers,hidden_checks=hidden,rows=rows)
    record(RUN/'functional.json',r);operation(f'functional passed: E/original exact{equal} states, simulation{checks}, hidden{hidden}');return r

def arrays(cap):
    return dict(scores=np.zeros(cap+1,np.int64),lines=np.zeros(cap+1,np.int32),latencies_ms=np.zeros(cap,np.float64),codes=np.full((cap,9),-1,np.int16),root_codes=np.full((cap,4,4),-1,np.int16),costs=np.full((cap,4,8),np.nan,np.float64))

def compact(arr,trace,action,index):
    if trace is None:
        arr['codes'][index]=[-1,-1,-1,-1,action,0,0,0,0];return
    arr['codes'][index]=[trace['student_action'],trace['bc_action'],trace['geometry_action'],trace['clear_action'],action,trace['nominal_depth'],len(trace['roots']),int(trace['override']),int(action not in (trace['student_action'],trace['bc_action']))]
    for j,b in enumerate(trace['alternatives']):
        arr['root_codes'][index,j]=[b['action'],len(b['path']),b['known_tokens_consumed'],TERMINALS[b['terminal']]]
        arr['costs'][index,j]=[b[k] for k in COST_FIELDS]

def completed(path,name,seed,cap):
    r=read_record(path)
    if (r['variant'],r['seed'],r['cap'],r['identity'])!=(name,seed,cap,IDENTITY):raise RuntimeError('Game identity mismatch')
    if digest(path.with_suffix('.end.pkl'))!=r['snapshot_sha256']:raise RuntimeError('Endpoint snapshot checksum mismatch')
    npz=path.with_suffix('.npz')
    if digest(npz)!=r['artifact_sha256']:raise RuntimeError('Game NPZ checksum mismatch')
    with np.load(npz) as a:
        data={k:a[k] for k in ('scores','latencies_ms')}
    evaluation.validate_row(dict(r,scores=data['scores']),name,seed,cap,IDENTITY)
    if len(data['latencies_ms'])!=r['pieces']:raise RuntimeError('Latency series mismatch')
    return r


def finish_game(path,cp,env,s,sliced,over,capped,info,name,seed,cap):
    s.update(game_over=bool(over),capped=bool(capped),lines=info['lines'],final_height=info['height_after'],final_holes=info['holes_after'])
    temp=path.with_suffix('.tmp.npz')
    with temp.open('wb') as f:np.savez_compressed(f,**sliced);f.flush();os.fsync(f.fileno())
    os.replace(temp,path.with_suffix('.npz'));s['artifact_sha256']=digest(path.with_suffix('.npz'))
    evaluation.validate_row(dict(s,scores=sliced['scores']),name,seed,cap,IDENTITY)
    snapshot=path.with_suffix('.end.pkl')
    save_blob(snapshot,dict(identity=IDENTITY,state=s,env=env.get_state(),rng=rng_state(),artifact_sha256=s['artifact_sha256']))
    s['snapshot_sha256']=digest(snapshot);record(path,s)
    if cp.exists():cp.unlink()
    env.close();return str(path)

def continuation_state(saved,cap):
    state=copy.deepcopy(saved)
    if state['core'].phase!='playing':raise RuntimeError('Cannot continue a dead Core')
    state['max_pieces']=cap;state['finished']=False
    return state

def game(name,seed,cap,label):
    folder=RUN/'evaluation'/label/name;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'seed_{seed}.json';cp=path.with_suffix('.checkpoint.pkl')
    if path.exists():completed(path,name,seed,cap);return str(path)
    env=LongCandidateTetrisEnv(max_pieces=cap);env.reset(seed=seed);arr=arrays(cap)
    controller=AblationController(MODELS['A'],MODELS['BC'],name) if name!='BC' else None
    s=dict(identity=IDENTITY,variant=name,seed=seed,cap=cap,pieces=0,score=0,clears_0_to_4=[0]*5,new_holes=0,high_steps=0,high_streak=0,max_high_streak=0,height_sum=0,holes_sum=0,inference_seconds=0.,wall_seconds=0.,overrides=0,third_actions=0,root_membership_counts={},stop_counts={},next_consumed_counts={},cost_component_sum={k:0. for k in COST_FIELDS},trace_override_samples=0,traces=[])
    if not cp.exists() and label=='200k':
        parent=RUN/'evaluation'/'100k'/name/f'seed_{seed}.json'
        base=completed(parent,name,seed,MAIN_CAP);saved=read_blob(parent.with_suffix('.end.pkl'))
        s=copy.deepcopy(saved['state']);s['cap']=cap
        for key in ('artifact_sha256','snapshot_sha256','game_over','capped'):s.pop(key,None)
        s.update(parent_100k_result_sha256=digest(parent),continuation_start_pieces=base['pieces'],already_dead_at_100k=base['game_over'])
        state=copy.deepcopy(saved['env'])
        if not base['game_over']:state=continuation_state(state,cap)
        else:state['max_pieces']=cap
        env.set_state(state);restore_rng(saved['rng'])
        with np.load(parent.with_suffix('.npz')) as data:
            for key in arr:arr[key][:len(data[key])]=data[key]
        save_blob(cp,dict(state=s,env=env.get_state(),rng=rng_state(),arrays={k:v[:env.pieces+1 if k in ('scores','lines') else env.pieces].copy() for k,v in arr.items()}))
    if cp.exists():
        saved=read_blob(cp);s=saved['state']
        if (s['identity'],s['variant'],s['seed'],s['cap'])!=(IDENTITY,name,seed,cap):raise RuntimeError('Checkpoint identity mismatch')
        env.set_state(saved['env']);restore_rng(saved['rng'])
        for k,v in saved['arrays'].items():arr[k][:len(v)]=v
        if env.pieces!=s['pieces'] or arr['scores'][env.pieces]!=s['score']:raise RuntimeError('Resume progress mismatch')
    if label=='200k':
        parent=RUN/'evaluation'/'100k'/name/f'seed_{seed}.json';base=completed(parent,name,seed,MAIN_CAP)
        if s['parent_100k_result_sha256']!=digest(parent):raise RuntimeError('Continuation parent changed')
        with np.load(parent.with_suffix('.npz')) as data:
            for key in arr:
                if not np.array_equal(arr[key][:len(data[key])],data[key],equal_nan=True):raise RuntimeError('Continuation prefix differs:'+key)
        if base['game_over']:
            n=env.pieces;sliced={k:v[:n+1 if k in ('scores','lines') else n] for k,v in arr.items()}
            info=dict(lines=base['lines'],height_after=base['final_height'],holes_after=base['final_holes'])
            return finish_game(path,cp,env,s,sliced,True,False,info,name,seed,cap)
    start=time.monotonic()-s['wall_seconds']
    while True:
        guard_due=env.pieces%CHECKPOINT_INTERVAL==0
        if guard_due:guard()
        begin=time.perf_counter();env._candidate_cache=None;obs,mask=env.get_observation(),env.action_masks();public=env.core.get_public_observation()
        action,trace=controller.decide(obs,mask,public) if controller else (fast_action(MODELS['BC'],obs,mask),None)
        ms=1000*(time.perf_counter()-begin);i=env.pieces;arr['latencies_ms'][i]=ms;s['inference_seconds']+=ms/1000
        if not mask[action]:raise RuntimeError('Illegal real action')
        if guard_due and env._placement_cache!=env.core.get_legal_placements():raise RuntimeError('Official fast mask parity')
        compact(arr,trace,action,i)
        if trace and name!='A':
            s['overrides']+=int(trace['override']);s['third_actions']+=int(action not in (trace['student_action'],trace['bc_action']))
            membership='+'.join(k for k,key in [('student','student_action'),('bc','bc_action'),('geometry','geometry_action'),('clear','clear_action')] if trace[key]==action)
            s['root_membership_counts'][membership]=s['root_membership_counts'].get(membership,0)+1
            b=trace['selected'];stop=b['stop_reason'];s['stop_counts'][stop]=s['stop_counts'].get(stop,0)+1
            consumed=str(b['known_tokens_consumed']);s['next_consumed_counts'][consumed]=s['next_consumed_counts'].get(consumed,0)+1
            for k in COST_FIELDS:s['cost_component_sum'][k]+=b[k]
        if trace and (i%128==0 or (trace['override'] and s['trace_override_samples']<64)):
            s['traces'].append(dict(piece=i,public=public,trace=trace))
            if trace['override']:s['trace_override_samples']+=1
        _,_,over,capped,info=env.step(int(action));n=env.pieces;s['pieces']=n;s['score']=info['score'];arr['scores'][n]=info['score'];arr['lines'][n]=info['lines'];s['clears_0_to_4'][info['cleared_lines']]+=1;s['new_holes']+=info['new_holes'];s['height_sum']+=info['height_after'];s['holes_sum']+=info['holes_after']
        high=info['height_after']>=17;s['high_steps']+=int(high);s['high_streak']=s['high_streak']+1 if high else 0;s['max_high_streak']=max(s['max_high_streak'],s['high_streak'])
        s['wall_seconds']=time.monotonic()-start
        sliced={k:v[:n+1 if k in ('scores','lines') else n].copy() for k,v in arr.items()} if over or capped or n%CHECKPOINT_INTERVAL==0 else None
        if over or capped:return finish_game(path,cp,env,s,sliced,over,capped,info,name,seed,cap)
        if n%CHECKPOINT_INTERVAL==0:save_blob(cp,dict(state=s,env=env.get_state(),rng=rng_state(),arrays=sliced))

def stage(label,variants,seeds,cap):
    status('evaluation',label=label,completed=0,total=len(variants)*len(seeds))
    with ProcessPoolExecutor(4,mp_context=get_context('spawn'),initializer=init,initargs=(digest(RUN/'protocol.json'),)) as pool:
        tasks=[pool.submit(game,name,seed,cap,'100k' if cap==MAIN_CAP else '200k') for seed in seeds for name in variants]
        try:
            for i,f in enumerate(as_completed(tasks),1):
                f.result();status('evaluation',label=label,completed=i,total=len(tasks));guard()
        except BaseException as exc:
            record(RUN/'stop.json',dict(at=now(),error=repr(exc)))
            pool.shutdown(wait=False,cancel_futures=True)
            raise
    operation(f'{label} completed all{len(tasks)} games; immutable configs')

def main():
    global IDENTITY
    p=freeze();IDENTITY=digest(RUN/'protocol.json');record(RUN/'runtime.json',dict(pid=os.getpid(),at=now()));os.sched_setaffinity(0,p['cpu_affinity'])
    if (RUN/'closed.json').exists():return
    operation('fixed-budget tmux queue started/resumed; source/model identities checked')
    if not (RUN/'functional.json').exists():status('functional');functional()
    else:
        if not read_record(RUN/'functional.json')['passed']:raise RuntimeError('Previous functional failure')
    stage('main32',['A','B','C','E','BC'],GROUPS['main'],MAIN_CAP)
    stage('extension16',['BC','C','E'],GROUPS['extension'],EXT_CAP)
    from training.lookahead_long_validation_report import finish
    finish('completed')

if __name__=='__main__':
    def interrupted(*_):raise BudgetStop('external wall-clock stop')
    signal.signal(signal.SIGTERM,interrupted)
    try:main()
    except BaseException as exc:
        RUN.mkdir(parents=True,exist_ok=True);record(RUN/'stop.json',dict(at=now(),error=repr(exc)));status('stopping',error=repr(exc));operation(f'stopped:{exc!r}; completed results and checkpoints retained')
        if (RUN/'protocol.json').exists():
            from training.lookahead_long_validation_report import finish
            finish('resource_stop' if isinstance(exc,BudgetStop) else 'technical_failure',repr(exc))
        else:raise
