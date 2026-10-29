"""One bounded public-information planning experiment, no policy training."""
import copy
import os
import pickle
import resource
import shutil
import signal
import stat
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO

from training.train_ppo import ROOT,now
from training.bc_sequence_transfer import record,read_record,save_blob,read_blob,rng_state,restore_rng
from training.bc_sequence_safety import STUDENT
from training.evaluation.hybrid_queue import BC,PPO,digest,verify_models
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv,fast_action,_DeterministicAdapter
from training.hybrid_policy import HybridController,HybridConfig
from training.public_lookahead import PublicLookaheadController,PublicSimulator,PlanningConfig
import training.bc_score_rl as evaluation

RUN=ROOT/'runs/model_based_planning'
MODELS={}
IDENTITY=None

def operation(message):
    with (ROOT/'OPERATIONS.md').open('a') as f:f.write(f'{now()} | Model based planning: {message}\n')

def status(stage,**kw):record(RUN/'status.json',dict(at=now(),pid=os.getpid(),stage=stage,**kw))

def disk_bytes():
    total=0
    for p in RUN.rglob('*'):
        try:s=p.stat()
        except FileNotFoundError:continue
        if stat.S_ISREG(s.st_mode):total+=s.st_size
    return total

def guard():
    protocol=read_record(RUN/'protocol.json')
    if time.time()-protocol['started']>86400:raise RuntimeError('Budget:24h wall clock')
    if disk_bytes()>5.5*1024**3:raise RuntimeError('Budget:disk closing reserve reached')
    if shutil.disk_usage(RUN).free<8*1024**3:raise RuntimeError('Budget:free disk below8GiB')
    todo=[read_record(RUN/'runtime.json')['pid']];rss=0;seen=set()
    while todo:
        pid=todo.pop()
        if pid in seen:continue
        seen.add(pid)
        try:
            rss+=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            todo+=list(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
        except FileNotFoundError:pass
    if rss>10*1024**3:raise RuntimeError('Budget:process tree RSS above10GiB')

def freeze():
    RUN.mkdir(parents=True,exist_ok=True);verify_models()
    if digest(STUDENT)!='e40bfad269566c052d880cb01857f5bd998e22a1730c6a71d75845387f605914':raise RuntimeError('Student identity mismatch')
    files=['training/public_lookahead.py','training/model_based_planning.py','training/model_based_planning_report.py','tests/test_public_lookahead.py','tests/test_model_based_planning.py','reports/experiments/model_based_planning_design.md','ops/run_model_based_planning.sh',
        'training/evaluation/fast_legal.py','training/evaluation/fast_candidate_features.py','training/evaluation/failure_curriculum_eval.py','training/bc_score_rl.py','training/candidate_features.py','training/hybrid_policy.py']
    files+=list(map(str,Path('training/tetris_core').rglob('*.py')))+list(map(str,Path('training/env').glob('*.py')))+list(map(str,Path('tetris-game').glob('*.js')))
    sources={f:digest(ROOT/f) for f in files};models={str(p.relative_to(ROOT)):digest(p) for p in (BC,PPO,STUDENT,ROOT/'runs/bc_sequence_safety/checkpoints/epoch_01/model.zip')}
    history={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'reports/experiments').glob('*.md') if not p.name.startswith('model_based_planning_')}
    path=RUN/'protocol.json'
    if path.exists():
        p=read_record(path)
        if (p['sources'],p['models'],p['history'])!=(sources,models,history):raise RuntimeError('Frozen experiment/source/history identity mismatch')
        return p
    if subprocess.run(['git','diff','--quiet','--',*files],cwd=ROOT).returncode:raise RuntimeError('Commit before freezing')
    trees=subprocess.check_output(['git','worktree','list','--porcelain'],text=True,cwd=ROOT)
    roots=sorted({str(Path(line[9:])/d) for line in trees.splitlines() if line.startswith('worktree ') for d in ('runs','reports','training','ops') if (Path(line[9:])/d).exists()})
    groups=dict(functional_seeds=[3102831000,3102831001],short_seeds=list(range(3102832000,3102832016)),long_seeds=list(range(3102833000,3102833016)),confirm_seeds=list(range(3102835000,3102835032)))
    pattern=r'\b(?:'+'|'.join(str(seed) for values in groups.values() for seed in values)+r')\b'
    scan=subprocess.run(['rg','-l','--hidden','-g','*.md','-g','*.json','-g','*.jsonl','-g','*.py','-g','*.sh','-g','*.csv',pattern,*roots],capture_output=True,text=True)
    own={str(ROOT/f) for f in files};matches=[s for s in scan.stdout.splitlines() if s not in own and '/model_based_planning/' not in s]
    if scan.returncode not in (0,1) or matches:raise RuntimeError(f'Seed collision/audit error:{matches} {scan.stderr[:200]}')
    record(RUN/'seed_audit.json',dict(at=now(),roots=roots,matches=matches,method='exact integer textual audit of all registered worktrees; all prior reserved groups untouched'))
    started=read_record(RUN/'task_started.json')['epoch'] if (RUN/'task_started.json').exists() else time.time()
    p=dict(at=now(),started=started,git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=sources,models=models,history=history,config=vars(PlanningConfig()),**groups,device='cpu',workers=4,seed_audit=digest(RUN/'seed_audit.json'))
    record(path,p);operation('protocol frozen before new seeds; one public-only configuration, no training, fixed functional/development/confirmation gates');return p

def init(identity):
    global MODELS,IDENTITY
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));torch.set_num_threads(1)
    MODELS={n:MaskablePPO.load(p,device='cpu') for n,p in dict(bc=BC,ppo=PPO,student=STUDENT).items()};IDENTITY=identity

def validate_rollouts(core,trace):
    checks=0;frontiers=0
    for branch in trace['alternatives']:
        real=copy.deepcopy(core);sim=PublicSimulator(core.get_public_observation())
        for action in branch['path']:
            #Only the validator owns a real Core. Its hidden data never enter the planner.
            real.step(action);pred=sim.step(action);actual=real.get_public_observation()
            for key in ('board','score','lines','level','hold'):
                if pred[key]!=actual[key]:raise RuntimeError(f'Public simulation parity:{key}')
            if sim.phase!='unknown_frontier':
                for key in ('currentPiece','gameOver','phase','holdAvailable'):
                    if pred[key]!=actual[key]:raise RuntimeError(f'Public known-spawn parity:{key}')
            else:frontiers+=1
            if actual['next'][:len(pred['next'])]!=pred['next']:raise RuntimeError('Public Hold/queue prefix parity')
            checks+=1
    return checks,frontiers

def game(name,seed,cap,functional=False):
    folder=RUN/('functional' if functional else f'evaluation/{cap}')/name;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'seed_{seed}.json';cp=path.with_suffix('.checkpoint.pkl')
    if path.exists():
        r=read_record(path);evaluation.validate_row(r,name,seed,cap,IDENTITY);return str(path)
    env=LongCandidateTetrisEnv(max_pieces=cap);env.reset(seed=seed)
    planner=PublicLookaheadController(MODELS['student'],MODELS['bc'])
    b12=HybridController(_DeterministicAdapter(MODELS['ppo']),_DeterministicAdapter(MODELS['bc']),HybridConfig('B',12,8))
    s=dict(identity=IDENTITY,seed=seed,variant=name,cap=cap,pieces=0,score=0,scores=[0],clears_0_to_4=[0]*5,new_holes=0,high_steps=0,high_streak=0,max_high_streak=0,inference_seconds=0.,latencies_ms=[],wall_seconds=0.,overrides=0,not_bc_or_student=0,bc_selected=0,student_rollout_failed=0,rollout_failure_avoided=0,reason_counts={},traces=[],simulation_checks=0,unknown_frontiers_checked=0,hidden_checks=0,b12_differences=0)
    if cp.exists():
        saved=read_blob(cp);s=saved['state']
        if s['identity']!=IDENTITY or (s['seed'],s['variant'],s['cap'])!=(seed,name,cap):raise RuntimeError('Job resume identity mismatch')
        env.set_state(saved['env']);restore_rng(saved['rng']);b12.active,b12.last_switch,b12.step=saved['b12']
        if env.pieces!=s['pieces'] or len(s['scores'])!=env.pieces+1:raise RuntimeError('Job resume progress mismatch')
    start=time.monotonic()-s['wall_seconds']
    while True:
        #Measure a fresh public preprocessing path as well as the actual decision.
        begin=time.perf_counter();env._candidate_cache=None
        obs,mask=env.get_observation(),env.action_masks();public=env.core.get_public_observation();trace=None
        if name=='planner':action,trace=planner.decide(obs,mask,public)
        elif name=='B12':action,_=b12.decide(obs,mask,public,env._placement_cache)
        else:action=fast_action(MODELS[name],obs,mask)
        ms=1000*(time.perf_counter()-begin);s['latencies_ms'].append(ms);s['inference_seconds']+=ms/1000
        if not mask[action]:raise RuntimeError('Illegal real decision')
        if env.pieces%1000==0 and env._placement_cache!=env.core.get_legal_placements():raise RuntimeError('Fast official placement parity mismatch')
        if trace:
            s['overrides']+=int(trace['override']);s['bc_selected']+=int(action==trace['bc_action']);s['not_bc_or_student']+=int(action not in (trace['student_action'],trace['bc_action']))
            failed=trace['alternatives'][0]['terminal'] in ('over','no_mechanical_action');s['student_rollout_failed']+=int(failed);s['rollout_failure_avoided']+=int(failed and trace['selected']['terminal'] not in ('over','no_mechanical_action'))
            reason=trace['reason'];s['reason_counts'][reason]=s['reason_counts'].get(reason,0)+1
            if functional:
                checks,frontiers=validate_rollouts(env.core,trace);s['simulation_checks']+=checks;s['unknown_frontiers_checked']+=frontiers
                other,btrace=b12.decide(obs,mask,public,env._placement_cache);s['b12_differences']+=int(action!=other)
                trace=dict(trace,b12_action=other,piece=env.pieces)
                if env.pieces%100==0:
                    before=pickle.dumps(env.core);altered=copy.deepcopy(env.core);altered.rng.state^=0x12345678;altered.bag=list(reversed(altered.bag));altered.queue[3:]=list('II')
                    a,again=planner.decide(obs,mask,altered.get_public_observation())
                    if a!=action or again!={k:v for k,v in trace.items() if k not in ('b12_action','piece')} or pickle.dumps(env.core)!=before:raise RuntimeError('Hidden-state leakage or real-state mutation')
                    s['hidden_checks']+=1
                s['traces'].append(trace)
            elif len(s['traces'])<32 and trace['override']:s['traces'].append(dict(trace,piece=env.pieces))
        _,_,over,capped,info=env.step(int(action));s['pieces']=env.pieces;s['score']=info['score'];s['scores'].append(info['score']);s['clears_0_to_4'][info['cleared_lines']]+=1;s['new_holes']+=info['new_holes']
        high=info['height_after']>=17;s['high_steps']+=int(high);s['high_streak']=s['high_streak']+1 if high else 0;s['max_high_streak']=max(s['max_high_streak'],s['high_streak'])
        if over or capped:
            s.update(game_over=bool(over),capped=bool(capped),lines=info['lines'],final_height=info['height_after'],final_holes=info['holes_after'],wall_seconds=time.monotonic()-start)
            evaluation.validate_row(s,name,seed,cap,IDENTITY);record(path,s)
            if cp.exists():cp.unlink()
            env.close();return str(path)
        if env.pieces%1000==0:
            guard();s['wall_seconds']=time.monotonic()-start;save_blob(cp,dict(state=s,env=env.get_state(),rng=rng_state(),b12=(b12.active,b12.last_switch,b12.step)))

def summarize(rows):
    s=evaluation.summary(rows);lat=[v for row in rows for v in row['latencies_ms']]
    s.update(mean_latency_ms=float(np.mean(lat)),p95_ms=float(np.percentile(lat,95)),p99_ms=float(np.percentile(lat,99)),overrides=sum(r['overrides'] for r in rows),third_actions=sum(r['not_bc_or_student'] for r in rows),rollout_failure_avoided=sum(r['rollout_failure_avoided'] for r in rows))
    return s

def gate(r,long=False,confirm=False):
    a,b,student=r['summary']['bc'],r['summary']['planner'],r['summary']['student'];p=r['paired']['planner']
    c=dict(survival=b['mean_pieces']>=a['mean_pieces']*(.95 if long else .98),early=b['early']<=a['early'] if long else b['early']==0,score=p['common_score_ratio']>=(1.08 if long else 1.10),seeds=p['positive_seeds']>=int(np.ceil(a['games']*.8125)),latency=b['p95_ms']<=50,high=b['high_fraction']<=a['high_fraction']+.02,holes=b['new_holes_per_100']<=max(1.5*a['new_holes_per_100'],a['new_holes_per_100']+.5),streak=b['mean_max_high_streak']<=max(1.5*a['mean_max_high_streak'],a['mean_max_high_streak']+50))
    if long:c.update(p10=b['p10']>=a['p10']*.9,total_score=b['mean_score']>=a['mean_score'])
    else:c.update(capped=b['capped']>=15,student_survival=b['mean_pieces']>=student['mean_pieces']*.98,student_early=b['early']<=student['early'],student_p10=b['p10']>=student['p10']*.9)
    if confirm:c['bootstrap']=p['ratio_ci95'][0]>1
    return dict(passed=all(c.values()),checks=c)

def evaluate(seeds,cap,label):
    rows={name:[] for name in ('bc','student','B12','planner')}
    with ProcessPoolExecutor(4,mp_context=get_context('spawn'),initializer=init,initargs=(digest(RUN/'protocol.json'),)) as pool:
        tasks={pool.submit(game,n,seed,cap):(n,seed) for seed in seeds for n in rows}
        for i,f in enumerate(as_completed(tasks),1):
            n,seed=tasks[f];rows[n].append(read_record(Path(f.result())));status('evaluation',label=label,completed=i,total=len(tasks));guard()
    rows={n:sorted(v,key=lambda r:r['seed']) for n,v in rows.items()};result=dict(label=label,cap=cap,summary={n:summarize(v) for n,v in rows.items()},paired={n:evaluation.paired(rows['bc'],v) for n,v in rows.items() if n!='bc'},vs_student=evaluation.paired(rows['student'],rows['planner']))
    record(RUN/f'{label}_summary.json',result);return result

def main():
    p=freeze();record(RUN/'runtime.json',dict(pid=os.getpid(),at=now()));os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    if (RUN/'closed.json').exists():return
    status('functional');operation('tmux fixed-budget planning study started/resumed')
    with ProcessPoolExecutor(2,mp_context=get_context('spawn'),initializer=init,initargs=(digest(RUN/'protocol.json'),)) as pool:
        paths=list(pool.map(game,['planner']*2,p['functional_seeds'],[500]*2,[True]*2))
    rows=[read_record(Path(path)) for path in paths];functional=dict(summary=summarize(rows),simulation_checks=sum(r['simulation_checks'] for r in rows),unknown_frontiers_checked=sum(r['unknown_frontiers_checked'] for r in rows),hidden_checks=sum(r['hidden_checks'] for r in rows),b12_differences=sum(r['b12_differences'] for r in rows),passed=summarize(rows)['p95_ms']<=50)
    record(RUN/'functional.json',functional);operation(f'functional gate={functional}')
    reason='stopped_functional'
    if functional['passed']:
        for label,key,cap,long,confirm in (('short16','short_seeds',20000,False,False),('long16','long_seeds',100000,True,False),('confirmation32','confirm_seeds',100000,True,True)):
            r=evaluate(p[key],cap,label);g=gate(r,long,confirm);record(RUN/f'{label}_gate.json',g);operation(f'{label} gate={g}');reason='stopped_'+label
            if not g['passed']:break
            if confirm:reason='confirmed'
    from training.model_based_planning_report import finish
    finish(reason)

if __name__=='__main__':
    def timeout_handler(*_):raise RuntimeError('Budget:external timeout')
    signal.signal(signal.SIGTERM,timeout_handler)
    try:main()
    except BaseException as exc:
        status('failed',error=repr(exc));operation(f'failure/interruption {exc!r}; all evidence retained')
        if not (RUN/'protocol.json').exists():raise
        from training.model_based_planning_report import finish
        finish('resource_stop' if str(exc).startswith('Budget:') else 'technical_failure',error=repr(exc));raise
