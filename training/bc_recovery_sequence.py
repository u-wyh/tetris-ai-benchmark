"""Preregistered multi-step recovery counterfactuals; strong labels require actual deaths."""
import os
import random
import resource
import shutil
import signal
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

import training.bc_sequence_transfer as old
from training.bc_sequence_transfer import record, read_record, save_blob, read_blob, rng_state, restore_rng
from training.bc_sequence_supervised import public_sample
from training.bc_sequence_safety import visible, STUDENT
from training.evaluation.hybrid_queue import BC, PPO, digest, verify_models
from training.evaluation.failure_curriculum_eval import fast_action
from training.train_ppo import ROOT, now

RUN=ROOT/'runs/bc_recovery_sequence'
SAFETY=ROOT/'runs/bc_sequence_safety/checkpoints/epoch_01/model.zip'
SEEDS=tuple(range(3102821000,3102821048))
BRANCHES={'A':0,'B':1,'C':16,'D':64}
WINDOW=3000
IDENTITY=None

def operation(message):
    with (ROOT/'OPERATIONS.md').open('a') as f:f.write(f'{now()} | BC recovery sequence: {message}\n')

def status(stage,**kw):record(RUN/'status.json',dict(at=now(),stage=stage,pid=os.getpid(),**kw))

def guard():
    p=read_record(RUN/'protocol.json')
    if time.time()-p['started']>86400:raise RuntimeError('Budget:24h wall clock')
    if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())>6*1024**3:raise RuntimeError('Budget:6GiB disk')
    if shutil.disk_usage(RUN).free<8*1024**3:raise RuntimeError('Budget:8GiB free disk')
    pending=[read_record(RUN/'runtime.json')['pid']];rss=0;seen=set()
    while pending:
        pid=pending.pop()
        if pid in seen:continue
        seen.add(pid)
        try:
            rss+=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            pending+=list(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
        except FileNotFoundError:pass
    if rss>10*1024**3:raise RuntimeError('Budget:10GiB aggregate RSS')

def freeze():
    RUN.mkdir(parents=True,exist_ok=True);verify_models()
    expected={STUDENT:'e40bfad269566c052d880cb01857f5bd998e22a1730c6a71d75845387f605914',SAFETY:'7737c0bf148c1cb0627924b956c346e85117fe060530267a4bce781498edd174'}
    if any(digest(p)!=sha for p,sha in expected.items()):raise RuntimeError('Frozen checkpoint mismatch')
    files=['training/bc_recovery_sequence.py','training/bc_recovery_sequence_report.py','training/bc_recovery_sequence_train.py','reports/experiments/bc_recovery_sequence_design.md','ops/run_bc_recovery_sequence.sh']
    files+=['training/bc_sequence_transfer.py','training/bc_sequence_safety.py','training/bc_sequence_supervised.py','training/bc_score_rl.py']
    files+=list(map(str,Path('training/tetris_core').rglob('*.py')))+list(map(str,Path('training/env').glob('*.py')))
    sources={f:digest(ROOT/f) for f in files}
    history={str(p.relative_to(ROOT)):digest(p) for p in [BC,PPO,STUDENT,SAFETY]}
    for folder in ('bc_sequence_transfer','bc_sequence_safety'):
        for p in (ROOT/'reports/experiments').glob(folder+'*'):history[str(p.relative_to(ROOT))]=digest(p)
        for p in (ROOT/'runs'/folder).rglob('*'):
            if p.is_file():history[str(p.relative_to(ROOT))]=digest(p)
    path=RUN/'protocol.json'
    if path.exists():
        p=read_record(path)
        if p['sources']!=sources or p['history']!=history:raise RuntimeError('Frozen source/history changed')
        return p
    if subprocess.run(['git','diff','--quiet','--',*files],cwd=ROOT).returncode:raise RuntimeError('Commit before protocol freeze')
    trees=subprocess.check_output(['git','worktree','list','--porcelain'],text=True,cwd=ROOT)
    roots=sorted({str(Path(line[9:])/d) for line in trees.splitlines() if line.startswith('worktree ') for d in ('runs','reports','training','ops') if (Path(line[9:])/d).exists()})
    audit=subprocess.run(['rg','-l','--hidden','-g','*.json','-g','*.jsonl','-g','*.md','-g','*.py','-g','*.sh','-g','*.csv',r'\b310282[1235][0-9]{3}\b',*roots],capture_output=True,text=True)
    own={str(ROOT/f) for f in files}
    collisions=[f for f in audit.stdout.splitlines() if f not in own and '/bc_recovery_sequence/' not in f]
    if audit.returncode not in (0,1) or collisions:raise RuntimeError(f'Seed audit:{collisions} {audit.stderr[:300]}')
    record(RUN/'seed_audit.json',dict(at=now(),roots=roots,collisions=collisions,method='exact integer text audit across registered worktrees; new groups excluded only in own protocol sources'))
    p=dict(started=time.time(),at=now(),pid=os.getpid(),git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=sources,history=history,data_seeds=SEEDS,train_seeds=SEEDS[:36],validation_seeds=SEEDS[36:],short_seeds=list(range(3102822000,3102822032)),long_seeds=list(range(3102823000,3102823016)),confirm_seeds=list(range(3102825000,3102825032)),device='cuda' if torch.cuda.is_available() else 'cpu',budget=dict(wall_seconds=86400,disk_bytes=6*1024**3,rss_bytes=10*1024**3,workers=4),seed_audit_sha=digest(RUN/'seed_audit.json'))
    record(path,p);operation('protocol frozen before fresh data; strong evidence needs observed student death and>=256-piece advantage');return p

def init(identity):
    global IDENTITY
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));torch.set_num_threads(1)
    old.MODELS={n:MaskablePPO.load(p,device='cpu') for n,p in dict(student=STUDENT,bc=BC).items()};IDENTITY=identity

def sample(seed):
    path=RUN/'episodes'/f'{seed}.pkl';path.parent.mkdir(exist_ok=True)
    if path.exists():
        r=read_blob(path)
        if r['identity']!=IDENTITY:raise RuntimeError('Episode identity')
        return str(path)
    env=old.LongCandidateTetrisEnv(max_pieces=20000);env.reset(seed=seed)
    r=dict(identity=IDENTITY,seed=seed,rows=[],snapshots=[],normal=[],severe=0,moderate=0,last=-999,exposure={x:0 for x in ('normal','moderate','severe','other')})
    cp=path.with_suffix('.checkpoint.pkl')
    if cp.exists():
        saved=read_blob(cp);r=saved['result']
        if r['identity']!=IDENTITY:raise RuntimeError('Episode resume identity')
        env.set_state(saved['env']);restore_rng(saved['rng'])
    while True:
        piece=env.pieces;risk=visible(env.get_observation()['state']);tier=risk['tier'];r['exposure'][tier]+=1
        choose=(tier=='severe' and r['severe']<3 and piece-r['last']>=128) or (tier=='moderate' and r['moderate']==0) or (tier=='normal' and piece==512)
        if choose:
            point=RUN/'snapshots'/f'{seed}_{piece}.pkl';point.parent.mkdir(exist_ok=True)
            save_blob(point,dict(identity=IDENTITY,seed=seed,piece=piece,risk=risk,env=env.get_state(),rng=rng_state()))
            r['snapshots'].append(str(point))
            if tier in ('severe','moderate'):r[tier]+=1
            if tier=='severe':r['last']=piece
        frame=public_sample(env) if tier=='normal' and piece%512==0 and len(r['normal'])<32 else None
        row,over,capped=old.move(env,'student');r['rows'].append(row)
        if frame is not None:frame.update(action=row['action'],episode=seed);r['normal'].append(frame)
        if over or capped:break
        if env.pieces%512==0:guard();save_blob(cp,dict(result=r,env=env.get_state(),rng=rng_state()))
    r.update(death=over,right_censored=capped,env=env.get_state(),rng=rng_state());save_blob(path,r)
    if cp.exists():cp.unlink()
    env.close();return str(path)

def classify(branches):
    a=branches['A'];strong=[];harm=[];ambiguous=[]
    for name in ('B','C','D'):
        b=branches[name]
        if a['death'] and len(b['rows'])>=len(a['rows'])+256 and any(f['action']!=f['student_action'] for f in b['frames']):strong.append(name)
        if b['death'] and len(a['rows'])>=len(b['rows'])+256:harm.append(name)
        if not a['death'] and not b['death']:ambiguous.append(name)
    return dict(strong=strong,harm=harm,ambiguous=ambiguous,selected=next(iter(strong),None),multistep=bool(strong and 'B' not in strong),first_same=branches['B']['rows'][0]['action']==a['rows'][0]['action'])

def counterfactual(source):
    source=Path(source);point=read_blob(source);path=RUN/'counterfactuals'/source.name;path.parent.mkdir(exist_ok=True)
    if path.exists():
        r=read_blob(path)
        if r['identity']!=IDENTITY or r['snapshot_sha']!=digest(source):raise RuntimeError('CF identity')
        return str(path)
    r=dict(identity=IDENTITY,snapshot_sha=digest(source),seed=point['seed'],piece=point['piece'],risk=point['risk'],start_score=point['env']['core'].score,branches={})
    cp=path.with_suffix('.checkpoint.pkl')
    saved=read_blob(cp) if cp.exists() else None
    if saved:
        r=saved['result']
        if r['identity']!=IDENTITY or r['snapshot_sha']!=digest(source):raise RuntimeError('CF resume identity')
    for name,length in BRANCHES.items():
        if name in r['branches']:continue
        cap=point['piece']+WINDOW;env=old.LongCandidateTetrisEnv(max_pieces=cap);env.reset(seed=point['seed'])
        state=dict(point['env'],max_pieces=cap);env.set_state(state);restore_rng(point['rng'])
        rows=[];frames=[]
        if saved and saved['name']==name:
            rows=saved['rows'];frames=saved['frames'];env.set_state(saved['env']);restore_rng(saved['rng'])
        over=False
        for index in range(len(rows),WINDOW):
            risk=visible(env.get_observation()['state']);control='bc' if index<length else 'student'
            frame=public_sample(env) if index<length else None
            obs=env.get_observation();mask=env.action_masks()
            student_action=int(fast_action(old.MODELS['student'],obs,mask))
            bc_action=int(fast_action(old.MODELS['bc'],obs,mask))
            row,over,capped=old.move(env,control);row.update(before_risk=risk,window_step=index,control=control,handoff=index==length)
            row.update(student_action=student_action,bc_action=bc_action,disagreement=student_action!=bc_action,legal_count=int(mask.sum()))
            if frame is not None:
                frame.update(action=row['action'],student_action=student_action,episode=point['seed'],window_step=index,risk=risk);frames.append(frame)
                row['student_action']=student_action;row['bc_action']=row['action']
            rows.append(row)
            if over or capped:break
            if len(rows)%256==0:
                guard();save_blob(cp,dict(result=r,name=name,rows=rows,frames=frames,env=env.get_state(),rng=rng_state()))
        r['branches'][name]=dict(rows=rows,frames=frames,death=over,right_censored=not over,env=env.get_state(),rng=rng_state(),handoff_reached=len(rows)>=length)
        save_blob(cp,dict(result=r,name=name,rows=rows,frames=frames,env=env.get_state(),rng=rng_state()));saved=None;env.close()
    r['classification']=classify(r['branches'])
    if r['classification']['first_same']:
        a,b=r['branches']['A'],r['branches']['B']
        if a['death']!=b['death'] or [(v['action'],v['score'],v['placed']) for v in a['rows']]!=[(v['action'],v['score'],v['placed']) for v in b['rows']]:raise RuntimeError('Identical first action must reproduce student branch')
    save_blob(path,r)
    if cp.exists():cp.unlink()
    return str(path)

def analyze():
    result={p:dict(states=0,strong_states=0,episodes=set(),frames=0,multistep_episodes=set(),first_same_multistep=0,normal=0,strong_branches={n:0 for n in ('B','C','D')},harm_branches=0,both_censored=0) for p in ('train','validation')}
    for seed in SEEDS:
        split='train' if seed in SEEDS[:36] else 'validation';result[split]['normal']+=len(read_blob(RUN/'episodes'/f'{seed}.pkl')['normal'])
    for path in sorted((RUN/'counterfactuals').glob('*.pkl')):
        r=read_blob(path);split='train' if r['seed'] in SEEDS[:36] else 'validation';s=result[split];c=r['classification'];s['states']+=1
        s['harm_branches']+=len(c['harm']);s['both_censored']+=len(c['ambiguous'])
        for name in c['strong']:s['strong_branches'][name]+=1
        if c['selected']:
            s['strong_states']+=1;s['episodes'].add(r['seed']);frames=r['branches'][c['selected']]['frames'];s['frames']+=sum(f['action']!=f['student_action'] for f in frames)
            if c['multistep']:s['multistep_episodes'].add(r['seed']);s['first_same_multistep']+=c['first_same']
    a,b=result['train'],result['validation']
    checks=dict(train_states=a['strong_states']>=12,val_states=b['strong_states']>=4,train_episodes=len(a['episodes'])>=6,val_episodes=len(b['episodes'])>=3,train_frames=a['frames']>=64,val_frames=b['frames']>=16,train_multi=len(a['multistep_episodes'])>=2,val_multi=len(b['multistep_episodes'])>=1)
    for s in result.values():
        for key in ('episodes','multistep_episodes'):s[key]=sorted(s[key])
    data=dict(groups=result,checks=checks,passed=all(checks.values()),sha256={str(p.relative_to(RUN)):digest(p) for folder in ('episodes','snapshots','counterfactuals') for p in sorted((RUN/folder).glob('*.pkl'))})
    record(RUN/'analysis.json',data);return data

def main():
    protocol=freeze();record(RUN/'runtime.json',dict(pid=os.getpid(),at=now()));os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    if (RUN/'closed.json').exists():return
    status('sampling');operation('tmux bounded fresh sampling started/resumed')
    with ProcessPoolExecutor(4,mp_context=get_context('spawn'),initializer=init,initargs=(digest(RUN/'protocol.json'),)) as pool:
        tasks=[pool.submit(sample,seed) for seed in SEEDS]
        for i,f in enumerate(as_completed(tasks),1):f.result();status('sampling',completed=i,total=48);guard()
        snapshots=[p for seed in SEEDS for p in read_blob(RUN/'episodes'/f'{seed}.pkl')['snapshots']]
        if len(snapshots)>256:raise RuntimeError('Snapshot limit')
        tasks=[pool.submit(counterfactual,p) for p in snapshots]
        for i,f in enumerate(as_completed(tasks),1):f.result();status('counterfactuals',completed=i,total=len(tasks));guard()
    data=analyze();operation(f'learnability gate={data["checks"]}')
    if data['passed']:
        from training.bc_recovery_sequence_train import run
        run(protocol);return
    from training.bc_recovery_sequence_report import finish
    finish('insufficient_high_confidence_labels')

if __name__=='__main__':
    def timeout_handler(*_):raise RuntimeError('Budget:external wall timeout')
    signal.signal(signal.SIGTERM,timeout_handler)
    try:main()
    except BaseException as exc:
        status('failed',error=repr(exc));operation(f'failed/interrupted:{exc!r}; evidence retained')
        from training.bc_recovery_sequence_report import finish
        finish('resource_stop' if str(exc).startswith('Budget:') else 'technical_failure',error=repr(exc))
        raise
