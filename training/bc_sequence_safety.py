"""One bounded state-dependent safety distillation study; immutable prior artifacts."""
import hashlib
import json
import math
import os
import random
import resource
import signal
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

import training.bc_sequence_transfer as old
import training.bc_score_rl as evaluation
from training.bc_sequence_transfer import record,read_record,save_blob,read_blob,rng_state,restore_rng
from training.bc_sequence_supervised import public_sample
from training.evaluation.hybrid_queue import BC,PPO,digest,verify_models
from training.imitation_bc import batch,metrics
from training.train_ppo import ROOT,now

RUN=ROOT/'runs/bc_sequence_safety'
PRIOR=ROOT/'runs/bc_sequence_transfer/supervised'
STUDENT=PRIOR/'checkpoints/epoch_01/model.zip'
SEEDS=tuple(range(3102811000,3102811024))
MODELS={}
IDENTITY=None


def operation(message):
    with (ROOT/'OPERATIONS.md').open('a') as f:f.write(f'{now()} | BC sequence safety: {message}\n')


def status(stage,**kw):
    record(RUN/'status.json',dict(at=now(),stage=stage,pid=os.getpid(),**kw))


def guard():
    p=read_record(RUN/'protocol.json')
    if time.time()-p['started']>86400:raise RuntimeError('24h wall budget exhausted')
    if shutil.disk_usage(RUN).free<8*1024**3:raise RuntimeError('Less than8GiB free')
    if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())>6*1024**3:raise RuntimeError('6GiB disk budget exceeded')


def visible(state):
    board=np.asarray(state[:200]).reshape(20,10)>0
    heights=[next((20-y for y in range(20) if board[y,x]),0) for x in range(10)]
    holes=sum(sum(not board[y,x] for y in range(20-h,20)) for x,h in enumerate(heights))
    wells=[max(0,min(20 if x==0 else heights[x-1],20 if x==9 else heights[x+1])-h) for x,h in enumerate(heights)]
    h,w,t=max(heights),max(wells),int(board[:4].sum())
    tier='severe' if h>=12 or t>0 or w>=10 else 'moderate' if h>=8 or w>=6 else 'normal' if h<=6 else 'other'
    return dict(height=h,holes=int(holes),well=w,spawn=t,tier=tier)


def freeze():
    RUN.mkdir(parents=True,exist_ok=True);verify_models()
    if digest(STUDENT)!='e40bfad269566c052d880cb01857f5bd998e22a1730c6a71d75845387f605914':raise RuntimeError('Frozen student mismatch')
    names=['training/bc_sequence_safety.py','training/bc_sequence_safety_report.py','training/bc_sequence_transfer.py',
        'training/bc_sequence_supervised.py','training/bc_score_rl.py','training/imitation_bc.py',
        'training/candidate_policy.py','training/candidate_features.py','training/hybrid_policy.py',
        'training/evaluation/failure_curriculum_eval.py','training/evaluation/fast_legal.py',
        'training/evaluation/fast_candidate_features.py','reports/experiments/bc_sequence_safety_design.md']
    names+=list(map(str,Path('training/tetris_core').rglob('*.py')))+list(map(str,Path('training/env').glob('*.py')))
    sources={n:digest(ROOT/n) for n in names}
    frozen={str(p.relative_to(ROOT)):digest(p) for p in [BC,PPO,STUDENT,*sorted((PRIOR/'data').glob('*.pkl'))]}
    frozen.update({str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'reports/experiments').glob('bc_sequence_transfer_*.md')})
    path=RUN/'protocol.json'
    if path.exists():
        p=read_record(path)
        if p['sources']!=sources or p['frozen']!=frozen:raise RuntimeError('Frozen identity changed; refusing resume')
        return p
    if subprocess.run(['git','diff','--quiet','--',*names],cwd=ROOT).returncode:raise RuntimeError('Commit before freeze')
    trees=subprocess.check_output(['git','worktree','list','--porcelain'],cwd=ROOT,text=True)
    roots=sorted({str(Path(l[9:])/d) for l in trees.splitlines() if l.startswith('worktree ') for d in ('runs','reports','training','ops') if (Path(l[9:])/d).exists()})
    scan=subprocess.run(['rg','-l','--hidden','-g','*.json','-g','*.jsonl','-g','*.md','-g','*.py','-g','*.sh','-g','*.csv',r'\b310281[1235][0-9]{3}\b',*roots],text=True,capture_output=True)
    own={str(ROOT/n) for n in names}
    matches=[n for n in scan.stdout.splitlines() if n not in own and '/bc_sequence_safety/' not in n]
    if scan.returncode not in (0,1) or matches:raise RuntimeError(f'Seed collision/error: {matches} {scan.stderr[:500]}')
    record(RUN/'seed_audit.json',dict(at=now(),roots=roots,matches=matches,method='conservative exact integer scan',reserved_old='3102804000..3102804031 unused'))
    p=dict(at=now(),started=time.time(),git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=sources,frozen=frozen,
        data_seeds=SEEDS,short_seeds=list(range(3102812000,3102812032)),long_seeds=list(range(3102813000,3102813016)),confirm_seeds=list(range(3102815000,3102815032)),
        device='cuda' if torch.cuda.is_available() else 'cpu',seed_audit_sha256=digest(RUN/'seed_audit.json'))
    record(path,p);operation('protocol frozen before new data; safety weighting and new disjoint evaluation groups fixed')
    return p


def init(identity,paths=None):
    global MODELS,IDENTITY
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));torch.set_num_threads(1)
    MODELS={n:MaskablePPO.load(p,device='cpu') for n,p in (paths or dict(bc=BC,student=STUDENT)).items()}
    old.MODELS=MODELS;IDENTITY=identity


def sample_episode(seed):
    folder=RUN/'episodes';folder.mkdir(exist_ok=True)
    path=folder/f'seed_{seed}.pkl'
    if path.exists():
        r=read_blob(path)
        if r['identity']!=IDENTITY:raise RuntimeError('Episode identity mismatch')
        return str(path)
    env=old.LongCandidateTetrisEnv(max_pieces=20000);env.reset(seed=seed)
    r=dict(identity=IDENTITY,seed=seed,rows=[],normal=[],snapshots=[],counts={s:0 for s in ('normal','severe','moderate','other')},taken=dict(severe=0,moderate=0),last=dict(severe=-999,moderate=-999))
    cp=path.with_suffix('.checkpoint.pkl')
    if cp.exists():
        saved=read_blob(cp);r=saved['result']
        if r['identity']!=IDENTITY:raise RuntimeError('Episode checkpoint mismatch')
        env.set_state(saved['env']);restore_rng(saved['rng'])
    while True:
        before=env.pieces;obs=env.get_observation();risk=visible(obs['state']);tier=risk['tier'];r['counts'][tier]+=1
        frame=None
        if tier=='normal' and before%512==0 and len(r['normal'])<32:frame=public_sample(env)
        if tier in ('severe','moderate') and r['taken'][tier]<8 and before-r['last'][tier]>=(4 if tier=='severe' else 32):
            point=RUN/'snapshots'/f'{seed}_{before}.pkl';point.parent.mkdir(exist_ok=True)
            save_blob(point,dict(identity=IDENTITY,seed=seed,piece=before,risk=risk,frame=public_sample(env),env=env.get_state(),rng=rng_state()))
            r['snapshots'].append(str(point));r['taken'][tier]+=1;r['last'][tier]=before
        row,over,capped=old.move(env,'student');r['rows'].append(row)
        if frame is not None:frame.update(action=row['action'],episode=f'new:{seed}');r['normal'].append(frame)
        if over or capped:break
        if env.pieces%512==0:
            guard();save_blob(cp,dict(result=r,env=env.get_state(),rng=rng_state()))
    r.update(death=over,right_censored=capped,env=env.get_state(),rng=rng_state());save_blob(path,r)
    if cp.exists():cp.unlink()
    env.close();return str(path)


def counterfactual(snapshot):
    source=Path(snapshot);saved=read_blob(source)
    path=RUN/'counterfactuals'/source.name;path.parent.mkdir(exist_ok=True)
    if path.exists():
        r=read_blob(path)
        if r['identity']!=IDENTITY or r['snapshot_sha256']!=digest(source):raise RuntimeError('Counterfactual identity changed')
        return str(path)
    result=dict(identity=IDENTITY,snapshot_sha256=digest(source),seed=saved['seed'],piece=saved['piece'],risk=saved['risk'],frame=saved['frame'],branches={})
    for name in ('bc','student'):
        env=old.LongCandidateTetrisEnv(max_pieces=20000);env.reset(seed=saved['seed']);env.set_state(saved['env'])
        rows=[];over=False
        for _ in range(256):
            row,over,capped=old.move(env,name);rows.append(row)
            if over or capped:break
        result['branches'][name]=dict(rows=rows,death=over,right_censored=not over,env=env.get_state(),rng=rng_state())
        env.close()
    a,b=result['branches']['bc'],result['branches']['student'];n=min(len(a['rows']),len(b['rows']))
    result['accepted']=qualifies(saved['risk'],a,b)
    result['common']=n;result['frame']['action']=a['rows'][0]['action'];result['frame']['risk']=saved['risk']
    save_blob(path,result);return str(path)


def qualifies(risk,a,b):
    n=min(len(a['rows']),len(b['rows']));ra,rb=a['rows'][-1],b['rows'][n-1]
    return bool(not a['death'] and len(a['rows'])==256 and ra['height']<=6 and risk['height']-ra['height']>=3 and ra['holes']<=max(risk['holes'],2) and (b['death'] or ra['height']<=rb['height']+2))


def old_frames(source,validation=False):
    seeds=range(3102802006,3102802008) if validation else range(3102802000,3102802006)
    return [f for seed in seeds for f in read_blob(PRIOR/'data'/f'{source}_{seed}.pkl')['frames']]


def dataset():
    groups={p:dict(normal=[],safe=[],bc=old_frames('bc',p=='validation'),ppo=old_frames('ppo',p=='validation')) for p in ('train','validation')}
    episodes={p:set() for p in groups};severe={p:0 for p in groups}
    for seed in SEEDS:
        p='train' if seed in SEEDS[:18] else 'validation';r=read_blob(RUN/'episodes'/f'seed_{seed}.pkl');groups[p]['normal']+=r['normal']
    accepted=rejected=0;all_frames=[];exposure={t:0 for t in ('normal','moderate','severe','other')}
    for seed in SEEDS:
        for t,n in read_blob(RUN/'episodes'/f'seed_{seed}.pkl')['counts'].items():exposure[t]+=n
    for path in sorted((RUN/'counterfactuals').glob('*.pkl')):
        r=read_blob(path);p='train' if r['seed'] in SEEDS[:18] else 'validation'
        all_frames.append(r['frame'])
        if r['accepted']:
            accepted+=1;groups[p]['safe'].append(r['frame']);episodes[p].add(r['seed']);severe[p]+=r['risk']['tier']=='severe'
        else:rejected+=1
    counts={p:{s:len(rows) for s,rows in g.items()} for p,g in groups.items()}
    checks=dict(train=counts['train']['safe']>=32,val=counts['validation']['safe']>=8,train_episode=len(episodes['train'])>=4,val_episode=len(episodes['validation'])>=2,severe_train=severe['train']>=8,severe_val=severe['validation']>=4)
    record(RUN/'dataset.json',dict(counts=counts,severe=severe,episodes={p:sorted(v) for p,v in episodes.items()},exposure=exposure,accepted=accepted,rejected=rejected,checks=checks,passed=all(checks.values()),sha256={str(p.relative_to(RUN)):digest(p) for folder in ('episodes','counterfactuals') for p in sorted((RUN/folder).glob('*.pkl'))}))
    new_distribution(all_frames)
    return groups,all(checks.values())


def new_distribution(frames):
    torch.set_num_threads(1);bc=MaskablePPO.load(BC,device='cpu');student=MaskablePPO.load(STUDENT,device='cpu')
    support={tuple(x) for x in read_record(RUN/'mechanism.json')['bc_support_bins']};result={}
    for tier in ('severe','moderate'):
        rows=[f for f in frames if f['risk']['tier']==tier];kl=agree=outside=0
        with torch.inference_mode():
            for i in range(0,len(rows),16):
                part=rows[i:i+16];a,prob,_=logits(bc.policy,part,'cpu');b,_,_=logits(student.policy,part,'cpu')
                kl+=float((prob*(a-b)).sum());agree+=int((a.argmax(1)==b.argmax(1)).sum())
        for f in rows:
            r=f['risk'];outside+=(r['height']//2,min(r['holes'],10),r['well']//2) not in support
        result[tier]=dict(states=len(rows),bc_student_agreement=agree/max(1,len(rows)),masked_kl=kl/max(1,len(rows)),outside_bc_support=outside)
    record(RUN/'new_distribution.json',result)


def logits(policy,frames,device):
    obs,mask,actions=batch(frames,device);dist=policy.get_distribution(obs,action_masks=mask)
    return dist.distribution.logits,dist.distribution.probs,actions


def mechanism():
    if (RUN/'mechanism.json').exists():return
    torch.set_num_threads(1);bc=MaskablePPO.load(BC,device='cpu');student=MaskablePPO.load(STUDENT,device='cpu')
    result={};support=set()
    for source in ('bc','ppo','transfer'):
        frames=old_frames(source)+old_frames(source,True);risks=[visible(f['state']) for f in frames]
        if source=='bc':support={ (r['height']//2,min(r['holes'],10),r['well']//2) for r in risks }
        risky=[f for f,r in zip(frames,risks) if r['tier'] in ('severe','moderate')]
        selected=risky[:1024] or frames[::max(1,len(frames)//1024)][:1024]
        kl=agree=0
        with torch.inference_mode():
            for i in range(0,len(selected),16):
                a,prob,_=logits(bc.policy,selected[i:i+16],'cpu');b,_,_=logits(student.policy,selected[i:i+16],'cpu')
                kl+=float((prob*(a-b)).sum());agree+=int((a.argmax(1)==b.argmax(1)).sum())
        result[source]=dict(frames=len(frames),risk_counts={t:sum(r['tier']==t for r in risks) for t in ('normal','moderate','severe','other')},compared=len(selected),high_risk_comparison=bool(risky),bc_student_agreement=agree/len(selected),bc_to_student_kl=kl/len(selected))
    result['bc_support_bins']=[list(x) for x in sorted(support)]
    result['student_bc_replay']=metrics(student.policy,old_frames('bc',True)[::2],'cpu')
    record(RUN/'mechanism.json',result);operation('bounded mechanism audit of frozen prior data completed; historical death seed not replayed')


def save_checkpoint(model,opt,state,label):
    root=RUN/'checkpoints';root.mkdir(exist_ok=True);temp=root/'.pending'
    if temp.exists():shutil.rmtree(temp)
    temp.mkdir();model.save(temp/'model.zip')
    torch.save(dict(optimizer=opt.state_dict(),state=state,rng=rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),temp/'training.pt')
    record(temp/'manifest.json',dict(identity=digest(RUN/'protocol.json'),dataset=digest(RUN/'dataset.json'),git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sha256={p.name:digest(p) for p in temp.iterdir()}))
    path=root/label
    if path.exists():raise RuntimeError('Refusing checkpoint overwrite')
    os.replace(temp,path);record(RUN/'latest.json',dict(path=str(path),manifest=digest(path/'manifest.json')));return path


def load_checkpoint(path,device):
    m=read_record(path/'manifest.json')
    if m['identity']!=digest(RUN/'protocol.json') or m['dataset']!=digest(RUN/'dataset.json') or any(digest(path/n)!=h for n,h in m['sha256'].items()):raise RuntimeError('Checkpoint identity mismatch')
    model=MaskablePPO.load(path/'model.zip',device=device);opt=torch.optim.Adam(actor(model),lr=1e-5)
    saved=torch.load(path/'training.pt',map_location=device,weights_only=False);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    if saved['cuda_rng'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return model,opt,saved['state']


def actor(model):return [p for n,p in model.policy.named_parameters() if n.startswith(('state_encoder.','candidate_encoder.','score_head.'))]


def monitor(model,groups,device):
    result={}
    for source in ('normal','safe','bc','ppo'):
        rows=groups['validation'][source];rows=rows[::max(1,len(rows)//2048)][:2048]
        result[source]=metrics(model.policy,rows,device)
    severe=[f for f in groups['validation']['safe'] if f['risk']['tier']=='severe']
    result['severe']=metrics(model.policy,severe,device) if severe else None
    return result


def train(groups,protocol):
    device=protocol['device'];torch.set_num_threads(1);random.seed(20261028);np.random.seed(20261028);torch.manual_seed(20261028)
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.5)
    bc=MaskablePPO.load(BC,device=device);student=MaskablePPO.load(STUDENT,device=device)
    for ref in (bc,student):
        for p in ref.policy.parameters():p.requires_grad_(False)
    if (RUN/'latest.json').exists():
        pointer=read_record(RUN/'latest.json');folder=Path(pointer['path'])
        if digest(folder/'manifest.json')!=pointer['manifest']:raise RuntimeError('Pointer mismatch')
        model,opt,state=load_checkpoint(folder,device)
    else:
        model=MaskablePPO.load(STUDENT,device=device);opt=torch.optim.Adam(actor(model),lr=1e-5)
        state=dict(epoch=1,next_batch=0,updates=0,history=[dict(epoch=0,metrics=monitor(model,groups,device))]);save_checkpoint(model,opt,state,'initial')
    if len(state['history'])>1 and state['history'][-1]['stop']:
        record(RUN/'training.json',dict(updates=state['updates'],history=state['history'],model=str(folder/'model.zip'),sha256=digest(folder/'model.zip')))
        return folder/'model.zip'
    baseline=state['history'][0]['metrics']
    while state['epoch']<=2:
        epoch=state['epoch'];losses=[];grads=[]
        for index in range(state['next_batch'],256):
            loss=0
            for source,reference,weight,ce in (('normal',student,2.,.5),('bc',bc,.25,0.),('safe',bc,1.,.25)):
                rows=random.choices(groups['train'][source],k=16)
                current,_,_=logits(model.policy,rows,device)
                with torch.no_grad():target,prob,actions=logits(reference.policy,rows,device);labels=target.argmax(1)
                kl=(prob*(target-current)).sum(1)
                if source=='safe':kl=kl*torch.tensor([min(3,max(1,1+(r['risk']['height']-8)/4)) for r in rows],device=device)
                loss=loss+weight*kl.mean()-ce*current.gather(1,labels[:,None]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor(model),.5,error_if_nonfinite=True);opt.step()
            losses.append(float(loss.detach()));grads.append(float(norm));state['updates']+=1;state['next_batch']=index+1
            if state['updates']==2 or state['updates']%64==0:
                if any(not torch.equal(v,student.policy.value_net.state_dict()[n]) for n,v in model.policy.value_net.state_dict().items()):raise RuntimeError('Critic changed')
                guard();folder=save_checkpoint(model,opt,state,f'update_{state["updates"]:06d}')
                if state['updates']==2:
                    expected={n:v.clone() for n,v in model.policy.state_dict().items()};model,opt,state=load_checkpoint(folder,device)
                    if any(not torch.equal(v,expected[n]) for n,v in model.policy.state_dict().items()):raise RuntimeError('Save/load mismatch')
                    record(RUN/'gradient_smoke.json',dict(real_updates=2,passed=True,optimizer_steps=[int(s['step']) for s in opt.state.values()]))
        measured=monitor(model,groups,device)
        signal=measured['safe']['loss']<=baseline['safe']['loss']*.99 or measured['safe']['top1']>baseline['safe']['top1']+.01
        stop=not signal or measured['normal']['top1']<baseline['normal']['top1']-.10 or measured['ppo']['top1']<baseline['ppo']['top1']-.05
        state['history'].append(dict(epoch=epoch,metrics=measured,stop=stop,learning_signal=signal,loss=float(np.mean(losses)),gradient=float(np.mean(grads))))
        state.update(epoch=epoch+1,next_batch=0);folder=save_checkpoint(model,opt,state,f'epoch_{epoch:02d}')
        record(RUN/'training.json',dict(updates=state['updates'],history=state['history'],model=str(folder/'model.zip'),sha256=digest(folder/'model.zip')))
        operation(f'epoch{epoch} complete, updates={state["updates"]}, high-score-proxy loss stop={stop}')
        if stop:break
    record(RUN/'training.json',dict(updates=state['updates'],history=state['history'],model=str(folder/'model.zip'),sha256=digest(folder/'model.zip')))
    return folder/'model.zip'


def gate(result,long=False,confirm=False):
    a,b=result['summary']['bc'],result['summary']['candidate'];p=result['paired']['candidate'];n=a['games']
    c=dict(survival=b['mean_pieces']>=(.95 if long else .98)*a['mean_pieces'],early=b['early']<=a['early'] if long else b['early']==0,
        score=p['common_score_ratio']>=1.08,seeds=p['positive_seeds']>=math.ceil(n*.8125),high=b['high_fraction']<=a['high_fraction']+.02,
        holes=b['new_holes_per_100']<=max(1.5*a['new_holes_per_100'],a['new_holes_per_100']+.5),streak=b['mean_max_high_streak']<=max(1.5*a['mean_max_high_streak'],a['mean_max_high_streak']+50))
    if long:
        pairs=sorted(p.get('pairs',[]),key=lambda r:r['candidate_score']-r['baseline_score'],reverse=True)
        remaining=pairs[max(1,math.ceil(n/8)):]
        broad=bool(len(pairs)==n and remaining and sum(r['candidate_score'] for r in remaining)>=1.08*sum(r['baseline_score'] for r in remaining))
        c.update(p10=b['p10']>=.9*a['p10'],total_score=b['mean_score']>=a['mean_score'],trimmed_score=broad)
    else:c['capped']=b['capped']>=31
    if confirm:c['ci']=p['ratio_ci95'][0]>1
    return dict(passed=all(c.values()),checks=c)


def eval_init(paths):
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3));evaluation.eval_init(paths)


def evaluate(model,seeds,cap,label):
    paths=dict(bc=str(BC),ppo=str(PPO),student=str(STUDENT),candidate=str(model));names=(*paths,'B12')
    identity=dict(protocol=digest(RUN/'protocol.json'),models={n:digest(Path(p)) for n,p in paths.items()});folder=RUN/'evaluation'/label
    rows={n:[] for n in names};pending=[]
    for seed in seeds:
        for name in names:
            path=folder/name/f'seed_{seed}.json';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():r=read_record(path);evaluation.validate_row(r,name,seed,cap,identity);rows[name].append(r)
            else:pending.append((name,seed,path))
    with ProcessPoolExecutor(4,mp_context=get_context('spawn'),initializer=eval_init,initargs=(paths,)) as pool:
        tasks={pool.submit(evaluation.eval_game,n,s,cap,identity,p.with_suffix('.checkpoint.pkl')):(n,s,p) for n,s,p in pending}
        for f in as_completed(tasks):
            n,s,p=tasks[f];r=f.result();evaluation.validate_row(r,n,s,cap,identity);record(p,r);rows[n].append(r)
            status('evaluation',label=label,completed=sum(map(len,rows.values())),total=len(seeds)*5);guard()
    rows={n:sorted(r,key=lambda x:x['seed']) for n,r in rows.items()}
    result=dict(label=label,cap=cap,summary={n:evaluation.summary(r) for n,r in rows.items()},paired={n:evaluation.paired(rows['bc'],rows[n]) for n in names if n!='bc'},candidate_vs_student=evaluation.paired(rows['student'],rows['candidate']))
    record(folder/'summary.json',result);return result


def main():
    protocol=freeze();os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    if (RUN/'outcome.json').exists():
        from training.bc_sequence_safety_report import finish
        finish();return
    mechanism();identity=digest(RUN/'protocol.json');status('sampling');operation('tmux bounded sampling started/resumed')
    with ProcessPoolExecutor(4,mp_context=get_context('spawn'),initializer=init,initargs=(identity,)) as pool:
        tasks=[pool.submit(sample_episode,s) for s in SEEDS]
        for i,f in enumerate(as_completed(tasks),1):f.result();status('sampling',completed=i,total=24)
        snapshots=[p for s in SEEDS for p in read_blob(RUN/'episodes'/f'seed_{s}.pkl')['snapshots']]
        tasks=[pool.submit(counterfactual,p) for p in snapshots]
        for i,f in enumerate(as_completed(tasks),1):f.result();status('counterfactuals',completed=i,total=len(tasks));guard()
    groups,qualified=dataset();reason='insufficient_safety_data';confirmed=False;candidate=None
    if qualified:
        status('training');candidate=train(groups,protocol)
        for label,seeds,cap,long,confirm in (('short32',protocol['short_seeds'],20000,False,False),('long16',protocol['long_seeds'],100000,True,False),('confirmation32',protocol['confirm_seeds'],100000,True,True)):
            result=evaluate(candidate,seeds,cap,label);decision=gate(result,long,confirm);record(RUN/f'{label}_gate.json',decision);operation(f'{label} gate={decision}')
            reason='stopped_'+label
            if not decision['passed']:break
            if confirm:confirmed=True;reason='confirmed'
    record(RUN/'outcome.json',dict(reason=reason,confirmation=confirmed,candidate=str(candidate) if candidate else None))
    from training.bc_sequence_safety_report import finish
    finish()


if __name__=='__main__':
    def terminate(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    try:main()
    except BaseException as exc:
        RUN.mkdir(parents=True,exist_ok=True);status('failed',error=repr(exc));operation(f'failure/interruption: {type(exc).__name__}: {exc}');raise
