"""Only executed after preregistered cross-episode strong-evidence gates pass."""
import math
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
import training.bc_sequence_safety as safety
import training.bc_score_rl as evaluation
from training.imitation_bc import metrics
from training.bc_recovery_sequence import RUN,ROOT,SEEDS,BC,STUDENT,SAFETY,read_record,record,read_blob,digest,guard,status,operation

def groups():
    data={p:dict(sequences=[],normal=[]) for p in ('train','validation')}
    for seed in SEEDS:
        split='train' if seed in SEEDS[:36] else 'validation';data[split]['normal']+=read_blob(RUN/'episodes'/f'{seed}.pkl')['normal']
    for path in sorted((RUN/'counterfactuals').glob('*.pkl')):
        r=read_blob(path);name=r['classification']['selected']
        if not name:continue
        split='train' if r['seed'] in SEEDS[:36] else 'validation'
        data[split]['sequences'].append(r['branches'][name]['frames'])
    return data

def monitor(model,data,device):
    val=data['validation'];frames=[f for seq in val['sequences'] for f in seq if f['action']!=f['student_action']]
    return dict(correction=metrics(model.policy,frames,device),normal=metrics(model.policy,val['normal'],device),bc=metrics(model.policy,safety.old_frames('bc',True)[::4],device),ppo=metrics(model.policy,safety.old_frames('ppo',True)[::4],device))

def train(protocol):
    safety.RUN=RUN;data=groups();device=protocol['device'];torch.set_num_threads(1)
    random.seed(20261029);np.random.seed(20261029);torch.manual_seed(20261029)
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.5)
    teacher=MaskablePPO.load(STUDENT,device=device)
    for p in teacher.policy.parameters():p.requires_grad_(False)
    if not (RUN/'dataset.json').exists():
        record(RUN/'dataset.json',dict(analysis=digest(RUN/'analysis.json'),protocol=digest(RUN/'protocol.json'),source=digest(Path(__file__)),counts={k:dict(sequences=len(v['sequences']),normal=len(v['normal'])) for k,v in data.items()}))
    if (RUN/'latest.json').exists():
        pointer=read_record(RUN/'latest.json');folder=Path(pointer['path'])
        if digest(folder/'manifest.json')!=pointer['manifest']:raise RuntimeError('Checkpoint pointer mismatch')
        model,opt,state=safety.load_checkpoint(folder,device)
    else:
        model=MaskablePPO.load(STUDENT,device=device);opt=torch.optim.Adam(safety.actor(model),lr=3e-6)
        state=dict(epoch=1,next_batch=0,updates=0,history=[dict(epoch=0,metrics=monitor(model,data,device))]);folder=safety.save_checkpoint(model,opt,state,'initial')
    baseline=state['history'][0]['metrics'];critic={k:v.clone() for k,v in model.policy.value_net.state_dict().items()}
    while state['epoch']<=2 and not state['history'][-1].get('stop',False):
        epoch=state['epoch'];losses=state.setdefault('epoch_losses',[]);grads=state.setdefault('epoch_grads',[])
        for index in range(state['next_batch'],128):
            opt.zero_grad(set_to_none=True);sequences=random.choices(data['train']['sequences'],k=4)
            frames=[f for seq in sequences for f in seq if f['action']!=f['student_action']]
            ce_value=0.
            for start in range(0,len(frames),16):
                part=frames[start:start+16];current,_,actions=safety.logits(model.policy,part,device)
                loss=-current.gather(1,actions[:,None]).sum()/len(frames);loss.backward();ce_value+=float(loss.detach())
            normal=random.choices(data['train']['normal'],k=32)
            current,_,_=safety.logits(model.policy,normal,device)
            with torch.no_grad():target,prob,_=safety.logits(teacher.policy,normal,device)
            loss=4*(prob*(target-current)).sum(1).mean()-.5*current.gather(1,target.argmax(1)[:,None]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite ordinary retention loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(safety.actor(model),.5,error_if_nonfinite=True);opt.step()
            losses.append(ce_value+float(loss.detach()));grads.append(float(norm));state.update(updates=state['updates']+1,next_batch=index+1)
            if state['updates']==2 or state['updates']%32==0:
                guard()
                if any(not torch.equal(v,critic[k]) for k,v in model.policy.value_net.state_dict().items()):raise RuntimeError('Critic changed')
                folder=safety.save_checkpoint(model,opt,state,f'update_{state["updates"]:06d}')
                if state['updates']==2:
                    expected={k:v.clone() for k,v in model.policy.state_dict().items()};model,opt,state=safety.load_checkpoint(folder,device)
                    if any(not torch.equal(v,expected[k]) for k,v in model.policy.state_dict().items()):raise RuntimeError('Save/load mismatch')
                    record(RUN/'gradient_smoke.json',dict(real_updates=2,passed=True,optimizer_steps=[int(v['step']) for v in opt.state.values()]))
        measured=monitor(model,data,device);stop=measured['correction']['loss']>baseline['correction']['loss']*.99 or measured['normal']['top1']<.95
        state['history'].append(dict(epoch=epoch,metrics=measured,stop=stop,loss=float(np.mean(losses)),gradient=float(np.mean(grads))))
        state.update(epoch=epoch+1,next_batch=0,epoch_losses=[],epoch_grads=[]);folder=safety.save_checkpoint(model,opt,state,f'epoch_{epoch:02d}')
        record(RUN/'training.json',dict(model=str(folder/'model.zip'),sha256=digest(folder/'model.zip'),updates=state['updates'],history=state['history']));operation(f'epoch{epoch} complete; updates={state["updates"]}; stop={stop}')
    candidate=folder/'model.zip';candidate.chmod(0o444);return candidate

def evaluate(candidate,seeds,cap,label):
    paths=dict(bc=str(BC),student=str(STUDENT),safety=str(SAFETY),candidate=str(candidate));identity=dict(protocol=digest(RUN/'protocol.json'),models={k:digest(Path(p)) for k,p in paths.items()});folder=RUN/'evaluation'/label
    rows={n:[] for n in paths};pending=[]
    for seed in seeds:
        for name in paths:
            path=folder/name/f'seed_{seed}.json';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                r=read_record(path);evaluation.validate_row(r,name,seed,cap,identity);rows[name].append(r)
            else:pending.append((name,seed,path))
    with ProcessPoolExecutor(4,mp_context=get_context('spawn'),initializer=safety.eval_init,initargs=(paths,)) as pool:
        tasks={pool.submit(evaluation.eval_game,n,s,cap,identity,p.with_suffix('.checkpoint.pkl')):(n,s,p) for n,s,p in pending}
        for f in as_completed(tasks):
            n,s,p=tasks[f];r=f.result();evaluation.validate_row(r,n,s,cap,identity);record(p,r);rows[n].append(r);status('evaluation',label=label,completed=sum(map(len,rows.values())),total=len(seeds)*4);guard()
    rows={n:sorted(v,key=lambda r:r['seed']) for n,v in rows.items()}
    result=dict(summary={n:evaluation.summary(v) for n,v in rows.items()},paired={n:evaluation.paired(rows['bc'],v) for n,v in rows.items() if n!='bc'},candidate_vs_student=evaluation.paired(rows['student'],rows['candidate']))
    record(folder/'summary.json',result);return result

def gate(r,long=False):
    a,b=r['summary']['bc'],r['summary']['candidate'];p=r['paired']['candidate']
    c=dict(survival=b['mean_pieces']>=a['mean_pieces']*(.95 if long else .98),early=b['early']<=a['early'] if long else b['early']==0,score=p['common_score_ratio']>=(1.08 if long else 1.10),seeds=p['positive_seeds']>=math.ceil(a['games']*.8125),high=b['high_fraction']<=a['high_fraction']+.02,holes=b['new_holes_per_100']<=max(1.5*a['new_holes_per_100'],a['new_holes_per_100']+.5),streak=b['mean_max_high_streak']<=max(1.5*a['mean_max_high_streak'],a['mean_max_high_streak']+50))
    if long:c['p10']=b['p10']>=a['p10']*.9
    else:c['capped']=b['capped']>=31
    return dict(passed=all(c.values()),checks=c)

def run(protocol):
    status('training');candidate=train(protocol);reason='stopped_short32'
    for label,key,cap,long in (('short32','short_seeds',20000,False),('long16','long_seeds',100000,True),('confirmation32','confirm_seeds',100000,True)):
        r=evaluate(candidate,protocol[key],cap,label);g=gate(r,long);record(RUN/f'{label}_gate.json',g);operation(f'{label} gate={g}');reason='stopped_'+label
        if not g['passed']:break
        if label=='confirmation32':reason='confirmed'
    from training.bc_recovery_sequence_report import finish
    finish(reason)
