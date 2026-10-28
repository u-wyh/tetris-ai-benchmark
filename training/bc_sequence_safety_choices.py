"""Descriptive public action preferences on all prespecified new risk snapshots."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from training.bc_sequence_safety import RUN,ROOT,BC,STUDENT,SEEDS,read_blob,read_record,record,digest,logits


def compute():
    if any(not (RUN/'episodes'/f'seed_{s}.pkl').exists() for s in SEEDS):
        raise RuntimeError('Wait for all24 prespecified episodes; no partial-sample preference summary')
    files=[Path(p) for s in SEEDS for p in read_blob(RUN/'episodes'/f'seed_{s}.pkl')['snapshots']]
    identity=dict(source=digest(Path(__file__)),bc=digest(BC),student=digest(STUDENT),snapshots={p.name:digest(p) for p in files})
    path=RUN/'mechanism_choices.json'
    if path.exists():
        result=read_record(path)
        if result['identity']!=identity:raise RuntimeError('Frozen descriptive sample/source identity changed')
        return result
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:]);torch.set_num_threads(1)
    rows=[read_blob(p) for p in files];result=dict(identity=identity,groups={},interpretation='public one-placement features; not future survival evidence; groups overlap')
    models={n:MaskablePPO.load(p,device='cpu') for n,p in [('bc',BC),('student',STUDENT)]}
    for group,predicate in [('severe',lambda r:r['tier']=='severe'),('moderate',lambda r:r['tier']=='moderate'),
                             ('spawn_tight',lambda r:r['spawn']>0),('extreme_well',lambda r:r['well']>=10)]:
        selected=[r for r in rows if predicate(r['risk'])];result['groups'][group]=dict(states=len(selected),models={})
        for name,model in models.items():
            features=[]
            with torch.inference_mode():
                for i in range(0,len(selected),16):
                    frames=[dict(r['frame'],action=int(r['frame']['legal_ids'][0])) for r in selected[i:i+16]]
                    scores,_,_=logits(model.policy,frames,'cpu');actions=scores.argmax(1).cpu().numpy()
                    for frame,action in zip(frames,actions):
                        index=int(np.flatnonzero(np.asarray(frame['legal_ids'])==action)[0]);features.append(frame['legal_features'][index])
            result['groups'][group]['models'][name]=dict(samples=len(features),means={
                label:float(np.mean([row[index] for row in features])) if features else None
                for label,index in [('hold',1),('clear',5),('holes_after',6),('height_after',8),('wells_after',10),('new_holes',11),('top4_after',14),('lock_out',15)]})
    record(path,result);return result


def append():
    result=compute()
    marker='\n## 公开危险状态下的动作偏好（描述性）\n'
    lines=[marker,'','本表使用全部预定危险快照，未按反事实成功筛选。分组可重叠；候选单步公开特征不能当作未来生存证明。', '',
        '|状态组|模型|状态数|选择Hold比例|即时消行|落子后高度|新增洞|顶四行占据|Lock Out特征|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for group,g in result['groups'].items():
        for name,r in g['models'].items():
            m=r['means'];values=['—' if m[k] is None else f'{m[k]:.3f}' for k in ('hold','clear','height_after','new_holes','top4_after','lock_out')]
            lines.append(f'|{group}|{name}|{r["samples"]}|'+'|'.join(values)+'|')
    path=ROOT/'reports/experiments/bc_sequence_safety_results.md';path.write_text(path.read_text().split(marker)[0]+'\n'.join(lines)+'\n')
    path=ROOT/'reports/experiments/bc_sequence_safety_evidence.json';evidence=json.loads(path.read_text());evidence['public_action_preferences']=result
    path.write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    import sys
    if '--append' in sys.argv:append()
    else:print(json.dumps(compute()['groups'],ensure_ascii=False))
