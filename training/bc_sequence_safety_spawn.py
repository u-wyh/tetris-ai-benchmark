"""Retrospective public spawn-space action audit; no training or new games."""
import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from training.bc_sequence_safety import RUN,ROOT,BC,STUDENT,read_record,record,digest,logits


def compute():
    provenance=BC.parent/'provenance.json';cfg=json.loads(provenance.read_text())['bc_config'];folder=Path(cfg['dataset'])
    frames=[];shards={}
    for seed in cfg['train_seeds']:
        meta=json.loads((folder/f'seed_{seed}.json').read_text())
        if not meta['height_buckets_0_3_4_7_8_11_12_15_16_plus'][-1]:continue
        path=folder/f'seed_{seed}.jsonl.gz'
        if digest(path)!=meta['sha256']:raise RuntimeError('Original spawn-corpus hash mismatch')
        shards[str(seed)]=meta['sha256']
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line)
                if any(row['state'][:40]):frames.append(row)
                if len(frames)>1000:raise RuntimeError('Spawn audit exceeded1000 fixed historical public frames')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:]);torch.set_num_threads(1)
    candidate=Path(read_record(RUN/'training.json')['model']);paths=dict(bc=BC,student=STUDENT,candidate=candidate)
    models={n:MaskablePPO.load(p,device='cpu') for n,p in paths.items()}
    result=dict(scope='all spawn-tight states from actual historical BC supervised train episodes; retrospective teacher-forced only; excluded from safety training/development',frames=len(frames),source_sha256=digest(Path(__file__)),shards=shards,models={})
    if frames:
        with torch.inference_mode():
            a,prob,_=logits(models['bc'].policy,frames,'cpu');base=a.argmax(1)
            for name,model in models.items():
                scores,p,labels=logits(model.policy,frames,'cpu');actions=scores.argmax(1)
                features=[np.asarray(frame['legal_features'])[int(np.flatnonzero(np.asarray(frame['legal_ids'])==action)[0])]
                    for frame,action in zip(frames,actions.cpu().numpy())]
                result['models'][name]=dict(sha256=digest(paths[name]),bc_agreement=float((actions==base).float().mean()),historical_expert_agreement=float((actions==labels).float().mean()),
                    bc_to_model_kl=float((prob*(a-scores)).sum(1).mean()),entropy=float(-(p*scores).sum(1).mean()),
                    chosen_features={key:float(np.mean([f[index] for f in features])) for key,index in [('clear',5),('height_after',8),('new_holes',11),('top4_after',14),('lock_out',15)]})
    record(RUN/'spawn_mechanism.json',result);return result


def append():
    r=compute();marker='\n## 既有BC语料的出生空间动作检查\n';path=ROOT/'reports/experiments/bc_sequence_safety_results.md'
    text=path.read_text().split(marker)[0]+marker+'\n'
    text+=f'对实际BC监督训练语料中全部 {r["frames"]} 个顶四行占据状态，比较冻结BC、旧学生和固定安全候选。'
    text+='这些是历史训练状态的回顾性teacher-forced分析，未加入本轮训练或开发，不是新增生存验证，也不假定BC/历史专家永远正确。\n\n'
    text+='|模型|BC动作一致率|历史专家一致率|BC→模型KL|平均即时消行|落子后高度|顶四行占据|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name,m in r['models'].items():
        f=m['chosen_features'];text+=f'|{name}|{m["bc_agreement"]:.3%}|{m["historical_expert_agreement"]:.3%}|{m["bc_to_model_kl"]:.3f}|{f["clear"]:.3f}|{f["height_after"]:.3f}|{f["top4_after"]:.3f}|\n'
    path.write_text(text);path=ROOT/'reports/experiments/bc_sequence_safety_evidence.json';e=json.loads(path.read_text());e['retrospective_spawn_choices']=r;path.write_text(json.dumps(e,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    import sys
    if '--append' in sys.argv:append()
    else:print(json.dumps(compute(),ensure_ascii=False))
