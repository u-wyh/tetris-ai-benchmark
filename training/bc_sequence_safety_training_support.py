"""Read-only, bounded audit of the BC champion's actual supervised training corpus."""
import gzip
import json
import os
from collections import Counter
from pathlib import Path

from training.bc_sequence_safety import RUN,ROOT,BC,SEEDS,visible,read_blob,read_record,record,digest


def compute():
    provenance=BC.parent/'provenance.json';info=json.loads(provenance.read_text());cfg=info['bc_config']
    folder=Path(cfg['dataset']);path=RUN/'bc_supervised_support.json'
    identity=dict(source=digest(Path(__file__)),provenance=digest(provenance),bc=digest(BC))
    if path.exists():
        result=read_record(path)
        if result['identity']!=identity:raise RuntimeError('Training-support audit identity changed')
        return result
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    counts=Counter();support=Counter();height_buckets=[0]*5;shards={};n=0
    for seed in cfg['train_seeds']:
        meta_path=folder/f'seed_{seed}.json';meta=json.loads(meta_path.read_text());shard=folder/f'seed_{seed}.jsonl.gz'
        if digest(shard)!=meta['sha256'] or meta['teacher_hash']!=cfg['teacher_hash']:raise RuntimeError('Original BC corpus hash/teacher mismatch')
        shards[str(seed)]=dict(sha256=meta['sha256'],meta_sha256=digest(meta_path));height_buckets=[a+b for a,b in zip(height_buckets,meta['height_buckets_0_3_4_7_8_11_12_15_16_plus'])]
        with gzip.open(shard,'rt') as stream:
            for line in stream:
                row=json.loads(line)
                if row['seed']!=seed or row['teacher_hash']!=cfg['teacher_hash']:raise RuntimeError('BC corpus row identity mismatch')
                risk=visible(row['state']);counts[risk['tier']]+=1;counts['spawn_tight']+=risk['spawn']>0
                support[(risk['height']//2,min(risk['holes'],10),risk['well']//2)]+=1;n+=1
                if n>250000:raise RuntimeError('Bounded corpus audit exceeded250K known frames')
    novelty={}
    for tier in ('severe','moderate'):
        total=outside=rare=0
        for seed in SEEDS:
            for snap in read_blob(RUN/'episodes'/f'seed_{seed}.pkl')['snapshots']:
                r=read_blob(snap)['risk']
                if r['tier']!=tier:continue
                count=support[(r['height']//2,min(r['holes'],10),r['well']//2)];total+=1;outside+=count==0;rare+=count/max(1,n)<.0001
        novelty[tier]=dict(states=total,outside_support=outside,training_bin_frequency_below_1e_4=rare)
    result=dict(identity=identity,corpus=str(folder),train_episodes=len(cfg['train_seeds']),frames=n,risk_counts=dict(counts),height_buckets=height_buckets,
        new_snapshot_support=novelty,shards=shards,scope='actual BC supervised-stage train episodes only; earlier20M PPO full state corpus unavailable; no strict full-history OOD claim')
    record(path,result);return result


def append():
    r=compute();marker='\n## BC原始监督训练语料的支持检查\n';path=ROOT/'reports/experiments/bc_sequence_safety_results.md'
    text=path.read_text().split(marker)[0]+marker+'\n'
    text+=f'从冻结冠军provenance定位并校验 {r["train_episodes"]} 个实际监督训练episode、{r["frames"]} 帧，按teacher/hash逐shard验证，未修改原语料。'
    text+=f'公开危险计数：`{r["risk_counts"]}`；高度桶0–3/4–7/8–11/12–15/16+：`{r["height_buckets"]}`。\n\n'
    text+=f'新预定危险快照对该真实监督语料的粗分箱支持：`{r["new_snapshot_support"]}`。'
    text+='这比BC稳定回放代理更接近原监督训练分布，但不包含早期20M PPO完整轨迹；粗分箱也不能代替完整状态OOD判定。无新增训练/评测episode。\n';path.write_text(text)
    path=ROOT/'reports/experiments/bc_sequence_safety_evidence.json';evidence=json.loads(path.read_text());evidence['actual_bc_supervised_support']=r;path.write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    import sys
    if '--append' in sys.argv:append()
    else:
        r=compute();print(json.dumps({k:v for k,v in r.items() if k!='shards'},ensure_ascii=False))
