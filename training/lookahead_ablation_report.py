"""Paired seed statistics and bounded automatic closeout; no configuration selection."""
import json
import math
import subprocess
import sys
import time
import urllib.request
import numpy as np
import training.lookahead_ablation as study
from training.bc_sequence_transfer import record,read_record
from training.evaluation.hybrid_queue import digest
from training.train_ppo import now

SHORT_PAIRS=[('A','B'),('B','C'),('D','E'),('C','E'),('B','D'),('A','E')]
LONG_PAIRS=[('C','E'),('D','E')]

def paired(left,right):
    seeds=sorted(set(left)&set(right))
    if not seeds:return dict(games=0,status='not_completed')
    pairs=[]
    for seed in seeds:
        a,b=left[seed],right[seed];n=min(a['pieces'],b['pieces']);x,y=int(a['scores'][n]),int(b['scores'][n])
        pairs.append(dict(seed=seed,common_pieces=n,baseline_score=x,candidate_score=y,ratio=y/x if x else None,
            baseline_pieces=a['pieces'],candidate_pieces=b['pieces'],survival_delta=b['pieces']-a['pieces'],baseline_death=a['game_over'],candidate_death=b['game_over']))
    d=np.asarray([(p['baseline_score'],p['candidate_score'],p['survival_delta']) for p in pairs],float)
    rng=np.random.default_rng(20261030);boot=d[rng.integers(0,len(d),(2000,len(d)))].sum(axis=1)
    ratio=float(d[:,1].sum()/d[:,0].sum());ci=np.percentile(boot[:,1]/boot[:,0],[2.5,97.5]).tolist();surv=np.percentile(boot[:,2]/len(d),[2.5,97.5]).tolist()
    loo=[float((d[:,1].sum()-r[1])/(d[:,0].sum()-r[0])) for r in d] if len(d)>1 else []
    positives=sum(p['candidate_score']>p['baseline_score'] for p in pairs)
    deaths=[sum(p[k] for p in pairs) for k in ('baseline_death','candidate_death')]
    early=[sum(row[seed]['game_over'] and row[seed]['pieces']<10000 for seed in seeds) for row in (left,right)]
    return dict(games=len(seeds),pairs=pairs,common_score_ratio=ratio,positive_seeds=positives,ratio_ci95=ci,
        observed_survival_delta=float(d[:,2].mean()),survival_difference_ci95=surv,deaths=deaths,early=early,
        leave_one_out_ratio_range=[min(loo),max(loo)] if loo else None,
        replicated_score_support=bool(len(seeds)>=4 and ci[0]>1 and positives>=math.ceil(.75*len(seeds))),
        observed_survival_support=bool(len(seeds)>=4 and surv[0]>0 and deaths[1]<deaths[0] and early[1]<=early[0]))

def load_rows(label,seeds,names):
    study.IDENTITY=digest(study.RUN/'protocol.json');cap=20000 if label=='short' else 100000;rows={n:{} for n in names}
    for name in names:
        for seed in seeds:
            path=study.RUN/'evaluation'/label/name/f'seed_{seed}.json'
            if not path.exists():continue
            r=study.completed(path,name,seed,cap);r.pop('traces',None)
            with np.load(path.with_suffix('.npz')) as artifact:r.update(scores=artifact['scores'],latencies_ms=artifact['latencies_ms'])
            rows[name][seed]=r
    return rows

def summarize_stage(label,seeds,names):
    rows=load_rows(label,seeds,names);summary={}
    for name,values in rows.items():
        if not values:continue
        rr=list(values.values());s=study.evaluation.summary(rr);n=sum(r['pieces'] for r in rr)
        lat=np.concatenate([r['latencies_ms'] for r in rr]);s.update(early=sum(r['game_over'] and r['pieces']<10000 for r in rr),mean_latency_ms=float(lat.mean()),p95_ms=float(np.percentile(lat,95)),p99_ms=float(np.percentile(lat,99)),mean_height=sum(r['height_sum'] for r in rr)/n,mean_holes=sum(r['holes_sum'] for r in rr)/n,
            overrides=sum(r['overrides'] for r in rr),override_fraction=sum(r['overrides'] for r in rr)/n,third_actions=sum(r['third_actions'] for r in rr),third_fraction=sum(r['third_actions'] for r in rr)/n,
            mechanism={k:{key:sum(r[k].get(key,0) for r in rr) for key in sorted({key for r in rr for key in r[k]})} for k in ('root_membership_counts','stop_counts','next_consumed_counts')},
            cost_component_mean={k:sum(r['cost_component_sum'][k] for r in rr)/n for k in study.COST_FIELDS})
        summary[name]=s
    comparisons={f'{b}_vs_{a}':paired(rows[a],rows[b]) for a,b in (SHORT_PAIRS if label=='short' else LONG_PAIRS)}
    comparisons.update({f'{n}_vs_BC':paired(rows['BC'],rows[n]) for n in names if n!='BC'})
    result=dict(label=label,planned_seeds=seeds,completed_seed_ids={n:sorted(v) for n,v in rows.items()},summary=summary,comparisons=comparisons,
        complete=all(len(v)==len(seeds) for v in rows.values()))
    record(study.RUN/f'{label}_summary.json',result);return result

def health():
    values={}
    for port,suffix in ((8080,'/'),(8081,'/api/health')):
        try:
            with urllib.request.urlopen(f'http://100.91.227.47:{port}{suffix}',timeout=5) as r:values[str(port)]=r.status
        except Exception as exc:values[str(port)]=repr(exc)
    return values

def description(c):
    if not c or not c.get('games'):return '没有完整配对结果，无法判断。'
    scope='完整16seed' if c['games']==16 else f"仅{c['games']}个已完成配对seed（整组未完成）"
    return (f"{scope}，共同正式得分{100*(c['common_score_ratio']-1):+.3f}%，正向{c['positive_seeds']}/{c['games']}，"
        f"95%分比区间{c['ratio_ci95']}；观察生存差{c['observed_survival_delta']:+.1f}块，区间{c['survival_difference_ci95']}，"
        f"真实死亡{c['deaths'][0]}→{c['deaths'][1]}、<10K {c['early'][0]}→{c['early'][1]}。"
        f"跨seed得分支持={c['replicated_score_support']}，已观察生存支持={c['observed_survival_support']}。")

def finish(reason,error=None):
    if (study.RUN/'closed.json').exists():return
    study.status('closing',reason=reason);p=study.freeze()
    short=summarize_stage('short',p['groups']['short'],['A','B','C','D','E','BC'])
    long=summarize_stage('long',p['groups']['long'],['C','D','E','BC'])
    functional=read_record(study.RUN/'functional.json') if (study.RUN/'functional.json').exists() else None
    lines=['# Public Lookahead 消融：实测结果','',f'完成{now()}；结束原因`{reason}`；错误`{error}`。','',
        '**本轮只解释冻结组合差异，零训练、零调参、没有新独立质量确认或线上更新。**','',
        f'功能状态{functional["states"] if functional else 0}，E与原动作/trace完全一致{functional["full_equivalence_states"] if functional else 0}，正式模拟前缀{functional["simulation_checks"] if functional else 0}，隐藏扰动/确定性检查{functional["hidden_checks"] if functional else 0}。完整配置动作一致不等于其他消融安全或得分更好。']
    for stage in (short,long):
        lines+=['',f'## {stage["label"]}：整组完成={stage["complete"]}','','|组|完成局|均值观测生存|P10|达到上限（右截断）|死亡/<10K|平均总分|四消事件占比|平均/P95/P99 ms|','|---|---:|---:|---:|---:|---|---:|---:|---|']
        for n,s in stage['summary'].items():lines.append(f'|{n}|{s["games"]}/16|{s["mean_pieces"]:.1f}|{s["p10"]:.1f}|{s["capped"]}|{s["deaths"]}/{s["early"]}|{s["mean_score"]:.1f}|{s["four_fraction"]:.4%}|{s["mean_latency_ms"]:.3f}/{s["p95_ms"]:.3f}/{s["p99_ms"]:.3f}|')
        lines+=['','缺失配置/seed：`'+str({n:sorted(set(stage['planned_seeds'])-set(done)) for n,done in stage['completed_seed_ids'].items()})+'`。各组原始均值只基于已完成局，不用不同seed子集直接判胜负；下面仅对完整结果的seed交集配对。','',
            '|对比（前者/后者）|配对seed|共同正式分变化|正向|95%分比区间|观测生存差/95%区间|死亡 基线→候选|删任一seed分比范围|','|---|---:|---:|---:|---|---|---|---|']
        for key,c in stage['comparisons'].items():
            if not c['games']:lines.append(f'|{key}|0|未完成|—|—|—|—|—|');continue
            lines.append(f'|{key}|{c["games"]}|{100*(c["common_score_ratio"]-1):+.3f}%|{c["positive_seeds"]}|{c["ratio_ci95"]}|{c["observed_survival_delta"]:+.1f} / {c["survival_difference_ci95"]}|{c["deaths"]}|{c["leave_one_out_ratio_range"]}|')
        lines+=['','|组|平均高度/洞|新洞/100|高度>=17比例|最长高位串均值|覆盖次数/比例|第三类次数/比例|','|---|---|---:|---:|---:|---|---|']
        for n,s in stage['summary'].items():lines.append(f'|{n}|{s["mean_height"]:.3f}/{s["mean_holes"]:.3f}|{s["new_holes_per_100"]:.3f}|{s["high_fraction"]:.5%}|{s["mean_max_high_streak"]:.2f}|{s["overrides"]}/{s["override_fraction"]:.4%}|{s["third_actions"]}/{s["third_fraction"]:.4%}|')
        lines+=['','### 逐seed配对明细','', '|对比|seed|共同块数|基线正式分|候选正式分|存活差|','|---|---:|---:|---:|---:|---:|']
        for key,c in stage['comparisons'].items():
            for pair in c.get('pairs',[]):lines.append(f'|{key}|{pair["seed"]}|{pair["common_pieces"]}|{pair["baseline_score"]}|{pair["candidate_score"]}|{pair["survival_delta"]}|')
        lines+=['','### 机制聚合（自身访问状态，描述性，非逐块独立统计）']
        for n,s in stage['summary'].items():lines+=['',f'**{n}**：根成员重叠计数`{s["mechanism"]["root_membership_counts"]}`；停止原因`{s["mechanism"]["stop_counts"]}`；公开Next消耗`{s["mechanism"]["next_consumed_counts"]}`；选中根分项均值`{s["cost_component_mean"]}`。']
    examples=[]
    for row in (functional or {}).get('rows',[]):
        if len(set(row['actions'].values()))>1:examples.append(row)
        if len(examples)==8:break
    lines+=['','## 预定共同局面样例：按时间顺序最早8个差异','','|seed/步|A|B|C|D|E|E理由|E实际深度/消耗Next/终止|E末端几何/路径最高几何/得分项/总value|','|---|---:|---:|---:|---:|---:|---|---|---|']
    for row in examples:
        a=row['actions'];t=row['traces']['E'];b=t['selected'];lines.append(f'|{row["seed"]}/{row["piece"]}|{a["A"]}|{a["B"]}|{a["C"]}|{a["D"]}|{a["E"]}|{t["reason"]}|{len(b["path"])}/{b["known_tokens_consumed"]}/{b["stop_reason"]}|{b["final_geometry"]:.2f}/{b["max_geometry"]:.2f}/{b["score_contribution"]:.2f}/{b["value"]:.2f}|')
    lines+=['','## 解释限制','','A使用Next3神经特征；B/D深度1仍有落子后出生和Hold的公开Next消费。深度对比不是纯Next3消融。额外几何/即时消行是组合候选，BC与一步评价也不能完全拆分，几何/正式分权重没有被单独消融。动作差异、分项代价和一条rollout成败不证明逐动作因果优劣。',
        'Bootstrap按seed抽样，非逐块独立样本；95%区间未多重校正。所有上限是右截断，均值/P10仅为min(寿命,cap)，[0,0]观测生存差不能证明无限或真实寿命相同。删一个seed范围只检验单seed敏感性，不代替新确认。',
        'NPZ保存每步动作/公开Next/全部根分项，详细trace采样按固定协议；真实Core/RNG仅用于正式对局与归档，不输入规划。未完成局保留checkpoint但不冒充死亡/完整上限结果。']
    folder=study.ROOT/'reports/experiments';(folder/'lookahead_ablation_results.md').write_text('\n'.join(lines)+'\n')
    sc=short['comparisons'];lc=long['comparisons'];prefer=lambda k:lc.get(k) if lc.get(k,{}).get('games') else sc.get(k)
    final=['# Public Lookahead 消融：最终结论','',f'结束{now()}，原因`{reason}`；短组完整={short["complete"]}，长组完整={long["complete"]}。','',
        '1. **额外BC候选是否有贡献？** '+description(sc.get('B_vs_A'))+' 这是BC候选与一步评价的组合，不足以分离纯BC贡献；没有把BC视作绝对安全。','',
        '2. **几何与即时消行候选是否有贡献？** 深度4的E/C：'+description(prefer('E_vs_C'))+' 深度1的D/B：'+description(sc.get('D_vs_B'))+' 两类候选合并评价，不能给其中单项分配功劳。','',
        '3. **公开未来规划深度是否有贡献？** 两根C/B：'+description(sc.get('C_vs_B'))+' 四根E/D：'+description(prefer('E_vs_D'))+' 这同时改变延伸步数、已知未来利用和累计代价范围，不能称纯Next3贡献。','',
        '4. **哪一部分贡献较多得分或生存优势？** 上述固定根集合的深度对比和固定深度的根集合对比分别给出可观察组合效应；E/A整体净效应：'+description(sc.get('E_vs_A'))+' 各效应不能相加，权重和候选身份重叠尚未拆分，不依据分项均值宣称严格因果贡献排名。','',
        '5. **跨seed证据和不确定性？** 以各对比配对seed数、正向数、Bootstrap区间、死亡数和删任一seed范围为准；表内“支持”采用冻结的描述性规则，不是新算法确认。部分完成组不冒充16seed；所有上限右截断，无法推出上限后寿命。','',
        '6. **哪些配置明显增加延迟？** 下面列同一阶段实测均值/P95/P99；深度/根数量的额外CPU成本须与相应配对收益共同判断，不与旧32seed研究耗时直接等同。']
    for r in (short,long):
        final+=['',r['label']+'：'+'；'.join(f'{n} {s["mean_latency_ms"]:.3f}/{s["p95_ms"]:.3f}/{s["p99_ms"]:.3f}ms' for n,s in r['summary'].items())]
    final+=['','7. **是否值得继续增加搜索深度？** 本轮只比较1/4，不能外推更深会更好。按本轮信息边界，Current与公开Next耗尽即停止，Hold还可能更早触及未知边界；机械增加超过4的上限不产生可合法使用的新未来信息。本轮不扩深度或调参。','',
        '8. **是否值得蒸馏成单网络？** 本轮没有训练或验证蒸馏。若完整配对显示Full相对学生/浅层的净收益且覆盖具有可重复模式，可据此提出另一份有限预算协议；这些动作日志不是自动成立的强安全标签，不能声称蒸馏会保留生存收益。本轮到此结束。','',
        '**模型、原规划器、游戏规则、线上8080/8081和默认AI均未修改；没有登记冠军、main合并或发布。**','',
        '[冻结设计](lookahead_ablation_design.md) · [完整结果与逐seed表](lookahead_ablation_results.md) · [机器证据](lookahead_ablation_evidence.json)']
    (folder/'lookahead_ablation_final.md').write_text('\n'.join(final)+'\n')
    services=health()
    evidence=dict(at=now(),reason=reason,error=error,protocol=p,short=short,long=long,services=services,retained_bytes=study.disk_bytes(),functional={k:v for k,v in (functional or {}).items() if k!='rows'},examples=examples)
    (folder/'lookahead_ablation_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    commands=[[sys.executable,'-m','pytest','-q','tests/test_lookahead_ablation.py','tests/test_public_lookahead.py','tests/test_tetris_env.py','tests/test_web_inference.py','tests/parity'],['node','--test',*map(str,sorted((study.ROOT/'tests').glob('*.test.cjs')))]]
    codes=[]
    for i,cmd in enumerate(commands):
        with (study.RUN/f'regression_{i}.log').open('w') as f:
            try:codes.append(subprocess.run(cmd,cwd=study.ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=600).returncode)
            except subprocess.TimeoutExpired:codes.append(124)
    record(study.RUN/'regression.json',dict(at=now(),commands=commands,returncodes=codes))
    with (folder/'lookahead_ablation_final.md').open('a') as f:f.write(f'\nPython/Parity、JS退出码{codes}；最终服务健康{services}；保留数据{study.disk_bytes()}字节。\n')
    study.operation(f'final reason={reason}; short complete={short["complete"]}, long complete={long["complete"]}; regressions={codes}; services={services}; no training or production changes')
    with (study.ROOT/'CHANGELOG.md').open('a') as f:f.write(f'\n### Public Lookahead ablation ({now()[:10]})\n\n- Completed bounded A/B/C/D/E mechanism study; reason={reason}, short complete={short["complete"]}, long complete={long["complete"]}. Paired seed Bootstrap, component traces and right-censored results archived; no planner weights, models or web defaults changed. See reports/experiments/lookahead_ablation_final.md.\n')
    subprocess.run(['git','add','CHANGELOG.md','OPERATIONS.md','reports/experiments/lookahead_ablation_results.md','reports/experiments/lookahead_ablation_final.md','reports/experiments/lookahead_ablation_evidence.json'],cwd=study.ROOT,check=True)
    subprocess.run(['git','commit','-m','Report frozen Lookahead ablations and paired mechanism evidence'],cwd=study.ROOT,check=True)
    push=None
    for attempt in range(2):
        try:
            subprocess.run(['git','push','-u','origin','codex/lookahead-ablation'],cwd=study.ROOT,check=True,timeout=90);push=True;break
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:push=repr(exc)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=study.ROOT,text=True).strip()
    record(study.RUN/'closed.json',dict(at=now(),reason=reason,error=error,git=head,pushed=push,regressions=codes))
    study.status('completed' if push is True else 'push_failed',reason=reason,git=head)
