"""Predeclared paired long-validation analysis; no tuning or automatic deployment."""
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import training.lookahead_long_validation as study
from training.bc_sequence_transfer import record,read_record
from training.evaluation.hybrid_queue import digest
from training.train_ppo import now

def paired(left,right):
    seeds=sorted(set(left)&set(right))
    if not seeds:return dict(games=0,status='not_completed')
    pairs=[]
    for seed in seeds:
        a,b=left[seed],right[seed];n=min(a['pieces'],b['pieces']);x,y=int(a['scores'][n]),int(b['scores'][n])
        pairs.append(dict(seed=seed,common_pieces=n,baseline_score=x,candidate_score=y,ratio=y/x if x else None,
            baseline_pieces=a['pieces'],candidate_pieces=b['pieces'],survival_delta=b['pieces']-a['pieces'],baseline_death=a['game_over'],candidate_death=b['game_over']))
    d=np.asarray([(p['baseline_score'],p['candidate_score'],p['survival_delta']) for p in pairs],float)
    rng=np.random.default_rng(20261031);boot=d[rng.integers(0,len(d),(5000,len(d)))].sum(axis=1)
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


def compare(left,right):
    result=paired(left,right)
    seeds=sorted(set(left)&set(right))
    if seeds:
        d=np.array([[left[s]['episode_p95_ms'],right[s]['episode_p95_ms']] for s in seeds])
        rng=np.random.default_rng(20261031)
        b=d[rng.integers(0,len(d),(5000,len(d)))].sum(axis=1)
        result.update(cpu_episode_pairs=[dict(seed=s,baseline_p95_ms=float(d[i,0]),candidate_p95_ms=float(d[i,1])) for i,s in enumerate(seeds)],mean_episode_p95_ratio=float(d[:,1].sum()/d[:,0].sum()),mean_episode_p95_ratio_ci95=np.percentile(b[:,1]/b[:,0],[2.5,97.5]).tolist(),cpu_ci_estimand='ratio of mean per-episode P95, not pooled decision P95')
    return result


def summarize(label,seeds,names,cap):
    study.IDENTITY=digest(study.RUN/'protocol.json');rows={name:{} for name in names};summaries={}
    for name in names:
        for seed in seeds:
            path=study.RUN/'evaluation'/label/name/f'seed_{seed}.json'
            if not path.exists():continue
            row=study.completed(path,name,seed,cap)
            row.pop('traces',None)
            with np.load(path.with_suffix('.npz')) as data:
                row['scores']=data['scores'];row['latencies_ms']=data['latencies_ms']
            row['episode_p95_ms']=float(np.percentile(row['latencies_ms'],95));rows[name][seed]=row
        rr=list(rows[name].values())
        if not rr:continue
        s=study.evaluation.summary(rr);lat=np.concatenate([r['latencies_ms'] for r in rr]);n=sum(r['pieces'] for r in rr)
        s.update(early=sum(r['game_over'] and r['pieces']<10000 for r in rr),p95_ms=float(np.percentile(lat,95)),p99_ms=float(np.percentile(lat,99)),mean_height=sum(r['height_sum'] for r in rr)/n,mean_holes=sum(r['holes_sum'] for r in rr)/n,override_fraction=sum(r['overrides'] for r in rr)/n,third_fraction=sum(r['third_actions'] for r in rr)/n)
        s['death_seed_locations']=[dict(seed=r['seed'],pieces=r['pieces']) for r in rr if r['game_over']]
        if label=='200k':
            s['later_deaths_after_100k']=[dict(seed=r['seed'],pieces=r['pieces']) for r in rr if r['game_over'] and r['pieces']>100000]
            extra=[r['latencies_ms'][r['continuation_start_pieces']:] for r in rr if len(r['latencies_ms'])>r['continuation_start_pieces']]
            if extra:
                x=np.concatenate(extra);s['extension_only_latency_ms']=dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)))
        summaries[name]=s
    contrasts=[('E','C'),('A','B'),('B','C'),('A','E')] if label=='100k' else [('E','C')]
    contrasts += [('BC',n) for n in names if n!='BC']
    comparisons={f'{b}_vs_{a}':compare(rows[a],rows[b]) for a,b in contrasts}
    result=dict(label=label,cap=cap,planned_seeds=seeds,completed_seed_ids={n:sorted(v) for n,v in rows.items()},complete=all(len(v)==len(seeds) for v in rows.values()),summary=summaries,comparisons=comparisons)
    record(study.RUN/f'{label}_summary.json',result)
    return result


def decision(main,extension):
    result=dict(engineering_candidate=False,recommend_independent_confirmation=False,survival_advantage='优势不确定',checks={},extension_opposition=None)
    if not main['complete'] or not extension['complete']:
        result['reason']='主实验或预定延长组未完成；不做积极准入或确认建议'
        return result
    c,e=main['summary']['C'],main['summary']['E'];pair=main['comparisons']['C_vs_E']
    xc,xe=extension['summary']['C'],extension['summary']['E'];xp=extension['comparisons']['C_vs_E']
    opposite=(xc['mean_pieces']<.95*xe['mean_pieces'] or xc['deaths']>=xe['deaths']+2 or xc['early']>xe['early'] or xp['survival_difference_ci95'][1]<0)
    checks=dict(observed_mean_survival=c['mean_pieces']>=e['mean_pieces'],deaths=c['deaths']<=e['deaths'],common_score=pair['common_score_ratio']>=.99,pooled_cpu_p95=c['p95_ms']<=e['p95_ms'],extension_no_obvious_opposition=not opposite)
    admitted=all(checks.values());survival=pair['survival_difference_ci95'];cost=pair['mean_episode_p95_ratio_ci95']
    reliable_tradeoff=(survival[0]>0 or (cost[1]<1 and pair['ratio_ci95'][0]>=.99))
    result.update(checks=checks,extension_opposition=opposite,engineering_candidate=admitted,reliable_tradeoff=reliable_tradeoff,recommend_independent_confirmation=admitted and reliable_tradeoff,survival_advantage='C优势有本组观察支持' if survival[0]>0 else ('E优势有本组观察支持' if survival[1]<0 else '优势不确定'),reason='按冻结规则；工程准入不等于统计安全优势，不自动执行确认或部署')
    return result


def description(result):
    if not result.get('games'):return '无完整配对结果，未完成。'
    return (f"n={result['games']}，共同正式得分{100*(result['common_score_ratio']-1):+.3f}%，正向{result['positive_seeds']}，95%分比区间{result['ratio_ci95']}；观察生存差{result['observed_survival_delta']:+.1f}块，区间{result['survival_difference_ci95']}；真实死亡基线→候选{result['deaths']}，<10K{result['early']}。")


def health():
    values={}
    for port,suffix in ((8080,'/'),(8081,'/api/health')):
        try:
            with urllib.request.urlopen(f'http://100.91.227.47:{port}{suffix}',timeout=5) as r:values[str(port)]=r.status
        except Exception as exc:values[str(port)]=repr(exc)
    return values


def finish(reason,error=None):
    if (study.RUN/'closed.json').exists():return
    study.status('closing',reason=reason);protocol=study.freeze()
    main=summarize('100k',protocol['groups']['main'],['A','B','C','E','BC'],100000)
    extension=summarize('200k',protocol['groups']['extension'],['BC','C','E'],200000)
    d=decision(main,extension);record(study.RUN/'decision.json',d)
    functional=read_record(study.RUN/'functional.json') if (study.RUN/'functional.json').exists() else {}
    folder=study.ROOT/'reports/experiments'
    lines=['# Public Lookahead 长期验证：结果','',f'时间{now()}；结束原因`{reason}`；错误`{error}`；源Git`{protocol["git"]}`。','',
        '新独立32-seed100K主组；事前选定其中16seed作200K同轨迹延长。两个上限分开报告，不能当成48个独立seed。未修改策略、权重、Core或服务。','',
        f'功能验证：`{ {k:v for k,v in functional.items() if k!="rows"} }`。',
        '', '得分为同seed各策略共同存活块数时正式累计分的比值；主指标为各seed累计分求和后取比。Bootstrap5000次、seed为单位、RNG20261031，区间探索性且未做多重比较校正。CPU区间针对seed平均的单局P95比；工程门槛针对全部决策汇总P95，两者不混用。']
    for group in (main,extension):
        lines += ['',f'## {group["label"]}：完整={group["complete"]}，上限{group["cap"]}', '', '|策略|完成|已观察平均生存|P10|触顶右截断|真实死亡 / <10K|平均正式总分|平均/P95/P99 ms|', '|---|---:|---:|---:|---:|---|---:|---|']
        for name,s in group['summary'].items():
            lines.append(f'|{name}|{s["games"]}/{len(group["planned_seeds"])}|{s["mean_pieces"]:.1f}|{s["p10"]:.1f}|{s["capped"]}|{s["deaths"]}/{s["early"]}|{s["mean_score"]:.1f}|{s["mean_inference_ms"]:.3f}/{s["p95_ms"]:.3f}/{s["p99_ms"]:.3f}|')
        lines += ['', '|对比|共同分变化|正向 / 配对seed|95%分比区间|生存差 / 95%区间|死亡基线→候选|删任一seed分比范围|', '|---|---:|---|---|---|---|---|']
        for key,p in group['comparisons'].items():
            if not p.get('games'):lines.append(f'|{key}|未完成||||||');continue
            lines.append(f'|{key}|{100*(p["common_score_ratio"]-1):+.3f}%|{p["positive_seeds"]}/{p["games"]}|{p["ratio_ci95"]}|{p["observed_survival_delta"]:+.1f} / {p["survival_difference_ci95"]}|{p["deaths"]}|{p["leave_one_out_ratio_range"]}|')
        for key,p in group['comparisons'].items():
            if p.get('games'):lines+=['',f'{key}：单局P95均值比 {p["mean_episode_p95_ratio"]:.5f}，seed Bootstrap95%区间 {p["mean_episode_p95_ratio_ci95"]}（不是汇总P95区间）。']
        lines+=['', '|策略|平均高度 / 洞|高位>=17比例|单局最长高位串均值|四消事件比例|学生覆盖 / 第三类比例|', '|---|---|---:|---:|---:|---|']
        for name,s in group['summary'].items():lines.append(f'|{name}|{s["mean_height"]:.3f}/{s["mean_holes"]:.3f}|{s["high_fraction"]:.6%}|{s["mean_max_high_streak"]:.2f}|{s["four_fraction"]:.4%}|{s["override_fraction"]:.4%}/{s["third_fraction"]:.4%}|')
        lines+=['','### 所有真实死亡与后期观察','']
        for name,s in group['summary'].items():lines.append(f'- {name}: `{s["death_seed_locations"]}`；新增100K后死亡`{s.get("later_deaths_after_100k", "不适用")}`；延长部分单独延迟`{s.get("extension_only_latency_ms", "不适用/无新增块")}`。')
        lines+=['','### 每个seed的配对结果','', '|对比|seed|共同块数|基线 / 候选得分|基线 / 候选生存|生存差|', '|---|---:|---:|---|---|---:|']
        for key,p in group['comparisons'].items():
            for row in p.get('pairs',[]):lines.append(f'|{key}|{row["seed"]}|{row["common_pieces"]}|{row["baseline_score"]}/{row["candidate_score"]}|{row["baseline_pieces"]}/{row["candidate_pieces"]}|{row["survival_delta"]}|')
        lines+=['',f'完成seed：`{group["completed_seed_ids"]}`。部分结果只比较完整配对交集；未完成局保留checkpoint，不冒充死亡或完整结果。']
    lines+=['','## 冻结工程准入判定','',json.dumps(d,ensure_ascii=False,indent=2),'','生存为min(T,上限)，触顶右截断。区间跨零必须写优势不确定。200K主组包含100K已死亡者的原死亡结果，没有仅挑存活者重新开局。A/BC无搜索代价；额外候选和深度为组合效应，不能分离单项因果贡献。']
    (folder/'lookahead_long_validation_results.md').write_text('\n'.join(lines)+'\n')
    mc,ec=main['comparisons'],extension['comparisons'];a=main['summary'].get('A')
    final=['# Public Lookahead 长期验证：最终结论','',f'结束{now()}，原因`{reason}`；100K完整={main["complete"]}，200K完整={extension["complete"]}。','',
        '1. **C和E在新100K组谁更好？** '+description(mc['C_vs_E'])+' 生存判断：'+d['survival_advantage']+'。平均正式总分和P10见结果表，不能用共同分替代。','',
        '2. **200K揭示后期死亡吗？** '+description(ec['C_vs_E'])+' 各策略新增100K后死亡均完整列在结果报告；延长组与主组高度相关，不合并统计。','',
        f'3. **原学生A长期风险？** '+(f'新组{a["games"]}局观察到{a["deaths"]}次100K内真实死亡（{a["deaths"]/a["games"]:.1%}），<10K {a["early"]}，P10 {a["p10"]:.1f}。这是本组上限内死亡率，不能外推上限后风险。' if a else '未取得完整对局，无法判断。'),' ',
        '4. **BC根候选B是否帮助？** '+description(mc['B_vs_A'])+' 这是BC根＋一步评价＋覆盖规则的组合，不能分离纯BC贡献。','',
        '5. **增加深度是否有长期价值？** '+description(mc['C_vs_B'])+' 增加深度也改变已知未来利用和累计代价，不能全部归因Next3。','',
        f'6. **C能否替代E？** 工程候选准入={d["engineering_candidate"]}，检查`{d["checks"]}`；200K明显相反证据={d["extension_opposition"]}。准入不是安全优势证明，也没有独立最终确认。','',
        f'7. **建议另做一次独立确认吗？** 按冻结规则建议={d["recommend_independent_confirmation"]}。{d["reason"]}。本轮不会启动下一轮确认；不满足工程门槛或可靠取舍证据时，保留E并停止简化路线。','',
        '8. **线上E是否完全不变？** 是。没有训练、调参、切换线上控制器、替换冠军、main合并或发布；8080/8081保留。',
        '', '[冻结设计](lookahead_long_validation_design.md) · [完整指标及逐seed结果](lookahead_long_validation_results.md) · [机器证据](lookahead_long_validation_evidence.json)']
    (folder/'lookahead_long_validation_final.md').write_text('\n'.join(final)+'\n')
    services=health();evidence=dict(at=now(),reason=reason,error=error,protocol=protocol,functional={k:v for k,v in functional.items() if k!='rows'},main=main,extension=extension,decision=d,services=services,retained_bytes=study.disk_bytes())
    (folder/'lookahead_long_validation_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    commands=[[sys.executable,'-m','pytest','-q','tests/test_lookahead_long_validation.py','tests/test_lookahead_ablation.py','tests/test_public_lookahead.py','tests/test_tetris_env.py','tests/test_web_inference.py','tests/parity'],['node','--test',*map(str,sorted((study.ROOT/'tests').glob('*.test.cjs')))]]
    codes=[]
    for i,cmd in enumerate(commands):
        with (study.RUN/f'regression_{i}.log').open('w') as f:
            try:codes.append(subprocess.run(cmd,cwd=study.ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=600).returncode)
            except subprocess.TimeoutExpired:codes.append(124)
    record(study.RUN/'regression.json',dict(at=now(),commands=commands,returncodes=codes))
    with (folder/'lookahead_long_validation_final.md').open('a') as f:f.write(f'\nPython/Parity、JS退出码{codes}；服务{services}；研究数据{study.disk_bytes()}字节。\n')
    study.operation(f'final reason={reason}; main complete={main["complete"]}, extension complete={extension["complete"]}; admission={d["engineering_candidate"]}, recommend confirmation={d["recommend_independent_confirmation"]}; regressions={codes}; services={services}; no production change')
    with (study.ROOT/'CHANGELOG.md').open('a') as f:f.write(f'\n### Lookahead long validation ({now()[:10]})\n\n- Frozen A/B/C/E/BC paired100K32-seed study and preset16-seed BC/C/E same-trajectory200K extension; reason={reason}, main complete={main["complete"]}, extension complete={extension["complete"]}. Engineering admission={d["engineering_candidate"]}; no training or production changes. See reports/experiments/lookahead_long_validation_final.md.\n')
    paths=['CHANGELOG.md','OPERATIONS.md']+[str(p.relative_to(study.ROOT)) for p in folder.glob('lookahead_long_validation_*')]
    subprocess.run(['git','add',*paths],cwd=study.ROOT,check=True)
    subprocess.run(['git','commit','-m','Report paired long validation and exact trajectory extensions'],cwd=study.ROOT,check=True)
    pushed=None
    for attempt in range(2):
        try:subprocess.run(['git','push','-u','origin','codex/lookahead-long-validation'],cwd=study.ROOT,check=True,timeout=90);pushed=True;break
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:pushed=repr(exc)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=study.ROOT,text=True).strip()
    record(study.RUN/'closed.json',dict(at=now(),reason=reason,error=error,git=head,pushed=pushed,regressions=codes))
    study.status('completed' if pushed is True else 'push_failed',git=head,reason=reason)
