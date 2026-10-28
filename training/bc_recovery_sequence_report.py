"""Bounded-study closeout, including negative data and resource outcomes."""
import subprocess
import sys
from pathlib import Path
from training.bc_recovery_sequence import RUN,ROOT,read_record,record,read_blob,operation,status,freeze,digest
from training.train_ppo import now

def finish(reason,error=None):
    if (RUN/'closed.json').exists():return
    status('closing',reason=reason);freeze()
    data=read_record(RUN/'analysis.json') if (RUN/'analysis.json').exists() else None
    training=read_record(RUN/'training.json') if (RUN/'training.json').exists() else None
    evaluations={label:read_record(RUN/'evaluation'/label/'summary.json') for label in ('short32','long16','confirmation32') if (RUN/'evaluation'/label/'summary.json').exists()}
    cases=[];hist={};deaths={};patterns={}
    for path in sorted((RUN/'counterfactuals').glob('*.pkl')):
        r=read_blob(path);c=r['classification'];b=r['branches'];key=r['risk']['tier'];hist[key]=hist.get(key,0)+1
        for name,v in b.items():deaths[name]=deaths.get(name,0)+int(v['death'])
        row=dict(seed=r['seed'],piece=r['piece'],risk=r['risk'],classification=c,branches={name:dict(pieces=len(v['rows']),death=v['death'],right_censored=v['right_censored'],score_gain=v['rows'][-1]['score']-r['start_score'],divergences=[f['window_step'] for f in v['frames'] if f['action']!=f['student_action']],handoff_reached=v['handoff_reached'],hold_path_differences=sum(x['placed']!=y['placed'] or x['held']!=y['held'] or x['preview']!=y['preview'] for x,y in zip(b['A']['rows'],v['rows']))) for name,v in b.items()})
        cases.append(row)
        if c['selected']:
            for f in b[c['selected']]['frames']:
                if f['action']!=f['student_action']:
                    k=f"{c['selected']}:{f['risk']['tier']}";patterns.setdefault(k,set()).add(r['seed'])
    evidence=dict(reason=reason,error=error,analysis=data,risk_hist=hist,observed_deaths=deaths,cases=cases,patterns={k:sorted(v) for k,v in patterns.items()},training=training,evaluations=evaluations,protocol_sha=digest(RUN/'protocol.json'))
    folder=ROOT/'reports/experiments';record(RUN/'outcome.json',dict(reason=reason,error=error,confirmation=reason=='confirmed'))
    lines=['# 序列级安全纠错：实测结果','',f'完成 {now()}；停止/完成原因 `{reason}`。',f'错误（若有）：`{error}`。','',f'完成反事实状态 {len(cases)}；风险层 {hist}；四分支真实死亡计数 {deaths}（状态聚集，不是独立死亡率）。',
        '所有3000块上限为右截断；A与替代同时触顶不证明安全性相同。强标签只来自A实际死亡及替代至少多活256块；其他棋盘改善没有升级为强标签。','',f'学习准入数据：`{data["groups"] if data else None}`。',f'冻结门槛：`{data["checks"] if data else None}`。','',
        '|seed|起始块|层级|A/B/C/D观察块数|强恢复|明确损害|首动作相同|需多步证据|','|---:|---:|---|---|---|---|---|---|']
    for r in cases:
        c=r['classification'];lines.append(f'|{r["seed"]}|{r["piece"]}|{r["risk"]["tier"]}|{[v["pieces"] for v in r["branches"].values()]}|{c["strong"]}|{c["harm"]}|{c["first_same"]}|{c["multistep"]}|')
    for label,r in evaluations.items():
        lines += ['',f'## {label}', '',f'固定评测摘要：`{r}`。']
    lines+=['','## 解释限制','', '每个状态的多分支共享起点Core/RNG；Hold导致的真实放置、队列差异逐步保存。BC与学生动作只在同一分支同一公开状态上比较。恢复片段完整保存，分歧标签与普通保留帧按完整episode隔离。', '恢复优势集中位置、首动作相同后的后续关键分歧及跨episode模式见紧凑证据；分歧数量不等于每个动作都具有单独因果作用。若没有多步强恢复，不能把BC能压低棋盘当作多步安全收益。', '未使用旧死亡seed、旧开发或预留3102804000..4031。没有运行的开发/确认组保持未使用；没有根据结果追加样本、调参或扩训练预算。']
    (folder/'bc_recovery_sequence_results.md').write_text('\n'.join(lines)+'\n')
    trained=training is not None
    last=list(evaluations.values())[-1] if evaluations else None
    final=['# 序列级安全纠错：最终结论','',f'结束 {now()}；原因 `{reason}`。','',
        f'1. **可验证多步恢复优势？** 强证据和跨episode统计：`{data["groups"] if data else "未完成；不作结论"}`。',
        '2. **单步为何不足，有无实证？** 只把B不强而C/D强的案例视为有限窗口多步证据；首步相同但后续产生关键分歧另计。没有这样的案例时不宣称已证明单步不足。',
        f'3. **高可信标签足够？** {data["passed"] if data else False}；冻结准入检查 `{data["checks"] if data else None}`。不足即停止。',
        f'4. **真实训练新模型？** {trained}'+(f'；更新{training["updates"]}次。' if trained else '；冻结历史模型未更新。'),
        '5. **得分保留多少？** '+(str(last['paired']['candidate']) if last else '未训练/未完成新开发评测，不外推保分效果。'),
        '6. **生存指标改善？** '+(str(last['summary']['candidate']) if last else '局部反事实仅适用于所选状态，不是新模型生存改善或长期安全验证。'),
        f'7. **独立确认通过？** {reason=="confirmed"}；'+('已按冻结协议完成。' if reason=='confirmed' else '没有通过确认，不登记冠军。'),
        '8. **值得继续投入？** '+('可依据通过的独立证据评估候选。' if reason=='confirmed' else '本轮不追加训练或采样；先解决证据覆盖/学习准入或实战门槛失败原因，再考虑单独预注册研究。'),'',
        '[冻结设计](bc_recovery_sequence_design.md) · [完整结果](bc_recovery_sequence_results.md) · [逐状态证据](bc_recovery_sequence_evidence.json)','', '全部原始失败、拒绝、反事实、RNG与哈希永久保留于runs/bc_recovery_sequence/。']
    (folder/'bc_recovery_sequence_final.md').write_text('\n'.join(final)+'\n')
    import json
    (folder/'bc_recovery_sequence_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    commands=[[sys.executable,'-m','pytest','-q','tests/test_bc_recovery_sequence.py','tests/test_tetris_env.py','tests/parity'],['node','--test',*map(str,sorted((ROOT/'tests').glob('*.test.cjs')))]]
    codes=[]
    for i,cmd in enumerate(commands):
        with (RUN/f'regression_{i}.log').open('w') as f:codes.append(subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode)
    record(RUN/'regression.json',dict(commands=commands,returncodes=codes,at=now()))
    operation(f'closed reason={reason}, regression={codes}; raw evidence preserved; frozen identities verified')
    with (ROOT/'CHANGELOG.md').open('a') as f:f.write(f'\n### BC recovery sequence research ({now()[:10]})\n\n- Pre-registered four-branch3000-piece recovery study; result={reason}; strong labels separated from weak board proxies. No champion without confirmation.\n')
    with (folder/'bc_recovery_sequence_final.md').open('a') as f:f.write(f'\nPython/Parity与JS回归退出码：{codes}。\n')
    if any(codes):raise RuntimeError('Required regressions failed; logs retained')
    subprocess.run(['git','add','OPERATIONS.md','CHANGELOG.md',*map(str,folder.glob('bc_recovery_sequence_*'))],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Report preregistered multi-step recovery evidence and stop decision'],cwd=ROOT,check=True)
    subprocess.run(['git','push','-u','origin','codex/bc-recovery-sequence'],cwd=ROOT,check=True)
    record(RUN/'closed.json',dict(at=now(),reason=reason,git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()));status('completed',reason=reason)
