"""Close out every allowed stage, including insufficient-data outcomes."""
import json
import subprocess
import sys
from pathlib import Path

from training.bc_sequence_safety import RUN,ROOT,record,read_record,operation,status,freeze,digest
from training.train_ppo import now


def report():
    outcome=read_record(RUN/'outcome.json');data=read_record(RUN/'dataset.json');mechanism=read_record(RUN/'mechanism.json')
    training=read_record(RUN/'training.json') if (RUN/'training.json').exists() else None
    distribution=read_record(RUN/'new_distribution.json');evaluations={};details={}
    lines=['# 高分学生安全保留：实测结果','',f'生成：{now()}；结束：`{outcome["reason"]}`。','',
        '## 冻结模型的机制检查','',f'旧BC验证回放中，原学生动作一致率：{mechanism["student_bc_replay"]["top1"]:.3%}。',
        '旧数据按来源统计落子前公开危险条件，不把BC的所有动作视为最优。高危mask分布比较只使用正式公开状态。','',
        '|旧数据来源|总帧|普通/中等/严重/其他|比较帧|BC/学生动作一致率|KL(BC||学生)|','|---|---:|---|---:|---:|---:|']
    for source in ('bc','ppo','transfer'):
        r=mechanism[source];lines.append(f'|{source}|{r["frames"]}|{r["risk_counts"]}|{r["compared"]}|{r["bc_student_agreement"]:.3%}|{r["bc_to_student_kl"]:.3f}|')
    lines+=['','高危样本不足时该来源比较会明确使用普通帧：'+str({s:mechanism[s]['high_risk_comparison'] for s in ('bc','ppo','transfer')})+'。','',
        '## 新学生轨迹与固定反事实','',f'逐块公开风险暴露：`{data["exposure"]}`；接受 {data["accepted"]}，拒绝 {data["rejected"]}。所有反事实（含拒绝）保留。',
        f'按episode分组计数：`{data["counts"]}`；严重合格数量：`{data["severe"]}`。',f'数据门槛：`{data["checks"]}`。',
        '新学生危险状态相对旧BC粗布局分箱支持（不代表完整状态的OOD判定）：`'+str(distribution)+'`。',
        'BC的256块恢复仅为局部有限窗口证据，右截断不等于长期安全。模型输入没有未来、RNG或真实死亡时间。','',
        '## 安全蒸馏','']
    if training:
        lines+=[f'真实梯度更新 {training["updates"]} 次；单模型 `{training["model"]}`；SHA256 `{training["sha256"]}`。',
            '|epoch|普通学生一致率|危险BC一致率|严重BC一致率|旧BC一致率|旧PPO示范动作率|loss|梯度范数|','|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in training['history']:
            m=r['metrics'];severe=m['severe']['top1'] if m['severe'] else 0
            lines.append(f'|{r["epoch"]}|{m["normal"]["top1"]:.3%}|{m["safe"]["top1"]:.3%}|{severe:.3%}|{m["bc"]["top1"]:.3%}|{m["ppo"]["top1"]:.3%}|{r.get("loss","—")}|{r.get("gradient","—")}|')
    else:lines+=['没有足够合格安全数据；未开始训练，不声称改善。']
    for label in ('short32','long16','confirmation32'):
        path=RUN/'evaluation'/label/'summary.json'
        if not path.exists():continue
        r=read_record(path);g=read_record(RUN/f'{label}_gate.json');evaluations[label]=dict(result=r,gate=g)
        lines+=['',f'## {label}','','|模型|平均已观测生存|P10|触顶|<10K早死|平均正式分|四消比例|推理ms/块|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for n,s in r['summary'].items():
            lines.append(f'|{n}|{s["mean_pieces"]:.2f}|{s["p10"]:.2f}|{s["capped"]}/{s["games"]}|{s["early"]}|{s["mean_score"]:.1f}|{s["four_fraction"]:.3%}|{s["mean_inference_ms"]:.3f}|')
        lines+=['','|模型|高位比例|新洞/100块|最长高位串均值|单/双/三/四消次数|死亡末高度/洞|','|---|---:|---:|---:|---|---|']
        details[label]={}
        for n,s in r['summary'].items():
            counts=[0]*5;dead=[]
            for path in sorted((RUN/'evaluation'/label/n).glob('seed_*.json')):
                row=read_record(path);counts=[a+b for a,b in zip(counts,row['clears_0_to_4'])]
                if row['game_over']:dead.append(dict(seed=row['seed'],piece=row['pieces'],height=row['final_height'],holes=row['final_holes'],high_streak=row['high_streak']))
            details[label][n]=dict(clears_0_to_4=counts,deaths=dead)
            terminal='—' if not dead else f'{sum(d["height"] for d in dead)/len(dead):.2f}/{sum(d["holes"] for d in dead)/len(dead):.2f}'
            lines.append(f'|{n}|{s["high_fraction"]:.4%}|{s["new_holes_per_100"]:.3f}|{s["mean_max_high_streak"]:.2f}|{counts[1:]}|{terminal}|')
        p=r['paired']['candidate'];vs=r['candidate_vs_student']
        lines+=['',f'候选/BC共同得分比 {p["common_score_ratio"]:.5f}，正向 {p["positive_seeds"]}，95%区间 {p["ratio_ci95"]}；候选/旧学生 {vs["common_score_ratio"]:.5f}。',f'冻结门槛：`{g}`。','',
            '|seed|共同块数|候选/BC分数比|候选−BC存活|','|---:|---:|---:|---:|']
        for pair in p['pairs']:lines.append(f'|{pair["seed"]}|{pair["common_pieces"]}|{pair["ratio"]:.5f}|{pair["survival_delta"]}|')
    lines+=['','所有达到上限的局均右截断。动作一致率不替代实战生存；没有运行的阶段保持未使用。旧死亡seed没有重新采样/训练/开发。']
    folder=ROOT/'reports/experiments';(folder/'bc_sequence_safety_results.md').write_text('\n'.join(lines)+'\n')
    last=list(evaluations.values())[-1]['result'] if evaluations else None
    score='未运行新模型评测，不能判断保留得分能力。' if last is None else f'最后允许阶段候选/BC共同分比 {last["paired"]["candidate"]["common_score_ratio"]:.5f}；候选/旧学生 {last["candidate_vs_student"]["common_score_ratio"]:.5f}。'
    safety='未训练，未声称改善。' if training is None else f'危险BC动作一致率 {training["history"][0]["metrics"]["safe"]["top1"]:.3%} → {training["history"][-1]["metrics"]["safe"]["top1"]:.3%}；这是代理指标。'
    deaths='没有全新开发证据，不能声称早死下降。' if last is None else f'最后允许阶段早死：候选 {last["summary"]["candidate"]["early"]}，旧学生 {last["summary"]["student"]["early"]}，BC {last["summary"]["bc"]["early"]}；有限样本不证明总体风险消除。'
    final=['# 高分学生安全保留：最终结论','',f'完成：{now()}；结果 `{outcome["reason"]}`。','',
        '1. **是否保留高分能力？** '+score,'2. **BC安全行为保留是否提高？** '+safety,
        '3. **早死风险是否下降？** '+deaths,
        '4. **全新20K/100K开发表现？** 已运行阶段：'+(', '.join(evaluations) or '无：数据门槛未过')+'；完整生存/得分/风险和逐seed表见结果。100K未运行时不外推。',
        f'5. **一次性独立确认？** {outcome["confirmation"]}。旧预留3102804000..4031没有使用。',
        '6. **是否登记新候选冠军？** '+('通过冻结确认门槛，归档为研究候选；没有发布或替换网页模型。' if outcome['confirmation'] else '不登记；按冻结停止条件结束，不增加预算或重复开发组调参。'),'',
        '本轮只尝试风险加权BC蒸馏与普通状态学生分布保留这一种方法；冻结模型、旧数据、旧报告和Core保持不变。',
        '证据：[完整结果](bc_sequence_safety_results.md)、[设计](bc_sequence_safety_design.md)、`runs/bc_sequence_safety/`。']
    (folder/'bc_sequence_safety_final.md').write_text('\n'.join(final)+'\n')
    evidence=dict(protocol=read_record(RUN/'protocol.json'),mechanism=mechanism,new_distribution=distribution,dataset=data,training=training,outcome=outcome,evaluation=evaluations,evaluation_details=details)
    (folder/'bc_sequence_safety_evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')


def finish():
    if (RUN/'closed.json').exists():return
    freeze();report()
    commands=[[sys.executable,'-m','pytest','-q','tests/test_bc_sequence_safety.py','tests/test_bc_sequence_transfer.py','tests/test_bc_sequence_supervised.py','tests/test_tetris_env.py','tests/parity'],
        ['node','--test',*map(str,sorted((ROOT/'tests').glob('*.test.cjs')))]]
    codes=[]
    for i,cmd in enumerate(commands):
        with (RUN/f'regression_{i}.log').open('w') as f:codes.append(subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode)
    record(RUN/'regression.json',dict(at=now(),commands=commands,returncodes=codes))
    operation(f'final outcome={read_record(RUN/"outcome.json")}; regressions={codes}; frozen artifacts unchanged')
    if any(codes):raise RuntimeError('Required regression failed')
    with (ROOT/'reports/experiments/bc_sequence_safety_final.md').open('a') as f:f.write('\nPython、Parity与JS回归全部通过；日志 `runs/bc_sequence_safety/regression_*.log`。\n')
    marker='### BC sequence safety research (2026-10-28)'
    if marker not in (ROOT/'CHANGELOG.md').read_text():
        with (ROOT/'CHANGELOG.md').open('a') as f:f.write('\n'+marker+'\n\n- Added fixed-risk sampling, matched BC/student recovery counterfactuals and conditional safety distillation from the high-score student.\n- Preserved prior artifacts; reported all allowed stages and the pre-registered stop/confirmation decision.\n')
    subprocess.run(['git','add','OPERATIONS.md','CHANGELOG.md',*map(str,(ROOT/'reports/experiments').glob('bc_sequence_safety_*.md')),'reports/experiments/bc_sequence_safety_evidence.json'],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:subprocess.run(['git','commit','-m','Report bounded safety retention study and frozen evaluation outcome'],cwd=ROOT,check=True)
    subprocess.run(['git','push','-u','origin','codex/bc-sequence-safety'],cwd=ROOT,check=True)
    record(RUN/'closed.json',dict(at=now(),outcome=read_record(RUN/'outcome.json'),regression=codes));status('completed',outcome=read_record(RUN/'outcome.json'))
