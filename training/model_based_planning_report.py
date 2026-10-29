"""Close every allowed planning stage, including resource and engineering stops."""
import json
import subprocess
import sys
import urllib.request
from training.model_based_planning import ROOT,RUN,read_record,record,freeze,operation,status,digest,disk_bytes
from training.train_ppo import now

def health():
    result={}
    for port,suffix in ((8080,'/'),(8081,'/api/health')):
        try:
            with urllib.request.urlopen(f'http://100.91.227.47:{port}{suffix}',timeout=5) as r:result[str(port)]=r.status
        except Exception as exc:result[str(port)]=repr(exc)
    return result

def finish(reason,error=None):
    if (RUN/'closed.json').exists():return
    status('closing',reason=reason);protocol=freeze()
    functional=read_record(RUN/'functional.json') if (RUN/'functional.json').exists() else None
    stages={label:dict(result=read_record(RUN/f'{label}_summary.json'),gate=read_record(RUN/f'{label}_gate.json')) for label in ('short16','long16','confirmation32') if (RUN/f'{label}_gate.json').exists()}
    traces=[]
    for seed in protocol['functional_seeds']:
        p=RUN/'functional/planner'/f'seed_{seed}.json'
        if p.exists():traces+=read_record(p)['traces']
    differences=[t for t in traces if t['selected']['action']!=t['b12_action']]
    third=[t for t in traces if t['selected']['action'] not in (t['student_action'],t['bc_action'])]
    examples=differences[:8]
    lines=['# 公开信息有限步规划：实测结果','',f'完成{now()}；结果`{reason}`；错误`{error}`。','',
        '**控制器身份：冻结高分学生、BC建议与规则搜索组成的混合控制器。零训练、零梯度更新，没有新单网络checkpoint。**','',
        '## 功能和信息隔离','',f'固定功能组：`{functional}`。',
        f'固定状态比较：规划/B12不同{len(differences)}/{len(traces)}；规划选择学生与BC之外动作{len(third)}/{len(traces)}。根候选、路径、合法落点数、边界、代价和理由逐步保存。','',
        '|功能步|学生Action|BC Action|B12 Action|规划Action|理由|','|---:|---:|---:|---:|---:|---|']
    for t in examples:lines.append(f'|{t["piece"]}|{t["student_action"]}|{t["bc_action"]}|{t["b12_action"]}|{t["selected"]["action"]}|{t["reason"]}|')
    for label,stage in stages.items():
        r=stage['result'];lines+=['',f'## {label}','','|模型|均值已观察生存|P10|触顶|<10K早死|平均正式总分|四消占比|平均/P95/P99 ms|','|---|---:|---:|---:|---:|---:|---:|---|']
        for n,s in r['summary'].items():lines.append(f'|{n}|{s["mean_pieces"]:.1f}|{s["p10"]:.1f}|{s["capped"]}/{s["games"]}|{s["early"]}|{s["mean_score"]:.1f}|{s["four_fraction"]:.3%}|{s["mean_latency_ms"]:.3f}/{s["p95_ms"]:.3f}/{s["p99_ms"]:.3f}|')
        p=r['paired']['planner'];b=r['vs_student'];s=r['summary']['planner']
        lines+=['',f'规划/BC共同存活正式得分{100*(p["common_score_ratio"]-1):+.3f}%，正向{p["positive_seeds"]}，seed级Bootstrap95%比率区间{p["ratio_ci95"]}。规划/学生共同得分{100*(b["common_score_ratio"]-1):+.3f}%。',f'覆盖学生动作{s["overrides"]}次，第三类动作{s["third_actions"]}次，规避所选学生rollout失败{s["rollout_failure_avoided"]}次。这是有限搜索路径的描述性差异，不是全搜索不可行性证明。',f'冻结门槛：`{stage["gate"]}`。','',
            '|模型|高度>=17比例|新洞/100块|最长高位串均值|','|---|---:|---:|---:|']
        for n,s in r['summary'].items():lines.append(f'|{n}|{s["high_fraction"]:.5%}|{s["new_holes_per_100"]:.3f}|{s["mean_max_high_streak"]:.2f}|')
        lines+=['','|seed|共同块数|BC正式分|规划正式分|分比|存活差|','|---:|---:|---:|---:|---:|---:|']
        for pair in p['pairs']:lines.append(f'|{pair["seed"]}|{pair["common_pieces"]}|{pair["baseline_score"]}|{pair["candidate_score"]}|{pair["ratio"]:.5f}|{pair["survival_delta"]}|')
    lines+=['','## 解释限制','',
        '仅使用实际公开Next3。未知Current或空Hold所需Next不可知时停止该分支，不生成隐藏Bag；真实已知出生阻塞/Lock Out与未知边界分开。没有无限生存保证；宽度1搜索未遍历其他后续落点，失败rollout不等价于该根所有未来动作不可行。',
        '根策略建议不能当作安全标签。BC动作可以被拒绝，规划也能选择第三类动作；没有高度阈值专家切换状态。Hold导致的已知队列消耗、正式分与消行已与真实Core前缀校验。',
        '所有达到20K/100K的局是右截断，未观察生存不外推；Bootstrap以seed为重采样单位，不能当成无限生存概率。没有使用此前开发/确认预留组。所有失败、已提交任务状态、原始得分/延迟序列、身份与哈希保留于runs/model_based_planning/。']
    folder=ROOT/'reports/experiments';(folder/'model_based_planning_results.md').write_text('\n'.join(lines)+'\n')
    last=list(stages.values())[-1]['result'] if stages else None
    metrics=last['summary']['planner'] if last else (functional['summary'] if functional else None)
    final=['# 公开信息有限步规划：最终结论','',f'结束{now()}；原因`{reason}`。','',
        '1. **与B12的本质区别？** 本轮为冻结学生+BC建议+公开规则搜索的混合控制器；基于最多4根、4次落子的公开rollout代价决定是否覆盖学生，而非高度阈值切BC。'+f'固定功能状态中与B12动作不同{len(differences)}/{len(traces)}，第三类动作{len(third)}次。',
        '2. **是否真正使用合法Next3？** 是。模拟器只装载公开3个Next及Hold，不创建RNG或Bag；耗尽即未知边界停止。'+(f'功能期验证{functional["simulation_checks"]}个正式模拟动作、{functional["hidden_checks"]}组隐藏状态扰动不变性；实际Core不被规划修改。' if functional else '功能阶段未完成，不声称运行验证全部通过。'),
        '3. **是否有新可行性或安全优势？** '+(f'最后允许阶段规避所选学生rollout失败{last["summary"]["planner"]["rollout_failure_avoided"]}次；必须结合下面实战生存结果，不把局部rollout成功当成长期保证。' if last else '没有新开发实战证据，不把短期模拟可行性当作生存改善。'),
        '4. **对学生早死有何影响？** '+(f'同组规划/学生<10K为{last["summary"]["planner"]["early"]}/{last["summary"]["student"]["early"]}；有限样本不证明总体风险消失。' if last else '未运行新开发组，不能判断早死变化。'),
        '5. **相对BC得分提升多少？** '+(f'共同存活正式得分{100*(last["paired"]["planner"]["common_score_ratio"]-1):+.3f}%，正向{last["paired"]["planner"]["positive_seeds"]}/{last["summary"]["bc"]["games"]}seed。' if last else '没有新开发得分证据。'),
        f'6. **是否通过20K/100K/确认？** '+str({label:stage['gate']['passed'] for label,stage in stages.items()})+f'；未运行阶段{[label for label in ("short16","long16","confirmation32") if label not in stages]}；确认{reason=="confirmed"}。',
        '7. **CPU耗时？** '+(f'最后允许阶段（无开发时为功能组）平均{metrics["mean_latency_ms"]:.3f}ms，P95 {metrics["p95_ms"]:.3f}ms，P99 {metrics["p99_ms"]:.3f}ms。固定目标P95<=50ms，不按得分放宽。' if metrics else '功能运行未完成，只有工程烟测，不能替代正式分位数。'),
        '8. **值得继续发展或蒸馏？** '+('通过冻结确认，可评估混合控制器候选；蒸馏仍是未验证的后续工作。' if reason=='confirmed' else '本轮按协议停止，不登记冠军、不追加相似方案。当前证据不足以直接投入蒸馏或扩大预算，应先解释失败门槛。'),'',
        '**没有训练新的单一神经网络。** 原模型、Core、计分、动作空间和历史报告保持冻结；没有合并main或发布版本。','',
        '[设计](model_based_planning_design.md) · [结果](model_based_planning_results.md) · [证据](model_based_planning_evidence.json)']
    (folder/'model_based_planning_final.md').write_text('\n'.join(final)+'\n')
    services=health();evidence=dict(reason=reason,error=error,functional=functional,stages=stages,functional_examples=examples,third_action_examples=third[:8],protocol=protocol,services=services,retained_bytes=disk_bytes())
    (folder/'model_based_planning_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    commands=[[sys.executable,'-m','pytest','-q','tests/test_public_lookahead.py','tests/test_model_based_planning.py','tests/test_tetris_env.py','tests/parity'],['node','--test',*map(str,sorted((ROOT/'tests').glob('*.test.cjs')))]]
    codes=[]
    for i,cmd in enumerate(commands):
        with (RUN/f'regression_{i}.log').open('w') as f:codes.append(subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode)
    record(RUN/'regression.json',dict(at=now(),commands=commands,returncodes=codes));record(RUN/'outcome.json',dict(reason=reason,error=error,confirmation=reason=='confirmed',services=services))
    operation(f'final reason={reason}; regressions={codes}; services={services}; frozen identities unchanged, no gradient updates')
    with (ROOT/'CHANGELOG.md').open('a') as f:f.write(f'\n### Public Next3 model-based planning ({now()[:10]})\n\n- Added a bounded public-only frozen-policy hybrid controller, exact official transitions, hidden-state isolation tests and a preregistered evaluation queue. Result:{reason}; no policy training or new single-network claim.\n')
    with (folder/'model_based_planning_final.md').open('a') as f:f.write(f'\nPython/Parity、JS退出码{codes}；8080/8081健康{services}。\n')
    if any(codes):raise RuntimeError('Required regression failed')
    subprocess.run(['git','add','OPERATIONS.md','CHANGELOG.md',*map(str,folder.glob('model_based_planning_*'))],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Report bounded public Next3 planning results and stop decision'],cwd=ROOT,check=True)
    subprocess.run(['git','push','-u','origin','codex/model-based-planning'],cwd=ROOT,check=True)
    record(RUN/'closed.json',dict(at=now(),reason=reason,git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()));status('completed',reason=reason)
