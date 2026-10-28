"""Idempotent unattended report assembly after the frozen experimental queue exits."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/bc_sequence_safety'


def main():
    if '--watch' in sys.argv:
        started=json.loads((RUN/'protocol.json').read_text())['data']['started']
        while not (RUN/'closed.json').exists():
            if time.time()-started>86400:raise RuntimeError('Postprocessing watcher reached the frozen24h budget')
            state=json.loads((RUN/'status.json').read_text())['data']
            if state['stage']=='failed':raise RuntimeError('Experimental queue failed; no result invented')
            time.sleep(30)
    if not (RUN/'closed.json').exists():raise RuntimeError('Experimental queue has not closed')
    lock=(RUN/'finalize.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    from training.bc_sequence_safety import freeze,read_record,record,status,operation,digest
    if (RUN/'finalized.json').exists():freeze();return
    freeze();os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    from training.bc_sequence_safety_choices import append as choices
    from training.bc_sequence_safety_labels import append as labels
    from training.bc_sequence_safety_training_support import append as support
    choices();labels();support()
    training=read_record(RUN/'training.json') if (RUN/'training.json').exists() else None
    if training:
        from training.bc_sequence_safety_spawn import append as spawn
        spawn()
        if digest(Path(training['model']))!=training['sha256']:raise RuntimeError('Final candidate hash changed')
    outcome=read_record(RUN/'outcome.json');evaluations={}
    for label in ('short32','long16','confirmation32'):
        path=RUN/'evaluation'/label/'summary.json'
        if path.exists():evaluations[label]=read_record(path)
    from training.train_ppo import now
    lines=['# 高分学生安全保留：最终结论','',f'完成：{now()}；停止/完成原因：`{outcome["reason"]}`。','']
    if evaluations:
        r=list(evaluations.values())[-1];a,b,c=r['summary']['bc'],r['summary']['student'],r['summary']['candidate'];p=r['paired']['candidate'];vs=r['candidate_vs_student']
        lines+=[f'1. **是否保留上一轮高分能力？** 最后允许阶段候选/BC共同得分 **{100*(p["common_score_ratio"]-1):+.3f}%**，候选/旧学生 **{100*(vs["common_score_ratio"]-1):+.3f}%**；正向 {p["positive_seeds"]}/{a["games"]}seed。以实际游戏结果判断，不用模仿率替代。']
    else:lines+=['1. **是否保留高分能力？** 未训练/评测合格模型，不能声称保留或改善。']
    if training:
        before,after=training['history'][0]['metrics'],training['history'][-1]['metrics']
        lines+=[f'2. **BC安全行为保留是否提高？** 危险BC动作一致率 {before["safe"]["top1"]:.3%} → {after["safe"]["top1"]:.3%}；严重状态 {before["severe"]["top1"]:.3%} → {after["severe"]["top1"]:.3%}；旧BC稳定回放 {before["bc"]["top1"]:.3%} → {after["bc"]["top1"]:.3%}。普通状态旧学生一致率 {after["normal"]["top1"]:.3%}。这些只是代理指标。真实Actor更新 {training["updates"]} 次。']
    else:lines+=['2. **BC安全行为保留是否提高？** 合格安全数据不足，未训练，不声称改善。']
    lines+=['3. **早死风险是否下降？** '+(f'最后允许组候选/旧学生/BC的<10K早死为 **{c["early"]}/{b["early"]}/{a["early"]}**。同组观察可比较，不宣称有限样本证明风险消失。' if evaluations else '没有新开发证据，不能声称下降。'),
        '4. **全新20K、100K开发表现？**','',
        '|阶段|模型|平均已观测存活|P10|触顶|<10K早死|共同得分相对BC|','|---|---|---:|---:|---:|---:|---:|']
    for label,r in evaluations.items():
        for name in ('bc','student','candidate'):
            m=r['summary'][name];score='基线' if name=='bc' else f'{100*(r["paired"][name]["common_score_ratio"]-1):+.3f}%'
            lines.append(f'|{label}|{name}|{m["mean_pieces"]:.1f}|{m["p10"]:.1f}|{m["capped"]}/{m["games"]}|{m["early"]}|{score}|')
    lines+=['', '未运行阶段：'+(', '.join(s for s in ('short32','long16','confirmation32') if s not in evaluations) or '无')+'。20K/100K上限均右截断，未观察部分不外推。',
        f'5. **是否通过一次性独立确认？** {outcome["confirmation"]}。训练、短开发、长开发、确认种子互不重叠；旧3102803000..3015和预留3102804000..4031未运行。',
        '6. **是否值得登记新候选冠军？** '+('通过冻结确认门槛，可登记研究候选；未替换网页或冻结模型。' if outcome['confirmation'] else '不登记。当前模型未通过全部冻结门槛，本轮停止，不扩样、加epoch或复用开发组调参。'), '',
        '## 解释范围','',
        '预定新快照中出生空间紧张类别为0；另对BC原监督语料的31个出生空间状态做了回顾性动作检查，不加入训练/开发，也不当作生存验证。'
        '原BC监督训练198809帧中严重83帧，新严重快照176/192不在该语料粗分箱支持内；更早20M PPO完整轨迹不可用，不能声称严格全历史OOD。',
        '局部BC恢复成功不等于其快照首动作造成恢复或优于学生。全部接受/拒绝的反事实、模型、优化器、RNG、数据哈希和失败记录保留于 `runs/bc_sequence_safety/`。', '',
        '[完整结果](bc_sequence_safety_results.md) · [冻结设计](bc_sequence_safety_design.md) · [紧凑证据](bc_sequence_safety_evidence.json)']
    codes=read_record(RUN/'regression.json')['returncodes']
    if codes!=[0,0]:raise RuntimeError('Final required regression did not pass')
    lines+=['','Python、Parity、JS回归全部通过。未修改Core、正式计分、动作协议、网页服务或冻结模型；没有新增PPO策略梯度训练。']
    (ROOT/'reports/experiments/bc_sequence_safety_final.md').write_text('\n'.join(lines)+'\n')
    operation('unattended final report assembled with actual corpus, spawn-space, action and label-quality diagnostics; no new games/updates')
    subprocess.run(['git','add','OPERATIONS.md','reports/experiments/bc_sequence_safety_results.md','reports/experiments/bc_sequence_safety_final.md','reports/experiments/bc_sequence_safety_evidence.json'],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:subprocess.run(['git','commit','-m','Finalize safety mechanism evidence and independent development results'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','codex/bc-sequence-safety'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    record(RUN/'finalized.json',dict(at=now(),git=commit,outcome=outcome));status('completed',outcome=outcome,final_commit=commit,postprocessed=True)


if __name__=='__main__':main()
