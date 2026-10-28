"""Descriptive closeout only: no sampling, labels, updates or model selection."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/bc_recovery_sequence'

def main():
    if '--watch' in sys.argv:
        start=json.loads((RUN/'protocol.json').read_text())['data']['started']
        while not (RUN/'closed.json').exists():
            if time.time()-start>86400:raise RuntimeError('Finalizer24h deadline reached')
            time.sleep(30)
    from training.bc_recovery_sequence import read_record,record,read_blob,freeze,operation,status,digest
    from training.train_ppo import now
    lock=(RUN/'finalize.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    freeze()
    if (RUN/'finalized.json').exists():return
    outcome=read_record(RUN/'outcome.json');analysis=read_record(RUN/'analysis.json') if (RUN/'analysis.json').exists() else None
    if read_record(RUN/'regression.json')['returncodes']!=[0,0]:raise RuntimeError('Required regressions did not pass')
    summary=dict(states=0,strong=0,single=0,multi=0,first_same_multi=0,weak_pairs=0,harm_pairs=0,both_censored_pairs=0,divergences=[],strong_episodes=[],coverage={},examples=[],patterns={})
    patterns={};episodes=set()
    for path in sorted((RUN/'episodes').glob('*.pkl')):
        if path.name.endswith('.checkpoint.pkl'):continue
        r=read_blob(path)
        for tier,n in r['exposure'].items():summary['coverage'][tier]=summary['coverage'].get(tier,0)+n
    for path in sorted((RUN/'counterfactuals').glob('*.pkl')):
        r=read_blob(path);c=r['classification'];summary['states']+=1
        summary['harm_pairs']+=len(c['harm']);summary['both_censored_pairs']+=len(c['ambiguous']);summary['weak_pairs']+=3-len(c['strong'])-len(c['harm'])-len(c['ambiguous'])
        if not c['selected']:continue
        summary['strong']+=1;episodes.add(r['seed']);summary['single']+=int('B' in c['strong']);summary['multi']+=int(c['multistep']);summary['first_same_multi']+=int(c['multistep'] and c['first_same'])
        selected=r['branches'][c['selected']];frames=[f for f in selected['frames'] if f['action']!=f['student_action']];positions=[f['window_step'] for f in frames];summary['divergences'].append(len(frames))
        summary['examples'].append(dict(seed=r['seed'],piece=r['piece'],selected=c['selected'],first_same=c['first_same'],multistep=c['multistep'],positions=positions,observed_lengths={n:len(b['rows']) for n,b in r['branches'].items()},deaths={n:b['death'] for n,b in r['branches'].items()},post_handoff_observed=max(0,len(selected['rows'])-{'B':1,'C':16,'D':64}[c['selected']])))
        split='train' if r['seed'] in read_record(RUN/'protocol.json')['train_seeds'] else 'validation'
        for f in frames:
            ids=list(f['legal_ids']);bc=f['legal_features'][ids.index(f['action'])];student=f['legal_features'][ids.index(f['student_action'])]
            sign=lambda a,b:'+' if int(a)>int(b) else '-' if int(a)<int(b) else '='
            key=f"hold{sign(bc[1],student[1])}/clear{sign(bc[5],student[5])}/height{sign(bc[8],student[8])}/holes{sign(bc[6],student[6])}"
            patterns.setdefault(key,dict(train=set(),validation=set()))[split].add(r['seed'])
    summary['strong_episodes']=sorted(episodes);summary['patterns']={k:{p:sorted(v) for p,v in g.items()} for k,g in patterns.items()}
    summary['cross_episode_patterns']={k:v for k,v in summary['patterns'].items() if len(v['train'])>=2 and len(v['validation'])>=1}
    record(RUN/'descriptive_closeout.json',summary)
    training=read_record(RUN/'training.json') if (RUN/'training.json').exists() else None
    evaluations={label:read_record(RUN/'evaluation'/label/'summary.json') for label in ('short32','long16','confirmation32') if (RUN/'evaluation'/label/'summary.json').exists()}
    last=list(evaluations.values())[-1] if evaluations else None
    source=ROOT/'reports/experiments';evidence_path=source/'bc_recovery_sequence_evidence.json';evidence=json.loads(evidence_path.read_text());evidence['descriptive_closeout']=summary
    amendment=RUN/'implementation_amendment.json'
    if amendment.exists():evidence['implementation_amendment']=read_record(amendment)
    evidence_path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    result=source/'bc_recovery_sequence_results.md';text=result.read_text();marker='\n## 完成后的描述性汇总\n';text=text.split(marker)[0]
    result.write_text(text+marker+f'\n强状态{summary["strong"]}，覆盖{len(episodes)}个独立episode；首步即强{summary["single"]}，B不强而C/D强{summary["multi"]}，其中首步相同{summary["first_same_multi"]}。弱证据分支对{summary["weak_pairs"]}，明确损害分支对{summary["harm_pairs"]}，双方触顶未区分分支对{summary["both_censored_pairs"]}。以上不同统计单位分开报告，不把分支对视为独立episode。\n\n实际分歧位置、交还后观察及跨episode动作特征模式见紧凑证据descriptive_closeout；模式重复仅为描述，不构成独立生存因果验证。\n\n技术中断证据保存failures/attempt_01：原子临时文件重命名与磁盘stat竞争；仅修复文件消失处理，保持原protocol SHA、所有种子、采样、标签、窗口、损失和门槛及墙钟起点。\n')
    lines=['# 序列级安全纠错：最终结论','',f'完成{now()}；原因`{outcome["reason"]}`。','',
        f'1. **是否有可验证多步恢复优势？** 预选状态中实际完成{summary["states"]}个预选状态。强恢复{summary["strong"]}个、{len(episodes)}个独立episode；其中B不强而C/D强{summary["multi"]}个。证据要求学生真实死亡及替代至少延长256块，仅适用于预定3000块局部窗口。',
        f'2. **单步模仿是否不足？** 多步强案例{summary["multi"]}个，其中BC与学生首动作相同{summary["first_same_multi"]}个。'+('这些案例为后续连续决策的重要性提供实测局部证据，不证明序列中每步都必要。' if summary['multi'] else '本轮没有足够实测案例证明多步恢复优于首步替换；不能用高度降低替代该证据。'),
        f'3. **高可信标签足够吗？** {bool(analysis and analysis["passed"])}。'+(f'训练/验证按完整episode隔离：`{analysis["groups"]}`；准入`{analysis["checks"]}`。' if analysis else '分析未完成，不声称标签足够。'),
        '4. **是否真实训练新模型？** '+(f'是，{training["updates"]}次Actor更新，SHA256`{training["sha256"]}`。' if training else '否。学习准入未通过或技术/资源停止，冻结高分学生和安全候选均未更新。'),
        '5. **得分能力保留多少？** '+(f'最后允许开发组共同存活得分相对BC {100*(last["paired"]["candidate"]["common_score_ratio"]-1):+.3f}%，相对旧学生 {100*(last["candidate_vs_student"]["common_score_ratio"]-1):+.3f}%。' if last else '没有新候选开发结果，不能声称保分成功或得分提高。原高分学生权重保持冻结。'),
        '6. **生存是否改善？** '+(f'最后允许组候选均值{last["summary"]["candidate"]["mean_pieces"]:.1f}、P10 {last["summary"]["candidate"]["p10"]:.1f}、<10K早死{last["summary"]["candidate"]["early"]}。' if last else '局部恢复证据不等于训练后策略生存改善。没有新模型20K/100K结果，不外推长期安全。'),
        f'7. **是否通过独立确认？** {outcome["confirmation"]}；已运行开发阶段：{list(evaluations)}。未运行组保持未使用，全部上限右截断。',
        '8. **是否继续投入？** '+('通过冻结确认，可评估研究候选；未替换网页模型。' if outcome['confirmation'] else '本轮正常停止，不登记冠军、不追加样本或epoch。若考虑后续研究，必须先解决跨episode强证据覆盖或失败的实战门槛，另行冻结新协议。'),'',
        '## 证据范围与工程收尾','',f'双方均右截断的分支对{summary["both_censored_pairs"]}个，不能解释为安全性相同。仅棋盘代理改善等弱证据{summary["weak_pairs"]}个，未混入强训练标签；明确接管损害{summary["harm_pairs"]}个。一个episode可贡献多个相关状态，不能据状态数量扩大独立样本量。',
        '原子临时文件stat竞争导致的一次技术中断已归档并修复；原协议、数据身份和24小时起点保持不变，原失败报告及traceback永久保留。全部历史模型/报告/数据/Core哈希核对通过。',
        'Python、Parity与JS回归全部通过。没有修改正式规则、网页服务、冻结模型；未合并main或发布版本。','',
        '[冻结设计](bc_recovery_sequence_design.md) · [完整结果](bc_recovery_sequence_results.md) · [逐状态证据](bc_recovery_sequence_evidence.json)']
    (source/'bc_recovery_sequence_final.md').write_text('\n'.join(lines)+'\n')
    operation(f'final descriptive closeout: strong={summary["strong"]}, episodes={len(episodes)}, multistep={summary["multi"]}, trained={training is not None}, reason={outcome["reason"]}; no new games/labels/updates')
    subprocess.run(['git','add','OPERATIONS.md',*map(str,source.glob('bc_recovery_sequence_*'))],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Finalize cross-episode recovery evidence and eight research answers'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','codex/bc-recovery-sequence'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();record(RUN/'finalized.json',dict(at=now(),git=commit,reason=outcome['reason']));status('completed',reason=outcome['reason'],final_commit=commit)

if __name__=='__main__':main()
