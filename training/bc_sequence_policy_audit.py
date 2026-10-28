"""One pre-frozen 16-seed deployment audit of the stopped epoch-1 policy."""
import os
import subprocess
import sys
from pathlib import Path

import training.bc_sequence_supervised as supervised
from training.bc_sequence_transfer import (ROOT, RUN, read_record, record, digest,
    operation, status, guard)
from training.train_ppo import now

AUDIT = RUN/'policy_audit'
LABEL = 'stopped_epoch01_audit_20k'


def freeze():
    supervised.freeze()
    training = read_record(supervised.WORK/'training_result.json')
    outcome = read_record(supervised.WORK/'outcome.json')
    if outcome['reason'] != 'stopped_supervised_learning_signal' or training['history'][-1]['epoch'] != 1:
        raise RuntimeError('Audit requires the original stopped epoch1; no model search permitted')
    model = Path(training['last_model'])
    if read_record(model.parent/'manifest.json')['sha256']['model.zip'] != digest(model):
        raise RuntimeError('Archived model no longer matches its training manifest')
    names = ('training/bc_sequence_policy_audit.py',
             'reports/experiments/bc_sequence_transfer_audit_design.md')
    sources = {name:digest(ROOT/name) for name in names}
    protocol_path = AUDIT/'protocol.json'
    if protocol_path.exists():
        protocol = read_record(protocol_path)
        if protocol['sources'] != sources or protocol['model_sha256'] != digest(model):
            raise RuntimeError('Audit identity changed')
        return protocol
    if subprocess.run(['git','diff','--quiet','--',*names],cwd=ROOT).returncode:
        raise RuntimeError('Commit audit source/design before consuming development seeds')
    if list((supervised.WORK/'evaluation').glob('**/seed_*.json')):
        raise RuntimeError('Development seeds already evaluated before audit freeze')
    AUDIT.mkdir(parents=True,exist_ok=True)
    protocol = dict(at=now(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources=sources,supervised_protocol_sha256=digest(supervised.WORK/'protocol.json'),
        checkpoint=str(model),model_sha256=digest(model),seeds=read_record(RUN/'protocol.json')['development_seeds'],
        cap=20000,mode='one diagnostic; no subsequent training/100k/confirmation',
        original_stop=outcome['reason'])
    record(protocol_path,protocol)
    operation('resumed once for preregistered epoch1 diagnostic, 16 fresh development seeds, no additional training')
    return protocol


def write_report(result,gate,protocol):
    rows = result['summary']
    p = result['paired_to_bc']['candidate']
    lines = ['\n## 归档epoch1：补充16seed×20K实战诊断\n',
        '此诊断在开发结果产生前单独冻结。监督保留门槛已失败，因此不启动长局、确认或再训练；它用于量化实际行为，不能把该模型视为合格候选。', '',
        f'模型SHA256：`{protocol["model_sha256"]}`；诊断协议提交：`{protocol["git_commit"]}`。', '',
        '|模型|平均已观测存活|P10|触顶|<10K早死|平均正式分|四消/全部消行事件|推理ms/块|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        lines.append(f'|{name}|{r["mean_pieces"]:.1f}|{r["p10"]:.1f}|{r["capped"]}/{r["games"]}|{r["early"]}|'
            f'{r["mean_score"]:.0f}|{r["four_fraction"]:.3%}|{r["mean_inference_ms"]:.3f}|')
    lines += ['',f'共同存活得分比 **{p["common_score_ratio"]:.5f}**，即 **{100*(p["common_score_ratio"]-1):+.3f}%**；'
        f'正向seed **{p["positive_seeds"]}/16**；95%seed bootstrap区间 **{p["ratio_ci95"]}**。',
        f'原20K门槛的描述性结果：`{gate}`。', '',
        '|模型|高度≥17比例|新增洞/100块|平均最长高位连续块数|', '|---|---:|---:|---:|']
    for name,r in rows.items():
        lines.append(f'|{name}|{r["high_fraction"]:.3%}|{r["new_holes_per_100"]:.3f}|{r["mean_max_high_streak"]:.1f}|')
    lines += ['', '|Seed|共同块数|候选/BC共同分比|候选−BC存活|', '|---:|---:|---:|---:|']
    for r in p['pairs']:
        lines.append(f'|{r["seed"]}|{r["common_pieces"]}|{r["ratio"]:.5f}|{r["survival_delta"]}|')
    with (ROOT/'reports/experiments/bc_sequence_transfer_results.md').open('a') as stream:
        stream.write('\n'.join(lines)+'\n')
    a,b = rows['bc'],rows['candidate']
    replacements = {
        '4. **': f'4. **是否学到新四消？** 归档单模型在16seed实战中的四消占消行事件比例为 {b["four_fraction"]:.3%}，原BC为 {a["four_fraction"]:.3%}。监督示范动作学习确实改善；实际四消变化不等于整体策略已合格。',
        '5. **': f'5. **效率提升多少？** 归档epoch1与BC在共同存活窗口的正式得分变化为 **{100*(p["common_score_ratio"]-1):+.3f}%**，正向 {p["positive_seeds"]}/16seed，95%bootstrap区间 {p["ratio_ci95"]}。冻结PPO接管的3.90%收益与这个单模型结果分别报告。',
        '6. **': f'6. **生存损失多少？** 平均已观测存活候选 {b["mean_pieces"]:.1f}、BC {a["mean_pieces"]:.1f}（变化 {100*(b["mean_pieces"]/a["mean_pieces"]-1):+.3f}%）；P10 {b["p10"]:.1f}/{a["p10"]:.1f}；触顶 {b["capped"]}/16 对 {a["capped"]}/16；<10K早死 {b["early"]} 对 {a["early"]}。20K上限右截断，不能外推100K。',
        '8. **': f'8. **失败根因？** 这一个混合监督配方学到了更多PPO动作，却将BC原状态动作一致率降至78.0%，未保住预注册90%门槛。一次归档模型实战诊断的20K门槛结果为 `{gate}`。源/目标示范验证损失下降与实际得分、生存是不同证据；共享Actor的这次监督折中没有同时满足全部要求，不能唯一归因于容量或某个超参数。',
    }
    path = ROOT/'reports/experiments/bc_sequence_transfer_final.md'
    final = path.read_text().splitlines()
    for i,line in enumerate(final):
        for prefix,replacement in replacements.items():
            if line.startswith(prefix):
                final[i] = replacement
    final += ['', '补充实战诊断已使用原预留16seed开发组；没有重复调参，没有启动100K或独立确认。'
        '这是原epoch3-only评测安排之外、在开发结果产生前冻结的一次补充诊断；原训练停止条件仍生效。']
    path.write_text('\n'.join(final)+'\n')
    import json
    evidence_path = ROOT/'reports/experiments/bc_sequence_transfer_evidence.json'
    evidence = json.loads(evidence_path.read_text())
    evidence['policy_audit'] = dict(protocol=protocol,result=result,gate=gate,
        row_sha256={str(path.relative_to(RUN)):digest(path) for path in
            sorted((supervised.WORK/'evaluation'/LABEL).glob('*/*.json'))})
    evidence_path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')


def main():
    protocol = freeze()
    if (AUDIT/'completed.json').exists():
        return
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,affinity[-2:])
    guard()
    result = supervised.evaluate(Path(protocol['checkpoint']),protocol['seeds'],20000,LABEL,digest(AUDIT/'protocol.json'))
    gate = supervised.development_gate(result)
    record(AUDIT/'gate.json',gate)
    # Rebuild canonical base reports first so interrupted reporting cannot duplicate sections.
    from training.bc_sequence_supervised_report import report
    report()
    write_report(result,gate,protocol)
    commands = [[sys.executable,'-m','pytest','-q','tests/test_bc_sequence_transfer.py','tests/test_bc_sequence_supervised.py',
                 'tests/test_bc_score_rl.py','tests/test_tetris_env.py','tests/parity'],
                ['node','--test',*map(str,sorted((ROOT/'tests').glob('*.test.cjs')))]]
    codes = []
    for i,command in enumerate(commands):
        with (AUDIT/f'regression_{i}.log').open('w') as stream:
            codes.append(subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=1800).returncode)
    record(AUDIT/'regression.json',dict(at=now(),returncodes=codes))
    if any(codes):
        raise RuntimeError('Audit regression failed')
    operation(f'archived-policy diagnostic complete: common_score={result["paired_to_bc"]["candidate"]["common_score_ratio"]}; gate={gate}; regressions={codes}; research terminated without more training')
    with (ROOT/'CHANGELOG.md').open('a') as stream:
        stream.write('\n- Added a single pre-frozen 16-seed/20K deployment diagnostic of the stopped epoch1 model; quantified actual scoring, four-clears and survival without more training.\n')
    subprocess.run(['git','add','OPERATIONS.md','CHANGELOG.md','reports/experiments/bc_sequence_transfer_results.md',
        'reports/experiments/bc_sequence_transfer_final.md','reports/experiments/bc_sequence_transfer_evidence.json'],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m','Quantify stopped sequence model with one frozen deployment audit'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','codex/bc-sequence-transfer'],cwd=ROOT,check=True)
    record(AUDIT/'completed.json',dict(at=now(),gate=gate,regression=codes))
    status('completed',reason='stopped_bc_retention_and_audited',confirmation=False,trained_updates=1280,audit=gate)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('policy_audit_failed',error=repr(exc))
        operation(f'policy audit failure/interruption: {type(exc).__name__}: {exc}')
        raise
