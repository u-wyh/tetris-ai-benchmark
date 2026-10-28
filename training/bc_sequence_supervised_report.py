"""Report and close the bounded supervised route, including negative results."""
import subprocess
import sys

from training.train_ppo import ROOT, now
from training.bc_sequence_transfer import RUN, read_record, record, operation, status
from training.evaluation.hybrid_queue import digest
from pathlib import Path


def report():
    from training.bc_sequence_report import report as first_report
    from training.bc_sequence_diagnostics import diagnostics
    work = RUN/'supervised'
    outcome = read_record(work/'outcome.json')
    training = read_record(work/'training_result.json')
    dataset = read_record(work/'dataset_manifest.json')
    first_report()
    evidence = diagnostics()
    evidence['supervised'] = dict(protocol=read_record(work/'protocol.json'), outcome=outcome,
        training=training,dataset=dataset,
        gradient_save_load=read_record(work/'gradient_save_load.json') if (work/'gradient_save_load.json').exists() else None)
    lines = ['\n## 第二阶段：真实监督迁移\n',
        f'结束原因：`{outcome["reason"]}`；真实Actor更新 **{training["updates"]}** 次；训练结果：`{training["reason"]}`。', '',
        '只更新Actor，Critic与原BC保持逐张量一致。数据按完整seed/episode分组，BC、PPO自然与派生接管轨迹不跨训练/验证；无未来信息进入网络。', '',
        '|划分|BC保留帧|PPO自然连续示范帧|BC分布接管连续示范帧|', '|---|---:|---:|---:|']
    if training.get('last_model'):
        evidence['supervised']['last_model_sha256'] = digest(Path(training['last_model']))
    for partition, counts in dataset['counts'].items():
        lines.append(f'|{partition}|{counts["bc"]}|{counts["ppo"]}|{counts["transfer"]}|')
    lines += ['', '|epoch|BC动作一致率|PPO自然示范动作率/NLL|BC分布示范动作率/NLL|学习门槛|',
              '|---:|---:|---|---|---|']
    for row in training['history']:
        v = row['validation']
        lines.append(f'|{row["epoch"]}|{v["bc"]["top1"]:.2%}|{v["ppo"]["top1"]:.2%}/{v["ppo"]["loss"]:.4f}|'
            f'{v["transfer"]["top1"]:.2%}/{v["transfer"]["loss"]:.4f}|{row.get("checks","初始化")}|')
    if training.get('last_model'):
        lines += ['',f'最后实际训练checkpoint：`{training["last_model"]}`；SHA256：`{evidence["supervised"]["last_model_sha256"]}`。'
                  '未通过学习门槛的checkpoint仅归档，不作为合格候选进入开发评测。']
    evaluations = {}
    for label in ('development_20k','development_100k','confirmation_100k'):
        path = work/'evaluation'/label/'summary.json'
        if not path.exists():
            continue
        result = read_record(path)
        gate = read_record(work/f'{label}_gate.json')
        evaluations[label] = dict(result=result,gate=gate)
        lines += ['',f'### {label}', '',
                  '|模型|平均已观测存活|P10|触顶|<10K早死|平均正式分|四消比例|推理ms/块|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,row in result['summary'].items():
            lines.append(f'|{name}|{row["mean_pieces"]:.1f}|{row["p10"]:.1f}|{row["capped"]}/{row["games"]}|{row["early"]}|'
                f'{row["mean_score"]:.0f}|{row["four_fraction"]:.3%}|{row["mean_inference_ms"]:.3f}|')
        p = result['paired_to_bc']['candidate']
        lines += ['',f'共同窗口得分比 **{p["common_score_ratio"]:.5f}**；正向seed **{p["positive_seeds"]}**；'
                  f'95% seed bootstrap区间 {p["ratio_ci95"]}；门槛 `{gate}`。']
    evidence['supervised']['evaluations'] = evaluations
    import json
    (ROOT/'reports/experiments/bc_sequence_transfer_evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    with (ROOT/'reports/experiments/bc_sequence_transfer_results.md').open('a') as stream:
        stream.write('\n'.join(lines)+'\n')
    final_result = list(evaluations.values())[-1]['result'] if evaluations else None
    score = ('未运行候选正式开发评测；无法报告新模型得分提升。' if final_result is None else
             f'最后允许评测的候选共同窗口得分变化为 {(final_result["paired_to_bc"]["candidate"]["common_score_ratio"]-1)*100:+.3f}%。')
    survival = ('未运行候选正式开发评测；无法外推新模型长局生存。' if final_result is None else
        f'平均已观测生存候选/BC为 {final_result["summary"]["candidate"]["mean_pieces"]/final_result["summary"]["bc"]["mean_pieces"]:.5f}，'
        f'低于10K早死候选/BC为 {final_result["summary"]["candidate"]["early"]}/{final_result["summary"]["bc"]["early"]}；观察上限右截断。')
    behavior = ('监督验证动作学习不等于实际四消能力，未通过正式对局证明新四消行为。' if final_result is None else
        f'实战消行事件中的四消比例，候选 {final_result["summary"]["candidate"]["four_fraction"]:.3%}，BC {final_result["summary"]["bc"]["four_fraction"]:.3%}；'
        '是否合格还要同时满足生存与效率门槛。')
    failed = (training['history'][-1].get('checks',{}) if not evaluations and training['history'] else
              list(evaluations.values())[-1]['gate'] if evaluations else training['reason'])
    final = ['# BC 连续策略迁移：最终结论', '',f'完成时间：{now()}；停止/完成原因：`{outcome["reason"]}`。', '',
        '1. **PPO高分是否依赖连续动作？** 128块冻结接管比BC提高3.90%，六seed全部正向，16/32块仅小幅收益，64块为2.25%。支持“多块布局有助于高分”的假设，但不能证明128块是必要长度或唯一机制。',
        '2. **BC局面上仍有效吗？** 第一阶段24个预定BC状态中，128块接管+2,048块BC随访通过冻结门槛，98.75%得分比区间[1.03359,1.04483]；未观察到死亡损失。此证据仅覆盖有限窗口。',
        f'3. **是否真正训练新单模型？** {"是" if training["updates"] else "否"}；实际Actor梯度更新 {training["updates"]} 次。'
        '模型保持原公开输入和动作协议；模型、独立Adam优化器、RNG、游标、Git与SHA256按事务保存，epoch checkpoint独立归档。',
        '4. **是否学到新四消？** '+behavior,
        '5. **效率提升多少？** '+score+'第一阶段3.90%是冻结PPO临时接管收益，不能记为BC训练收益。',
        '6. **生存损失多少？** '+survival,
        f'7. **是否通过独立确认？** {outcome["confirmation"]}。'+('候选已冻结且通过新32seed确认。' if outcome['confirmation'] else '未通过所有开发门槛，不登记新冠军；未启动的确认组保持未使用。'),
        f'8. **失败根因？** 本轮实际终止条件：`{failed}`。有限窗口策略收益不会自动转换成监督更新后的稳定Actor行为；验证上的源分布/BC分布学习与保留约束，或实战得分/生存门槛中的失败项给出了实测限制。不能唯一归因于网络容量。',
        '9. **是否继续投入资源？** '+('可基于独立确认结果讨论后续应用；本轮预算已结束。' if outcome['confirmation'] else
        '本轮停止，不增加epoch、样本或PPO训练。先解决已观测到的行为保留/迁移问题再考虑新的独立研究。'), '',
        '详细表、逐seed配对、协议和SHA256见 [结果](bc_sequence_transfer_results.md)、[证据JSON](bc_sequence_transfer_evidence.json)。'
        '没有修改Core/正式计分/动作规则、覆盖冻结模型或发布v1.0.0。']
    (ROOT/'reports/experiments/bc_sequence_transfer_final.md').write_text('\n'.join(final)+'\n')


def close_out():
    work = RUN/'supervised'
    outcome = read_record(work/'outcome.json')
    if (work/'closed.json').exists():
        return
    report()
    commands = [[sys.executable,'-m','pytest','-q','tests/test_bc_sequence_transfer.py',
                 'tests/test_bc_sequence_supervised.py','tests/test_bc_score_rl.py','tests/test_tetris_env.py','tests/parity'],
                ['node','--test',*map(str,sorted((ROOT/'tests').glob('*.test.cjs')))]]
    codes = []
    for i,command in enumerate(commands):
        with (RUN/f'regression_{i}.log').open('w') as stream:
            codes.append(subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=1800).returncode)
    record(RUN/'regression.json',dict(at=now(),returncodes=codes,commands=commands))
    operation(f'supervised outcome={outcome["reason"]}; required Python/Parity/JS return codes={codes}')
    if any(codes):
        raise RuntimeError('Required regression failed; inspect logs')
    with (ROOT/'reports/experiments/bc_sequence_transfer_final.md').open('a') as stream:
        stream.write('\nPython、Parity与JS回归均通过，日志保存于 `runs/bc_sequence_transfer/regression_*.log`。\n')
    if '### BC sequence transfer research (2026-10-28)' not in (ROOT/'CHANGELOG.md').read_text():
      with (ROOT/'CHANGELOG.md').open('a') as stream:
        stream.write('\n### BC sequence transfer research (2026-10-28)\n\n'
                     '- Fresh paired finite PPO sequences on BC states passed the registered 128-piece feasibility gate (+3.90% common score).\n'
                     '- Ran the single bounded supervised transfer route; see the final report for actual updates and stop gates.\n'
                     '- Preserved Core, official scoring, frozen models and service configuration; Python, Parity and JS regressions passed.\n')
    subprocess.run(['git','add','OPERATIONS.md','CHANGELOG.md','reports/experiments/bc_sequence_transfer_results.md',
                    'reports/experiments/bc_sequence_transfer_final.md','reports/experiments/bc_sequence_transfer_evidence.json'],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m','Report bounded supervised sequence transfer evidence and outcome'],cwd=ROOT,check=True)
    subprocess.run(['git','push','-u','origin','codex/bc-sequence-transfer'],cwd=ROOT,check=True)
    record(work/'closed.json',dict(at=now(),outcome=outcome,regression=codes))
    status('completed',reason=outcome['reason'],confirmation=outcome['confirmation'],regression=codes)
