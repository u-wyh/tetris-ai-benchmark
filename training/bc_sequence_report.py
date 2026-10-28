"""Evidence-only stage-I reporting and finite research close-out."""
import subprocess
import sys
from pathlib import Path

import numpy as np

from training.train_ppo import ROOT, now


def report():
    from training.bc_sequence_transfer import RUN, LENGTHS, SEEDS, STARTS, FOLLOW, read_record
    decision = read_record(RUN / 'decision.json')
    pairs = read_record(RUN / 'pairs.json')
    native = read_record(RUN / 'native_analysis.json')
    protocol = read_record(RUN / 'protocol.json')
    lines = ['# BC 连续策略迁移：第一阶段实测', '', f'生成时间：{now()}。', '',
             f'协议提交 `{protocol["git_commit"]}`；完整协议、源码/模型 SHA256、种子审计：`runs/bc_sequence_transfer/protocol.json`。',
             '冻结 BC/PPO 推理，没有训练、更新优化器或改动 Core。达到观察上限全部右截断。', '',
             '## 自然对局', '', '|模型|局数|平均已观测存活|真死亡|平均正式分|四消次数/全部消行事件|Hold比例|平均高度|平均洞|深井≥4比例|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for source in ('bc', 'ppo'):
        rows = [r for r in native if r['source'] == source]
        four = sum(r['clears'][4] for r in rows)
        clear = sum(sum(r['clears'][1:]) for r in rows)
        lines.append(f'|{source}|{len(rows)}|{np.mean([r["pieces"] for r in rows]):.1f}|{sum(r["death"] for r in rows)}|'
                     f'{np.mean([r["score"] for r in rows]):.0f}|{four}/{clear} ({four/max(1,clear):.2%})|'
                     f'{np.mean([r["hold_fraction"] for r in rows]):.2%}|{np.mean([r["mean_height"] for r in rows]):.2f}|'
                     f'{np.mean([r["mean_holes"] for r in rows]):.2f}|{np.mean([r["well_ge4_fraction"] for r in rows]):.2%}|')
    lines += ['', '## 四消前连续片段', '',
              '以下是所有有足够前序块的四消窗口；同局窗口重叠，不能作为独立样本。未来观察满2,048块且未死亡、2,048块内实际死亡、右截断分别列示；“观察期内存活”不等于长期安全。井深只描述公开棋盘的列高差，不代表无洞可用井。', '',
              '|模型|前序块数|未来分类|窗口数|平均Hold次数|平均最大高度|平均洞数|平均最大井深|平均正式分增量|',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for source in ('bc', 'ppo'):
        for length in LENGTHS:
            for category in ('survived_2048', 'death_within_2048', 'right_censored'):
                windows = [w for r in native if r['source'] == source
                           for w in r['windows'][str(length)][category]]
                if windows:
                    means = [np.mean([w[k] for w in windows]) for k in
                             ('hold', 'max_height', 'mean_holes', 'max_well', 'score_gain')]
                    lines.append(f'|{source}|{length}|{category}|{len(windows)}|' + '|'.join(f'{x:.2f}' for x in means) + '|')
    lines += ['', '## BC 实际访问状态的接管', '',
              f'预定 {len(SEEDS)} seed × {len(STARTS)} 状态；每次接管后 BC 最多再控制 {FOLLOW} 块。单步仅诊断，16/32/64/128 才参与准入。BC快照恢复分支已与自然BC逐块比对（动作、公开队列、正式分数与布局指标相同）。', '',
              '|PPO接管块数|状态对数|共同窗口得分比|正向seed|98.75%区间|BC/干预真死亡|干预/BC已观测生存|BC/干预高位比例|BC/干预新增洞/100块|准入|',
              '|---:|---:|---:|---:|---|---|---:|---|---|---|']
    # One-step uses the same exact common-window accounting, without selection inference.
    one = [p for p in pairs if p['length'] == 1]
    if one:
        a = sum(p['baseline']['score'] for p in one)
        b = sum(p['intervention']['score'] for p in one)
        lines.append(f'|1|{len(one)}|{b/max(1,a):.5f}|—|诊断，不用于准入|'
                     f'{sum(p["baseline"]["death"] for p in one)}/{sum(p["intervention"]["death"] for p in one)}|—|—|—|不适用|')
    for length in LENGTHS:
        item = decision['lengths'][str(length)]
        a, b = item['baseline'], item['intervention']
        ci = item['ratio_ci98_75']
        ratio = item['common_score_ratio']
        lines.append(f'|{length}|{sum(p["length"] == length for p in pairs)}|{ratio:.5f}|{item["positive_seeds"]}/6|'
                     f'[{ci[0]:.5f}, {ci[1]:.5f}]|{a["deaths"]}/{b["deaths"]}|{b["observed"]/max(1,a["observed"]):.5f}|'
                     f'{a["high_fraction"]:.3%}/{b["high_fraction"]:.3%}|'
                     f'{a["new_holes_per100"]:.3f}/{b["new_holes_per100"]:.3f}|{item["passed"]}|')
    lines += ['', '失败门槛：', '']
    for length in LENGTHS:
        item = decision['lengths'][str(length)]
        lines.append(f'- {length}块：' + ('全部通过' if item['passed'] else '、'.join(k for k, v in item['checks'].items() if not v)) + '。')
    divergent = sum(p['first_public_sequence_divergence'] is not None for p in pairs)
    lines += ['', f'{divergent}/{len(pairs)} 条干预配对的公开当前块/预览/Hold序列出现分歧；由逐步 Hold 和落下方块记录保留，不将这种分歧强行纠正。完整逐seed结果：`runs/bc_sequence_transfer/pairs.json`。', '',
              '## 解释范围', '',
              '自然PPO的高四消率与连续布局相关，但这里没有对布局历史做随机消融，不能证明“必须连续N块”。有限接管对比是政策干预的净效应，包含不同Hold、布局及交回BC后的行为。以seed聚合避免将同局状态或重叠片段当成独立证据。只比较共同窗口得分，死亡另计；早死不能凭短期高分获得通过。', '',
              '第一阶段通过：**' + str(decision['passed']) + '**。' +
              ('满足启动一种监督方案的必要证据；尚不代表长期有效。' if decision['passed'] else
               '按预注册规则停止，不采集监督训练数据、不运行梯度更新、不启动开发或独立确认。'), '',
              '证据：`runs/bc_sequence_transfer/{native,branches,snapshots,native_analysis.json,pairs.json,decision.json}`。每个pickle envelope包含SHA256、完整Core/RNG快照与逐块轨迹；协议绑定模型与源码。']
    out = ROOT / 'reports/experiments'
    (out / 'bc_sequence_transfer_results.md').write_text('\n'.join(lines)+'\n')
    if decision['passed']:
        return
    best = max(decision['lengths'].values(), key=lambda v: v['common_score_ratio'] or 0)
    worst_death = max(v['intervention']['deaths']-v['baseline']['deaths'] for v in decision['lengths'].values())
    answers = ['# BC 连续策略迁移：最终结论', '', f'完成时间：{now()}；停止原因：`stopped_stage1_gate`。', '',
        '1. **是否依赖连续动作？** PPO自然轨迹表现为四消前多块布局；现有观察和有限接管不能严格证明连续动作是必要条件。单步和16/32/64/128块均已实测，详见结果表。',
        '2. **BC局面上是否有效？** 没有任何预定长度同时通过跨seed得分、置信区间和生存/风险门槛；没有足够可靠的迁移证据。',
        '3. **是否真正训练新单模型？** 否。第一阶段失败后按授权停止，新增梯度更新0，新增模型/优化器checkpoint0。原BC/PPO/B12冻结。',
        '4. **是否学到新四消？** 没有训练，不能声称学到。PPO临时接管中的四消不是BC的新行为。',
        f'5. **效率提高多少？** 四种预定长度中最高的实测共同窗口得分变化为 {(best["common_score_ratio"]-1)*100:+.3f}%；这是探索性最高值，不是合格候选或训练收益。各长度和单步对照见结果。',
        f'6. **生存损失多少？** 各接管长度的真死亡、已观测生存与右截断见结果表；相对纯BC的最多新增死亡为 {worst_death}/24 条状态分支。2,048块随访不能推断100K生存。',
        '7. **独立确认是否通过？** 未启动。研究seed仅用于第一阶段；预留训练、开发和确认组未消耗。没有登记冠军。',
        '8. **失败根因？** 预定连续接管在BC状态分布上没有同时提供可靠得分收益与安全性；具体失效门槛见结果。自然PPO表现不能直接外推到BC棋盘。此设计不能把原因唯一归结为Hold、井结构或神经网络容量，禁止作超出证据的因果判断。',
        '9. **是否继续投入训练？** 本轮不值得进入监督训练；按冻结预算结束。若另立研究，应先提供更强的状态分布兼容或完整布局迁移证据，不能靠扩大epoch解决本轮准入失败。', '',
        '没有重复普通BC+PPO或KL-PPO2M；没有改动Core、正式计分、动作协议、服务和冻结模型。', '',
        '详细数据和限制：[第一阶段结果](bc_sequence_transfer_results.md)；[冻结设计](bc_sequence_transfer_design.md)。回归测试状态在 `runs/bc_sequence_transfer/regression.json` 与最终提交中记录。']
    (out / 'bc_sequence_transfer_final.md').write_text('\n'.join(answers)+'\n')


def close_out():
    from training.bc_sequence_transfer import RUN, operation, read_record, record, status
    report()
    if read_record(RUN / 'decision.json')['passed']:
        return
    commands = [[sys.executable, '-m', 'pytest', '-q', 'tests/test_bc_sequence_transfer.py',
                 'tests/test_bc_score_rl.py', 'tests/test_tetris_env.py', 'tests/parity'],
                ['node', '--test', *map(str, sorted((ROOT / 'tests').glob('*.test.cjs')))]]
    results = []
    for i, command in enumerate(commands):
        with (RUN / f'regression_{i}.log').open('w') as stream:
            results.append(subprocess.run(command, cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=1800).returncode)
    record(RUN / 'regression.json', dict(at=now(), returncodes=results, commands=commands))
    operation(f'final regression return codes={results}; stage-I negative report complete; no training')
    if any(results):
        raise RuntimeError('Required regression failed; inspect logs before publishing')
    changelog = ROOT / 'CHANGELOG.md'
    with changelog.open('a') as stream:
        stream.write('\n### BC sequence transfer research (2026-10-28)\n\n'
                     '- Added bounded, resumable paired sequence takeovers on fresh BC states with frozen PPO.\n'
                     '- Reported the pre-registered stage-I stop; no new champion or trained policy.\n'
                     '- Validated Python, Core/JS parity and JavaScript regression tests.\n')
    report_path = ROOT / 'reports/experiments/bc_sequence_transfer_final.md'
    with report_path.open('a') as stream:
        stream.write('\nPython、Parity与JS回归均通过，详见回归日志。\n')
    subprocess.run(['git', 'add', 'OPERATIONS.md', 'CHANGELOG.md',
                    'reports/experiments/bc_sequence_transfer_results.md',
                    'reports/experiments/bc_sequence_transfer_final.md'], cwd=ROOT, check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode:
        subprocess.run(['git', 'commit', '-m', 'Report finite sequence-transfer feasibility and stop decision'], cwd=ROOT, check=True)
    subprocess.run(['git', 'push', '-u', 'origin', 'codex/bc-sequence-transfer'], cwd=ROOT, check=True)
    status('completed', reason='stopped_stage1_gate', regression=results, trained=False,
           git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())


if __name__ == '__main__':
    close_out()
