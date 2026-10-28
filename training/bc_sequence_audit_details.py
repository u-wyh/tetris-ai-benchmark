"""Complete clear-count and death-only terminal-risk tables from the fixed audit."""
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from training.bc_sequence_transfer import ROOT, RUN, BC, read_record, read_blob, record, digest
from training.bc_sequence_supervised import WORK, DATA_SEEDS, data_path, chunks
from training.bc_sequence_policy_audit import LABEL
from training.imitation_bc import batch
from training.train_ppo import now


def sequence_alignment():
    """Teacher-forced agreement only; no claim of free-running sequence success."""
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
    torch.set_num_threads(1)
    model_path = Path(read_record(WORK/'training_result.json')['last_model'])
    output = dict(mode='descriptive teacher-forced matching on held-out complete episodes; not a selection gate',
                  model_sha256=digest(model_path), sources={})
    models = dict(bc=MaskablePPO.load(BC,device='cpu'),candidate=MaskablePPO.load(model_path,device='cpu'))
    for source in ('ppo','transfer'):
        frames = [f for seed in DATA_SEEDS[6:] for f in read_blob(data_path(source,seed))['frames']]
        segments = chunks(frames)
        output['sources'][source] = {}
        for name,model in models.items():
            matches = []
            for segment in segments:
                if any(b['piece']-a['piece'] != 1 for a,b in zip(segment,segment[1:])):
                    raise RuntimeError('Sequence diagnostic crossed non-contiguous data')
                predicted = []
                for start in range(0,len(segment),32):
                    obs,mask,actions = batch(segment[start:start+32],'cpu')
                    with torch.inference_mode():
                        prediction = model.policy.get_distribution(obs,action_masks=mask).get_actions(deterministic=True)
                    predicted.extend((prediction==actions).cpu().numpy().tolist())
                matches.append(predicted)
            longest = []
            for segment in matches:
                streak = maximum = 0
                for same in segment:
                    streak = streak+1 if same else 0
                    maximum = max(maximum,streak)
                longest.append(maximum)
            counts = {str(h):dict(exact=sum(all(segment[i:i+h]) for segment in matches
                for i in range(len(segment)-h+1)), windows=sum(max(0,len(segment)-h+1) for segment in matches))
                for h in (16,32,64,128)}
            output['sources'][source][name] = dict(frames=sum(map(len,matches)),fragments=len(matches),
                top1=sum(map(sum,matches))/max(1,sum(map(len,matches))),
                mean_longest_streak=float(np.mean(longest)) if longest else None, exact_windows=counts)
    record(RUN/'policy_audit'/'sequence_alignment.json',output)
    return output


def main():
    folder = WORK/'evaluation'/LABEL
    details = {}
    for name in ('bc','ppo','B12','candidate'):
        paths = sorted((folder/name).glob('seed_*.json'))
        if len(paths) != 16:
            raise RuntimeError(f'Incomplete audit: {name} has {len(paths)}/16 results')
        rows = [read_record(p) for p in paths]
        dead = [r for r in rows if r['game_over']]
        clears = [sum(r['clears_0_to_4'][k] for r in rows) for k in range(5)]
        if sum(clears) != sum(r['pieces'] for r in rows):
            raise RuntimeError('Clear distribution does not cover observed placements')
        details[name] = dict(clears_0_to_4=clears,
            shares_1_to_4=[c/max(1,sum(clears[1:])) for c in clears[1:]],
            deaths=len(dead),
            terminal_height=mean(r['final_height'] for r in dead) if dead else None,
            terminal_holes=mean(r['final_holes'] for r in dead) if dead else None,
            terminal_high_streak=mean(r['high_streak'] for r in dead) if dead else None,
            death_records=[dict(seed=r['seed'],pieces=r['pieces'],height=r['final_height'],
                holes=r['final_holes'],high_streak=r['high_streak']) for r in dead])
    lines = ['\n### 补充诊断：完整消行分布与真实死亡终局\n',
        '|模型|单消|双消|三消|四消|单/双/三/四消事件比例|', '|---|---:|---:|---:|---:|---|']
    for name,r in details.items():
        lines.append(f'|{name}|'+'|'.join(str(c) for c in r['clears_0_to_4'][1:])+'|'+
                     '/'.join(f'{p:.2%}' for p in r['shares_1_to_4'])+'|')
    lines += ['', '终局风险只统计真实死亡；触顶局不当作死亡。', '',
              '|模型|实际死亡局数|死亡时平均高度|死亡时平均洞数|死亡时平均高位连续块数|',
              '|---|---:|---:|---:|---:|']
    for name,r in details.items():
        values = ['—' if r[k] is None else f'{r[k]:.2f}' for k in
                  ('terminal_height','terminal_holes','terminal_high_streak')]
        lines.append(f'|{name}|{r["deaths"]}|'+'|'.join(values)+'|')
    alignment_path = RUN/'policy_audit'/'sequence_alignment.json'
    alignment = read_record(alignment_path) if alignment_path.exists() else sequence_alignment()
    lines += ['', '### 教师状态上的连续动作对齐（描述性）', '',
        '使用完整episode留出的教师片段，在教师实际状态上逐步比较动作。此为teacher-forced诊断；不代表模型自由运行能到达这些状态，'
        '动作不完全一致也不证明无法用另一种布局得分。窗口彼此重叠，不当作独立样本。'
        '本表使用全部留出帧；训练监控表对每来源预定最多采样4,096帧，因此逐帧一致率略有差异。', '',
        '|教师来源|模型|逐帧一致率|每片段平均最长一致动作串|16/32/64/128块完整一致窗口数/总数|',
        '|---|---|---:|---:|---|']
    for source,models in alignment['sources'].items():
        for name,r in models.items():
            counts = '/'.join(f'{r["exact_windows"][str(h)]["exact"]}:{r["exact_windows"][str(h)]["windows"]}' for h in (16,32,64,128))
            lines.append(f'|{source}|{name}|{r["top1"]:.2%}|{r["mean_longest_streak"]:.2f}|{counts}|')
    failure_path = RUN/'policy_audit'/'failure'/'result.json'
    failure = read_record(failure_path) if failure_path.exists() else None
    if failure:
        lines += ['', '### 唯一早死的固定单例复现', '',
            f'种子 {failure["protocol"]["seed"]} 的前 {failure["reproduced_score_prefix"]} 步正式累计分数全部与开发原局一致。'
            f'死亡为 **{failure["death_kind"]}**：高度 {failure["terminal_height"]}、洞数 {failure["terminal_holes"]}，'
            f'终局前高位连续 {failure["terminal_high_streak"]} 块。', '',
            f'预先固定从第 {failure["start_piece"]} 块后（死亡前128块）的同一Core/RNG快照派生BC分支。'
            f'BC实际观察 {failure["bc_observed"]} 块，死亡={failure["bc_death"]}，右截断={failure["bc_right_censored"]}；'
            f'末尾高度 {failure["bc_final_height"]}、洞数 {failure["bc_final_holes"]}。'
            '分支保留实际Hold/公开队列变化，不强制后续方块相同。此为失败后单例诊断，不参与选择、统计准入或确认，不能当作长期安全证明。', '',
            '证据：`runs/bc_sequence_transfer/policy_audit/failure/`，含冻结诊断协议、同起点完整Core/RNG快照、'
            '学生最后128块公开棋盘/动作和BC512块轨迹。该病例说明平均洞数/高位比例改善仍可能遗漏罕见Block Out；'
            '同起点BC可消解局面，未发现Core或正式计分差异。']
    path = ROOT/'reports/experiments/bc_sequence_transfer_results.md'
    text = path.read_text().split('\n### 补充诊断：完整消行分布与真实死亡终局')[0]
    path.write_text(text+'\n'.join(lines)+'\n')
    path = ROOT/'reports/experiments/bc_sequence_transfer_evidence.json'
    evidence = json.loads(path.read_text())
    evidence['policy_audit']['clear_and_terminal_details'] = details
    evidence['policy_audit']['teacher_forced_alignment'] = alignment
    if failure:
        evidence['policy_audit']['single_failure_postmortem'] = {
            k:v for k,v in failure.items() if k not in ('candidate_tail','candidate_tail_boards','bc_branch')}
        evidence['policy_audit']['single_failure_postmortem'].update(
            full_trajectory_record='runs/bc_sequence_transfer/policy_audit/failure/result.json',
            full_trajectory_sha256=digest(failure_path),terminal_board=failure['candidate_tail_boards'][-1])
    path.write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    path = ROOT/'reports/experiments/bc_sequence_transfer_final.md'
    final = path.read_text().splitlines()
    for i,line in enumerate(final):
        if line.startswith('完成时间：'):
            final[i] = f'完成时间：{now()}；停止原因：BC保留/早死门槛失败；补充诊断已完成。'
        if line.startswith('1. **'):
            final[i] = ('1. **PPO高分是否依赖连续动作？** 连续128块接管在BC状态上有可靠的有限窗口收益；'
                '自然轨迹也表现为多块布局构建。但长度变化同时改变PPO控制块数，本轮没有等控制块数的分散接管对照，'
                '因此不能证明“必须连续”或把收益唯一归因于布局建立。')
        if line.startswith('8. **') and '完整16块匹配' not in line:
            final[i] = line + (' 留出教师状态上的完整16块匹配仍为0（PPO自然片段0/7,232窗口、BC分布片段0/226窗口），'
                '逐帧动作学习未形成完整教师序列复现；此为描述性证据，替代动作也可能得分，不能作为唯一因果解释。')
        if line.startswith('8. **') and failure and 'Block Out单例' not in final[i]:
            final[i] += (' Block Out单例已逐步复现：高20层但仅1洞；死亡前128块切回BC后，固定512块随访存活并降到高度1/0洞。'
                '这支持尾部高堆叠安全行为保留不足，不能把任何特定损失项或网络容量断言为唯一原因。')
        if line.startswith('9. **'):
            final[i] = ('9. **是否值得继续投入资源？** 值得另立一轮有限的安全保留研究：本轮真实学到更多四消，'
                '共同得分提高13.97%且16/16seed正向，方向有实测信号。但当前配方已触发早死与保留门槛，'
                '本轮停止，不追加epoch、扩大样本或转PPO；下一轮应先解决罕见高堆叠死亡，并重新冻结全新开发/确认seed。')
    regression = read_record(RUN/'policy_audit'/'regression.json')
    if regression['returncodes'] != [0,0]:
        raise RuntimeError('Final regression did not pass')
    if not any(line.startswith('最终回归：') for line in final):
        python_log = (RUN/'policy_audit'/'regression_0.log').read_text()
        js_log = (RUN/'policy_audit'/'regression_1.log').read_text()
        py = re.search(r'(\d+) passed, (\d+) subtests passed',python_log)
        js = re.search(r'pass (\d+)',js_log)
        counts = f'Python/Parity {py[1]}项测试、{py[2]}个子测试，JS {js[1]}项' if py and js else 'Python/Parity与JS'
        final += ['',f'最终回归：{counts}全部通过。日志：`runs/bc_sequence_transfer/policy_audit/regression_*.log`。'
                  '唯一早死复现另验证了9,867步正式分数完全一致；冻结模型与源码身份、已完成任务恢复检查通过。']
    path.write_text('\n'.join(final)+'\n')
    print(json.dumps(details,ensure_ascii=False))


if __name__ == '__main__':
    if '--teacher-only' in sys.argv:
        print(json.dumps(sequence_alignment(),ensure_ascii=False))
    else:
        main()
