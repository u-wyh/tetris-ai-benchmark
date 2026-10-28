"""Complete clear-count and death-only terminal-risk tables from the fixed audit."""
import json
import os
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
    path = ROOT/'reports/experiments/bc_sequence_transfer_results.md'
    text = path.read_text().split('\n### 补充诊断：完整消行分布与真实死亡终局')[0]
    path.write_text(text+'\n'.join(lines)+'\n')
    path = ROOT/'reports/experiments/bc_sequence_transfer_evidence.json'
    evidence = json.loads(path.read_text())
    evidence['policy_audit']['clear_and_terminal_details'] = details
    evidence['policy_audit']['teacher_forced_alignment'] = alignment
    path.write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    path = ROOT/'reports/experiments/bc_sequence_transfer_final.md'
    final = path.read_text().splitlines()
    for i,line in enumerate(final):
        if line.startswith('1. **'):
            final[i] = ('1. **PPO高分是否依赖连续动作？** 连续128块接管在BC状态上有可靠的有限窗口收益；'
                '自然轨迹也表现为多块布局构建。但长度变化同时改变PPO控制块数，本轮没有等控制块数的分散接管对照，'
                '因此不能证明“必须连续”或把收益唯一归因于布局建立。')
        if line.startswith('8. **') and '完整16块匹配' not in line:
            final[i] = line + (' 留出教师状态上的完整16块匹配仍为0（PPO自然片段0/7,232窗口、BC分布片段0/226窗口），'
                '逐帧动作学习未形成完整教师序列复现；此为描述性证据，替代动作也可能得分，不能作为唯一因果解释。')
    path.write_text('\n'.join(final)+'\n')
    print(json.dumps(details,ensure_ascii=False))


if __name__ == '__main__':
    if '--teacher-only' in sys.argv:
        print(json.dumps(sequence_alignment(),ensure_ascii=False))
    else:
        main()
