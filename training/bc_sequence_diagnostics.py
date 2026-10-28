"""Descriptive diagnostics of native/transfer distribution overlap, never a selection gate."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from training.bc_sequence_transfer import (RUN, SEEDS, LENGTHS, STARTS, ROOT, job_path,
    read_blob, read_record, record, native_analysis)
from training.evaluation.hybrid_queue import digest
from training.train_ppo import now


def average(rows, names):
    return {name: float(np.mean([r[name] for r in rows])) for name in names} if rows else {}


def diagnostics():
    records = {source: [read_blob(job_path(source, seed)) for seed in SEEDS] for source in ('bc', 'ppo')}
    # Public layout signature: two-row height bins, capped holes, two-row well bins.
    # This is descriptive support overlap, not an input or a posthoc training gate.
    def signature(r):
        return (r['height']//2, min(10, r['holes']), min(10, max(r['wells']))//2)
    hist = {source: Counter(signature(r) for game in games for r in game['rows'])
            for source, games in records.items()}
    totals = {s: sum(c.values()) for s, c in hist.items()}
    overlap = sum(min(hist['bc'][key]/totals['bc'], hist['ppo'][key]/totals['ppo'])
                  for key in hist['bc'].keys() | hist['ppo'].keys())
    native, nonfour = [], []
    for source, games in records.items():
        for game in games:
            summary = native_analysis(game)
            summary.pop('windows')
            rows = game['rows']
            summary['hole_free_well_ge4_fraction'] = float(np.mean([
                r['holes'] == 0 and max(r['wells']) >= 4 for r in rows]))
            native.append(summary)
            for length in LENGTHS:
                # Fixed non-overlapping endpoints; no performance-driven window selection.
                selected = []
                for end in range(length, len(rows)+1, length):
                    if rows[end-1]['clear'] == 4:
                        continue
                    segment = rows[end-length:end]
                    selected.append(dict(max_height=max(r['height'] for r in segment),
                        mean_holes=float(np.mean([r['holes'] for r in segment])),
                        max_well=max(max(r['wells']) for r in segment),
                        hold=sum(r['hold'] for r in segment),
                        fours=sum(r['clear'] == 4 for r in segment)))
                nonfour.append(dict(source=source, seed=game['key'][1], length=length,
                    count=len(selected), means=average(selected, ('max_height', 'mean_holes', 'max_well', 'hold', 'fours'))))
    prefixes = []
    for length in (1, *LENGTHS):
        for seed in SEEDS:
            for start in STARTS:
                path = job_path('branch', seed, start, length)
                if not path.exists():
                    continue
                result = read_blob(path)
                prefix = result['rows'][:length]
                after = result['rows'][length:]
                prefixes.append(dict(seed=seed, start=start, length=length,
                    takeover_observed=len(prefix), takeover_fours=sum(r['clear'] == 4 for r in prefix),
                    takeover_hold=sum(r['hold'] for r in prefix),
                    takeover_max_height=max(r['height'] for r in prefix),
                    takeover_last_holes=prefix[-1]['holes'],
                    handback_observed=len(after), handback_fours=sum(r['clear'] == 4 for r in after),
                    death=result['game_over'], right_censored=result['capped']))
    evidence = dict(at=now(), analysis='descriptive only; not used to alter registered gate',
                    protocol=read_record(RUN / 'protocol.json'), decision=read_record(RUN / 'decision.json'),
                    pairs=read_record(RUN / 'pairs.json'), native=native, nonfour_windows=nonfour,
                    layout_distribution_overlap=overlap, takeover_prefixes=prefixes,
                    artifact_sha256={str(p.relative_to(RUN)): digest(p) for p in sorted(RUN.rglob('*.pkl'))
                                     if not p.name.endswith('.checkpoint.pkl')})
    record(RUN / 'descriptive_evidence.json', evidence)
    target = ROOT / 'reports/experiments/bc_sequence_transfer_evidence.json'
    # Compact, reviewable manifest; full trajectories stay in runs.
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    result_path = ROOT / 'reports/experiments/bc_sequence_transfer_results.md'
    text = result_path.read_text()
    marker = '\n## 补充：分布与交回行为（描述性，不改变门槛）\n'
    text = text.split(marker)[0]
    lines = [marker, '',
        '此项统计在采样期间定义，未用于选择长度或修改准入。按公开棋盘高度/洞/井深分箱，BC与PPO自然轨迹分布的重叠质量为 '
        f'**{overlap:.2%}**；该指标只描述这三个粗特征，不能当作完整状态相似度。', '',
        '|模型|无洞且井深≥4的块比例（按局平均）|', '|---|---:|']
    for source in ('bc', 'ppo'):
        lines.append(f'|{source}|{np.mean([r["hole_free_well_ge4_fraction"] for r in native if r["source"] == source]):.3%}|')
    lines += ['', '自然对局实测策略推理时间（候选特征构建不计入）：']
    for source in ('bc', 'ppo'):
        selected = [r for r in native if r['source'] == source]
        lines.append(f'- {source}：{1000*sum(r["inference_seconds"] for r in selected)/sum(r["pieces"] for r in selected):.3f} ms/块。')
    lines += ['', '|接管长度|实际PPO控制块数|接管期四消次数|交回BC后四消次数|分支真死亡|',
              '|---:|---:|---:|---:|---:|']
    for length in (1, *LENGTHS):
        selected = [r for r in prefixes if r['length'] == length]
        lines.append(f'|{length}|{sum(r["takeover_observed"] for r in selected)}|'
                     f'{sum(r["takeover_fours"] for r in selected)}|{sum(r["handback_fours"] for r in selected)}|'
                     f'{sum(r["death"] for r in selected)}|')
    lines += ['', '非四消终点的对照窗口按每16/32/64/128块的固定非重叠端点取样，不按得分挑选；统计与逐seed原始配对、协议和所有轨迹/快照SHA256一起提交在 '
              '`reports/experiments/bc_sequence_transfer_evidence.json`。所有窗口均源自原自然对局，没有新增episode。', '',
              '|长度|Seed|共同窗口得分变化|BC/干预真死亡|BC/干预已观测块数（4状态合计）|',
              '|---:|---:|---:|---|---|']
    pairs = evidence['pairs']
    for length in LENGTHS:
        for seed in SEEDS:
            rows = [r for r in pairs if r['length'] == length and r['seed'] == seed]
            a = sum(r['baseline']['score'] for r in rows)
            b = sum(r['intervention']['score'] for r in rows)
            lines.append(f'|{length}|{seed}|{100*(b/max(1,a)-1):+.3f}%|'
                         f'{sum(r["baseline"]["death"] for r in rows)}/{sum(r["intervention"]["death"] for r in rows)}|'
                         f'{sum(r["baseline"]["pieces"] for r in rows)}/{sum(r["intervention"]["pieces"] for r in rows)}|')
    result_path.write_text(text+'\n'.join(lines)+'\n')
    return evidence


if __name__ == '__main__':
    diagnostics()
