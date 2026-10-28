"""Complete clear-count and death-only terminal-risk tables from the fixed audit."""
import json
from pathlib import Path
from statistics import mean

from training.bc_sequence_transfer import ROOT, read_record
from training.bc_sequence_supervised import WORK
from training.bc_sequence_policy_audit import LABEL


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
    path = ROOT/'reports/experiments/bc_sequence_transfer_results.md'
    text = path.read_text().split('\n### 补充诊断：完整消行分布与真实死亡终局')[0]
    path.write_text(text+'\n'.join(lines)+'\n')
    path = ROOT/'reports/experiments/bc_sequence_transfer_evidence.json'
    evidence = json.loads(path.read_text())
    evidence['policy_audit']['clear_and_terminal_details'] = details
    path.write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(details,ensure_ascii=False))


if __name__ == '__main__':
    main()
