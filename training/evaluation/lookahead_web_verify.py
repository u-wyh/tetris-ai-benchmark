"""Verify real browser HTTP traces against frozen actions and the official Core."""
import argparse
import json
from pathlib import Path

from training.evaluation.web_longrun_verify import verify_chunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--pending', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--frozen', action='store_true')
    args = p.parse_args()
    if args.frozen:
        import numpy as np
        import torch
        from sb3_contrib import MaskablePPO
        from training.public_lookahead import PublicLookaheadController
        from training.web_inference import STUDENT
        from training.evaluation.hybrid_queue import BC
        torch.set_num_threads(1)
        planner = PublicLookaheadController(MaskablePPO.load(STUDENT, device='cpu'),
                                           MaskablePPO.load(BC, device='cpu'))
        rows = json.loads(args.pending.read_text())['entries']
        for row in rows:
            mask = np.zeros(1840, dtype=bool)
            mask[row['legal_ids']] = True
            action, detail = planner.decide(None, mask, row['before'])
            assert action == row['action'], 'HTTP action differs from frozen offline planner'
            assert detail['student_action'] == row['response']['studentActionId']
            assert detail['bc_action'] == row['response']['bcActionId']
            assert detail['override'] == row['response']['override']
    print(json.dumps(verify_chunk(args.run, args.seed, args.pending)))


if __name__ == '__main__':
    main()
