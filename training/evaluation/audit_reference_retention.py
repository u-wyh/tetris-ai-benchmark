"""Diagnose policy drift on the existing training-state corpus, using CUDA.

This is an audit of shared public states, not new validation or a survival test.
"""

import argparse
import hashlib
import json
from pathlib import Path


def distribution_metrics(reference, candidate, legal_ids):
    import numpy as np

    legal = np.asarray(legal_ids, dtype=int)
    p, q = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if (p.ndim != 1 or p.shape != q.shape or len(legal) == 0
            or len(set(legal.tolist())) != len(legal) or (legal < 0).any()
            or (legal >= len(p)).any()):
        raise ValueError('Invalid probability dimensions or legal actions')
    for probs in (p, q):
        if (not np.isfinite(probs).all() or (probs < 0).any()
                or not np.isclose(probs[legal].sum(), 1, atol=1e-5, rtol=0)
                or not np.isclose(probs.sum(), probs[legal].sum(), atol=1e-7, rtol=0)):
            raise ValueError('Invalid masked distribution')
    rank = legal[np.lexsort((legal, -p[legal]))]
    action = int(np.argmax(q))
    # The floor makes underflow handling explicit; this is a diagnostic metric.
    kl = float(np.sum(q[legal] * (np.log(np.maximum(q[legal], 1e-12))
                                 - np.log(np.maximum(p[legal], 1e-12)))))
    return dict(reference_action=int(rank[0]), candidate_action=action,
                top1_same=action == int(rank[0]), top5_retained=action in rank[:5],
                reference_margin=float(p[rank[0]]-p[rank[1]]) if len(rank)>1 else 1.0,
                masked_kl_candidate_to_reference=max(0.0, kl))


def aggregate(rows):
    if not rows:
        return dict(states=0)
    return dict(states=len(rows), top1_agreement=sum(r['top1_same'] for r in rows)/len(rows),
                top5_retention=sum(r['top5_retained'] for r in rows)/len(rows),
                masked_kl_candidate_to_reference=sum(r['masked_kl_candidate_to_reference'] for r in rows)/len(rows))


def run(reference, candidates, output):
    import torch
    from sb3_contrib import MaskablePPO
    from training.evaluation.ppo30m_overnight_audit import _normal_rows
    from training.imitation_bc import batch
    from training.train_vector_transaction import training_source_identity
    from training.train_ppo import atomic_json, now

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; retention audit will not fall back to CPU')
    torch.set_num_threads(1)
    rows = _normal_rows()
    if not rows:
        raise RuntimeError('No existing normal-state corpus rows')
    models = {'reference': reference, **candidates}
    if 'reference' in candidates:
        raise ValueError('Candidate label reference is reserved')
    policies = {label: MaskablePPO.load(str(path), device='cuda').policy.eval()
                for label,path in models.items()}
    audits = {label: [] for label in candidates}
    with torch.no_grad():
        for row in rows:
            obs, mask, _ = batch([{**row, 'action': row['ppo_action']}], 'cuda')
            probs = {label: policy.get_distribution(obs, action_masks=mask).distribution.probs[0].cpu().numpy()
                     for label,policy in policies.items()}
            for label in candidates:
                record = distribution_metrics(probs['reference'], probs[label], row['legal_ids'])
                audits[label].append(dict(seed=row['seed'], step=row['step'], **record))
    root = Path(__file__).resolve().parents[2]
    support = ['training/evaluation/audit_reference_retention.py',
               'training/evaluation/ppo30m_overnight_audit.py', 'training/imitation_bc.py',
               'training/local_correction_data.py']
    result = dict(at=now(), device='cuda', sample_kind='existing training corpus: first4 normal rows per seed',
        interpretation='Paired state diagnostic; not new validation, independent samples, or evidence of survival improvement',
        probability_log_floor=1e-12, training_source_sha256=training_source_identity(),
        audit_source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in support},
        selected_rows_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        models={label: dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for label,path in models.items()},
        candidates={})
    for label, records in audits.items():
        result['candidates'][label] = dict(**aggregate(records),
            by_seed={str(seed):aggregate([r for r in records if r['seed']==seed]) for seed in sorted({r['seed'] for r in records})},
            by_reference_margin={name:aggregate([r for r in records if (r['reference_margin']<=0.05)==small])
                                 for name,small in [('at_most_0.05',True),('above_0.05',False)]},
            per_state=records)
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',action='append',required=True,help='LABEL=PATH; repeatable')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    candidates={}
    for spec in args.candidate:
        label,path=spec.split('=',1)
        if not label or label in candidates:
            parser.error('Candidate labels must be nonempty and unique')
        candidates[label]=Path(path).resolve()
    run(args.reference.resolve(),candidates,args.output.resolve())
