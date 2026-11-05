"""Compare matched development results without mixing caps, seeds or evaluator identities."""

import argparse
import json
import math
from pathlib import Path
import statistics


def rows_by_seed(result):
    seeds=result['seeds']
    rows=result['per_seed']
    if len(seeds)!=result['num_seeds'] or len(set(seeds))!=len(seeds):
        raise ValueError('Invalid declared seed list')
    if len(rows)!=len(seeds) or {r['seed'] for r in rows}!=set(seeds):
        raise ValueError('Per-seed rows differ from declared seeds')
    for row in rows:
        pieces=row['pieces_survived']
        if not isinstance(pieces,int) or isinstance(pieces,bool) or not 0<=pieces<=result['max_pieces']:
            raise ValueError('Invalid survival count')
        if not math.isfinite(row['score']) or row['score']<0:
            raise ValueError('Invalid score')
    return {r['seed']:r for r in rows}


def compare(baseline,candidate):
    if not baseline.get('model_sha256') or not candidate.get('model_sha256'):
        return dict(comparable=False,reason='Missing model identity')
    if (not baseline.get('evaluation_source_sha256') or
            baseline['evaluation_source_sha256']!=candidate.get('evaluation_source_sha256')):
        return dict(comparable=False,reason='Missing or different evaluator identity')
    for key in ('runtime_versions','python_version','diagnostics'):
        if baseline.get(key) != candidate.get(key):
            return dict(comparable=False,reason='Different evaluator runtime or diagnostics: ' + key)
    if baseline['max_pieces']!=candidate['max_pieces']:
        return dict(comparable=False,reason='Different observation caps')
    try:
        left,right=rows_by_seed(baseline),rows_by_seed(candidate)
    except (ValueError,KeyError,TypeError) as exc:
        return dict(comparable=False,reason=str(exc))
    if set(left)!=set(right):
        return dict(comparable=False,reason='Different seeds')
    pairs=[]
    for seed in baseline['seeds']:
        a,b=left[seed],right[seed]
        pairs.append(dict(seed=seed,baseline_pieces=a['pieces_survived'],candidate_pieces=b['pieces_survived'],
                          baseline_score=a['score'],candidate_score=b['score'],
                          pieces_delta=b['pieces_survived']-a['pieces_survived'],score_delta=b['score']-a['score']))
    mean=lambda side,field:statistics.mean(p[side+'_'+field] for p in pairs)
    def ratio(field):
        denominator=mean('baseline',field)
        return mean('candidate',field)/denominator if denominator else None
    pieces_ratio,score_ratio=ratio('pieces'),ratio('score')
    censored=any(row['survived_cap'] for row in [*left.values(),*right.values()])
    return dict(comparable=True,committed_steps=candidate['committed_steps'],max_pieces=baseline['max_pieces'],
                num_seeds=len(pairs),mean_pieces_ratio=pieces_ratio,mean_observed_score_ratio=score_ratio,
                survival_wins=sum(p['pieces_delta']>0 for p in pairs),
                survival_losses=sum(p['pieces_delta']<0 for p in pairs),
                score_wins=sum(p['score_delta']>0 for p in pairs),
                score_losses=sum(p['score_delta']<0 for p in pairs),
                survival_is_right_censored=censored,
                review_recommended=any(r is not None and r<0.9 for r in [pieces_ratio,score_ratio]),
                interpretation='Exploratory paired development comparison; no promotion or automatic stop',
                per_seed=pairs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=compare(json.loads(args.baseline.read_text()),json.loads(args.candidate.read_text()))
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':
    main()
