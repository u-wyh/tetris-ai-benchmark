"""Run the archived evaluator or verify backend cache isolation under a guard."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['reference','cache_isolation'])
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();folder=args.run_dir.resolve();design=json.loads((folder/'design.json').read_text())
    if args.mode=='reference':
        archive=Path(design['archive_root']);sys.path.insert(0,str(archive))
        from training.evaluation import evaluate as original
        from training.train_ppo import atomic_json
        assert Path(original.__file__).resolve()==archive/'training/evaluation/evaluate.py'
        result=original.evaluate_model(Path(design['model']),design['seeds'],design['cap'],
            'archived_official_reference',0,diagnostics=True,workers=2)
        result['archive_root']=str(archive)
        result['archived_evaluator_sha256']=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()
        atomic_json(folder/'legacy_reference.json',result)
    else:
        root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
        from training.evaluation import evaluate as original
        from training.evaluation.evaluate_resumable import evaluate
        from training.train_ppo import atomic_json,now
        case=folder/'r0_official.json';result=json.loads(case.read_text());cache=case.with_suffix('.seeds')
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.iterdir() if p.is_file()}
        def forbidden(*args,**kwargs):raise AssertionError('Cache mismatch must fail before loading a model')
        original._init_worker=forbidden
        try:
            evaluate(Path(design['model']),design['seeds'],design['cap'],result['protocol'],0,
                     folder/'unexpected_mixed_backend.json',cache,workers=2,diagnostics=True,
                     candidate_feature_backend='bitboard')
        except ValueError as exc:
            assert 'identity differs' in str(exc)
        else:raise AssertionError('Mixed backend cache was accepted')
        assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.iterdir() if p.is_file()}
        assert not (folder/'unexpected_mixed_backend.json').exists()
        atomic_json(folder/'cache_isolation.json',dict(at=now(),status='passed',no_worker_started=True,
                                                     existing_cache_unchanged=True))


if __name__=='__main__':main()
