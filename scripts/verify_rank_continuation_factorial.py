"""Compare continuation policies while holding each saved first action fixed."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

from restore_rank_contexts import sha
from rank_branch_helpers import compare_pair
from run_rank_branches import aggregate
from training.evaluation.evaluate_resumable import durable_json


def compatible_protocols(reference,candidate,models):
    fields=['contexts_dir','training_sources','inference_sources','evaluation_runtime','native_feature_runtime',
            'horizon','n_envs','context_count','first_clear_expectations','grouping']
    if any(reference[k]!=candidate[k] for k in fields):raise ValueError('Continuation protocols differ')
    if (reference['model']!=models['reference'] or candidate['model']!=models['candidate'] or
            reference['model']==candidate['model']):raise ValueError('Continuation model identities differ')
    if (reference['horizon']!=256 or reference['n_envs']!=8 or reference['context_count']!=15 or
            reference['first_clear_expectations']!=[4,0]):raise ValueError('Fixed factorial scope differs')
    for relative in ['scripts/run_rank_branches.py','scripts/rank_branch_helpers.py']:
        path=str(ROOT/relative)
        if reference['frozen_files'][path]!=candidate['frozen_files'][path]:
            raise ValueError('Continuation execution source differs')


def validate_pair(manifest,data,group,keys,target,context_sha,model,counts,horizon):
    identity=manifest['identity']
    if (identity['group']!=group or identity['group_keys']!=keys or identity['target']!=target or
            identity['context_sha256']!=context_sha or identity['model']!=model or
            identity['torch_float32_matmul_precision']!='highest' or identity['tf32'] is not False):
        raise ValueError('Factorial pair context/group/model identity differs')
    audit=data['model_audit']
    if (audit['policy_adam_counts_unchanged'] is not True or audit['cpu_cuda_sampling_rng_unchanged'] is not True or
            [audit['num_timesteps'],audit['ppo_updates']]!=counts):
        raise ValueError('Factorial model count or immutability audit differs')
    for name,action,clears in [('reference',target['reference_action'],4),('candidate',target['candidate_action'],0)]:
        outcome=data[name];summary=outcome['summary'];trace=outcome['trace'];steps=summary['steps']
        if (type(steps) is not int or not 1<=steps<=horizon or len(trace)!=steps or
                trace[0]['action']!=action or trace[0]['cleared_lines']!=clears or
                summary['cleared_lines']!=sum(r['cleared_lines'] for r in trace) or
                not all(math.isfinite(r['reward']) for r in trace) or
                not math.isclose(summary['reward'],sum(r['reward'] for r in trace),abs_tol=1e-9)):
            raise ValueError('Factorial forced action, trace or outcome differs')
        flags=[summary[n] for n in ['game_over','time_limit','horizon_censored']]
        if any(type(f) is not bool for f in flags) or sum(flags)!=1:
            raise ValueError('Invalid factorial terminal/censoring flags')
        if (trace[-1]['terminated']!=summary['game_over'] or trace[-1]['truncated']!=summary['time_limit'] or
                summary['horizon_censored']!=(steps==horizon and not summary['game_over'] and not summary['time_limit']) or
                any(r['terminated'] or r['truncated'] for r in trace[:-1]) or
                any(r['pieces']!=trace[0]['pieces']+i for i,r in enumerate(trace))):
            raise ValueError('Factorial branch reset or terminal marker differs')


def contrasts(reference,candidate):
    # Outer names hold FIRST ACTION fixed; inner reference/candidate labels
    # now refer to the CONTINUATION model, not the selected initial action.
    result={}
    for first in ['reference','candidate']:
        pair=dict(reference=reference[first],candidate=candidate[first])
        pair['comparison']=compare_pair(pair)
        result[first+'_first_action']=pair
    return result


def run(folder):
    from resource_guard import now
    if (folder/'factorial_summary.json').exists():raise FileExistsError('Preserve completed factorial comparison')
    design=json.loads((folder/'design.json').read_text());baseline=Path(design['baseline_dir'])
    old=json.loads((baseline/'design.json').read_text())
    compatible_protocols(old,design,design['expected_continuation_models'])
    for name,digest in design['frozen_files'].items():
        if sha(name)!=digest:raise ValueError('Frozen factorial input changed: '+name)
    old_summary=json.loads((baseline/'branch_summary.json').read_text())
    new_summary=json.loads((folder/'branch_summary.json').read_text())
    if old_summary['status']!='passed' or new_summary['status']!='passed':raise ValueError('Branch runs not complete')
    targets=json.loads((Path(design['contexts_dir'])/'contexts_summary.json').read_text())['targets']
    rows=[];groups={name:[] for name in ['reference_first_action','candidate_first_action']}
    for index,target in enumerate(targets):
        group=index//4;keys=[t['key'] for t in targets[group*4:(group+1)*4]]
        context=json.loads((Path(design['contexts_dir'])/'contexts'/target['key']/'manifest.json').read_text())
        results={}
        for label,directory,protocol in [('reference',baseline,old),('candidate',folder,design)]:
            pair_dir=directory/'pairs'/target['key'];manifest=json.loads((pair_dir/'manifest.json').read_text())
            receipt=json.loads((directory/f'group_{group:02d}.json').read_text())
            if (set(manifest['files'])!={'result.json','states.pkl'} or
                    any(sha(pair_dir/n)!=h for n,h in manifest['files'].items()) or
                    receipt['pair_manifests'][target['key']]!=sha(pair_dir/'manifest.json')):
                raise ValueError('Factorial pair output checksum differs')
            data=json.loads((pair_dir/'result.json').read_text())
            validate_pair(manifest,data,group,keys,target,context['state_sha256'],protocol['model'],
                design['expected_model_counts'][label],design['horizon'])
            results[label]=data
        comparison=contrasts(results['reference'],results['candidate'])
        for name,pair in comparison.items():groups[name].append(pair)
        rows.append(dict(target=target,comparisons={name:pair['comparison'] for name,pair in comparison.items()},
            endpoints={label:{first:data[first]['summary'] for first in ['reference','candidate']}
                       for label,data in results.items()}))
    output=dict(at=now(),status='passed',model_promotion='none',horizon=256,contexts=15,
        continuation_models=design['expected_continuation_models'],
        reference_run=str(baseline),candidate_run=str(folder),
        aggregate={name:aggregate(pairs) for name,pairs in groups.items()},per_context=rows,
        interpretation='Hold each initial action fixed: reference_first_action compares A/C, candidate_first_action compares B/D. '
            'Differences mean candidate-continuation minus reference-continuation over common steps. '
            'Selected dependent15 contexts and finite censored horizon; no whole-policy quality conclusion.')
    durable_json(folder/'factorial_summary.json',output)
    print(json.dumps({k:output[k] for k in ['status','horizon','contexts','aggregate']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--execute-branches',action='store_true');args=parser.parse_args()
    folder=args.run_dir.resolve()
    if args.execute_branches:
        subprocess.run([sys.executable,str(ROOT/'scripts/run_rank_branches.py'),'--run-dir',str(folder),
                        '--verify-restart'],check=True)
    run(folder)
