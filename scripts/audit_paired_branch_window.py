"""Audit completed paired GPU traces and apply the unchanged finite-window rule."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from paired_branch_capture import specs,load_start,validate_snapshot,sha,identical
from restore_own_death_contexts import load_verified_chunk
from restore_rank_contexts import observation_digest
from audit_topk_preference_evidence import preference,validate_branch
from training.candidate_features import candidate_features
from training.env.tetris_env import TetrisEnv
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def run(folder):
    design=json.loads((folder/'window_design.json').read_text());frozen=design['frozen_files']
    if (folder/'window_result.json').exists():raise FileExistsError('Paired window audit already closed')
    def check():
        for path,want in frozen.items():
            if sha(path)!=want:raise ValueError('Frozen paired window input changed: '+path)
    check();job=Path(design['job']);jd=json.loads((job/'design.json').read_text());ordered=specs(jd['cases'])
    lock=(job/'capture.lock').open('r');fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
    try:
        identity=json.loads((job/'chunk_0000/manifest.json').read_text())['identity']
        if identity['design_sha256']!=sha(job/'design.json'):raise ValueError('Window job identity differs')
        traces=[[] for _ in range(8)];previous=None;last_sha=None;steps=0
        chunks=sorted(job.glob('chunk_[0-9][0-9][0-9][0-9]'))
        for index,directory in enumerate(chunks):
            saved,arrays,manifest=load_verified_chunk(directory,frozen[str(directory/'manifest.json')],identity)
            validate_snapshot(saved,jd)
            if (manifest['number']!=index or manifest['previous_manifest_sha256']!=last_sha or
                    manifest['audit'].get('policy_adam_counts_rng_unchanged') is not True or
                    manifest['audit']['model_sha256']!=jd['model']['sha256'] or
                    manifest['audit']['model_counts']!=[30003200,36625]):
                raise ValueError('Window chain, original model identity or audit differs')
            if index==0:
                if saved['round']!=0 or manifest['rows']!=0:raise ValueError('Initial window commit differs')
                for lane,spec in enumerate(ordered):
                    if not identical(saved['workers'][lane]['state'],load_start(spec['case'],jd)):
                        raise ValueError('Initial complete source/core/RNG differs')
            else:
                active=[w['active'] for w in previous['workers']];cursor=0
                for decision_round in range(previous['round'],saved['round']):
                    lanes=[i for i,a in enumerate(active) if a];stop=cursor+len(lanes)
                    if (not lanes or list(map(int,arrays['lane'][cursor:stop]))!=lanes or
                            not (arrays['round'][cursor:stop]==decision_round).all()):
                        raise ValueError('Trace ordered active lanes or decision round differs')
                    for offset,lane in enumerate(lanes):
                        row={k:v[cursor+offset].item() if v[cursor+offset].ndim==0 else v[cursor+offset]
                             for k,v in arrays.items()};spec=ordered[lane]
                        if (row['branch_id']!=spec['branch_id'] or row['episode_seed']!=spec['case']['seed'] or
                                row['relative_step']!=len(traces[lane])+1 or row['relative_step']!=decision_round+1 or
                                row['pieces']!=spec['case']['step']+row['relative_step'] or
                                row['forced_first']!=(decision_round==0) or row['truncated'] or
                                not 0<=row['cleared_lines']<=4):
                            raise ValueError('Trace branch/seed/piece/terminal or intervention differs')
                        expected=spec['first_action'] if decision_round==0 else row['natural_action']
                        if row['action']!=expected or (decision_round==0 and row['natural_action']!=spec['case']['baseline_action']):
                            raise ValueError('Fixed first action or frozen-policy continuation differs')
                        if decision_round==previous['round'] and dict(observation=row['observation_sha256'],mask=row['mask_sha256'])!=previous['views'][lane]:
                            raise ValueError('Trace first input differs from complete committed checkpoint')
                        traces[lane].append(row)
                        if row['terminated'] or row['relative_step']==5000:active[lane]=False
                    cursor=stop
                if cursor!=manifest['rows']:raise ValueError('Window trace not fully consumed')
                steps+=cursor
            for lane,worker in enumerate(saved['workers']):
                if worker['steps']!=len(traces[lane]):raise ValueError('Worker steps differ from trace')
                if traces[lane] and worker['state']['core'].score!=traces[lane][-1]['score']:
                    raise ValueError('Complete core final score differs')
                if worker['active']:
                    public=worker['state']['core'].get_public_observation();matrix,mask=candidate_features(public)
                    view=dict(observation=dict(state=TetrisEnv._encode_observation(public),candidate=matrix),mask=mask)
                    if observation_digest(**view)!=saved['views'][lane]:raise ValueError('Checkpoint public input differs')
            if steps!=saved['environment_steps']:raise ValueError('Whole-job unique work differs')
            previous=saved;last_sha=frozen[str(directory/'manifest.json')]
        if previous is None or any(w['active'] for w in previous['workers']):
            raise ValueError('Paired job still partial; do not infer5000-window labels')
        results=[]
        for lane,spec in enumerate(ordered):
            trace=traces[lane];initial=load_start(spec['case'],jd)['core'].score
            branch=dict(branch_id=spec['branch_id'],role=spec['role'],seed=spec['case']['seed'],
                first_action=spec['first_action'],model_sha256=jd['model']['sha256'],
                pieces=len(trace),initial_score=initial,scores=[r['score'] for r in trace],
                four_counts=np.cumsum([r['cleared_lines']==4 for r in trace]).tolist(),
                game_over=trace[-1]['terminated'],censored=not trace[-1]['terminated'],
                score_gain=trace[-1]['score']-initial)
            validate_branch(branch,5000);results.append(branch)
        historical=[]
        for spec,result in zip(ordered,results):
            path=Path(jd['source_root'])/'counterfactual_5000'/f"seed_{spec['case']['seed']}"/f"step_{spec['case']['step']}_action_{spec['first_action']}.json"
            seal_path=path.with_suffix('.sha256.json')
            if str(path) not in frozen or str(seal_path) not in frozen:
                raise ValueError('Historical outcome lacks an external seal')
            seal=json.loads(seal_path.read_text());old=json.loads(path.read_text())
            if (sha(path)!=seal['sha256'] or seal['snapshot_sha256']!=spec['case']['snapshot_sha256'] or
                    old['first_action']!=spec['first_action'] or old['horizon']!=5000):
                raise ValueError('Historical branch outcome identity differs')
            validate_branch(old,5000)
            keys=['pieces','game_over','censored','initial_score','score_gain','scores','four_counts']
            historical.append(dict(branch_id=spec['branch_id'],recorded_outcome_equal=all(old[k]==result[k] for k in keys),
                                   historical_model_identity_explicit='model_sha256' in old))
        comparisons=[]
        for pair in range(4):
            base,alternative=results[pair*2:pair*2+2];label,comparison=preference(base,alternative)
            comparisons.append(dict(case=jd['cases'][pair],label=label,result=comparison,
                base_pieces=base['pieces'],alternative_pieces=alternative['pieces'],
                base_censored=base['censored'],alternative_censored=alternative['censored']))
        receipts=[json.loads(Path(p).read_text()) for p in design['phase_receipts']]
        if (any(r['status']!='completed' or r['source_audit']['policy_adam_counts_rng_unchanged'] is not True for r in receipts) or
                sum(r['work']['environment_steps'] for r in receipts)!=steps or
                sum(r['work']['gpu_forward_calls'] for r in receipts)!=previous['round']):
            raise ValueError('Actual phase work/source state differs from complete trace')
        check()
        durable_json(folder/'branches.json',dict(branches=results))
        durable_json(folder/'window_result.json',dict(at=now(),status='passed',model=jd['model'],
            comparisons=comparisons,label_counts={label:sum(p['label']==label for p in comparisons)
                for label in ['beneficial','harmful','unsupported_within_window','both_right_censored_uncertain']},
            source_environment_steps=steps,source_gpu_forward_calls=previous['round'],optimizer_steps=0,
            source_pids=[r['pid'] for r in receipts],new_environment_steps=0,gpu_forward_calls=0,
            all_terminal=True,per_branch_model_identity_explicit=True,
            historical_recorded_outcome_comparison=historical,
            branches_sha256=sha(folder/'branches.json'),eligible_for_training=False,
            scope='Four predetermined training seeds from previously selected positive CPU cases. New matched-state current CUDA branch evidence, dependent/conditional exploratory sample; not historical runtime reproduction, unbiased frequency or held-out quality.'))
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    run(p.parse_args().run_dir.resolve())
