"""Verify semantic state and measured timing of paired GPU feature backends."""

import argparse
import hashlib
import shutil
import json
import math
from pathlib import Path
import pickle
import random
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def identical(a,b):
    import numpy as np
    import torch

    if type(a) is not type(b):return False
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,random.Random):return a.getstate()==b.getstate()
    if isinstance(a,dict):return a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and math.isnan(a):return math.isnan(b)
    if hasattr(a,'__dict__'):return identical(a.__dict__,b.__dict__)
    return bool(a==b)


def main():
    import torch
    from training.kl_maskable_ppo import KLMaskablePPO
    from training.train_vector_transaction import verify_task,verify_training_source_identity
    from training.train_ppo import atomic_json,now
    from training.native_feature_runtime import verify_runtime_identity

    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    folder=parser.parse_args().run_dir.resolve();design=json.loads((folder/'gpu_design.json').read_text());pairs=[]
    timing_excluded={'wall_seconds','gpu_peak_allocated_mib'}
    backends=design.get('comparison_backends', {
        'official':{'candidate_feature_backend':'official'},
        'bitboard':{'candidate_feature_backend':'bitboard'}})
    assert set(backends)=={'official','bitboard'}
    for repeat,seed in enumerate(design['pair_seeds']):
        groups={};quotas_equal=True
        for backend in ('official','bitboard'):
            label=f'r{repeat}_{backend}';training=folder/'gpu_pairs'/label/'training'
            config=json.loads((training/'config.json').read_text());state=json.loads((training/'transaction_state.json').read_text())
            assert config['device']=='cuda' and config['ppo']['n_envs']==8
            assert all(config.get(key,'formal' if key=='candidate_search_backend' else 'official')==value
                       for key,value in backends[backend].items())
            assert config['run_seed']==seed and state['committed_steps']==8192 and state['pid'] is None
            metadata=json.loads((training/'metadata.json').read_text())
            verify_training_source_identity(metadata)
            if design.get('verify_native_runtime_resume_rejection'):
                assert verify_runtime_identity(config,metadata)==design['native_feature_runtime_identity']
            tasks=[]
            for task in (1,2):
                checkpoint=training/'committed'/f'task_{task:06d}';verify_task(checkpoint,task,config)
                model=KLMaskablePPO.load(str(checkpoint/'model.zip'),device='cpu')
                metrics=json.loads((training/f'metric_history/task_{task:06d}.json').read_text())
                assert model.num_timesteps==task*4096 and model.seed==seed and metrics['gpu_peak_allocated_mib']>0
                workers=pickle.loads((checkpoint/'vector_env_state.pkl').read_bytes());assert len(workers)==8
                tasks.append(dict(ppo_updates=model._n_updates,policy={k:v.clone() for k,v in model.policy.state_dict().items()},
                    optimizer=model.policy.optimizer.state_dict(),trainer=pickle.loads((checkpoint/'trainer_state.pkl').read_bytes()),
                    workers=workers,metrics=metrics))
                guards=list((folder/'gpu_guards').glob(f'{label}_task_{task:02d}*'))
                logs=[p for p in guards if p.is_dir() and (p/'result.json').exists()
                      and json.loads((p/'result.json').read_text())['outcome']=='completed']
                assert logs
                samples=[json.loads(line) for line in logs[-1].joinpath('samples.jsonl').read_text().splitlines()]
                quotas_equal=quotas_equal and all(s['cpu_quota_percent']==250 and s['decision']!='reduce' for s in samples)
                del model
            groups[backend]=tasks
        for task,(left,right) in enumerate(zip(groups['official'],groups['bitboard']),1):
            for field in ('policy','optimizer','trainer','workers','ppo_updates'):
                assert identical(left[field],right[field]),f'Pair{repeat} task{task} {field} differs'
            clean=lambda m:{k:v for k,v in m.items() if k not in timing_excluded}
            assert identical(clean(left['metrics']),clean(right['metrics'])),f'Pair{repeat} task{task} numerical metrics differ'
        times={name:sum(t['metrics']['wall_seconds'] for t in values) for name,values in groups.items()}
        pairs.append(dict(seed=seed,all_policy_optimizer_rng_worker_states_and_non_timing_metrics_exact=True,
                          official_training_seconds=times['official'],bitboard_training_seconds=times['bitboard'],
                          speedup=times['official']/times['bitboard'],quota250_throughout=quotas_equal))
    speedup=statistics.geometric_mean(p['speedup'] for p in pairs)
    eligible=all(p['quota250_throughout'] and p['speedup']>1 for p in pairs) and speedup>=design['minimum_speedup']
    resume_rejection=None
    mismatch_field = None
    if design.get('verify_search_resume_rejection') or design.get('verify_feature_resume_rejection'):
        kind = 'feature' if design.get('verify_feature_resume_rejection') else 'search'
        mismatch_field = f'candidate_{kind}_backend'
        case=folder/'gpu_pairs/r0_official';training=case/'training'
        def tree_hashes(directory):
            return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in directory.rglob('*') if p.is_file()}
        before=tree_hashes(training)
        preset=json.loads((case/'preset.json').read_text())
        original_backend = preset.get(mismatch_field, 'formal' if kind == 'search' else 'official')
        preset[mismatch_field] = backends['bitboard'][mismatch_field]
        assert preset[mismatch_field] != original_backend, 'Resume rejection must change the selected backend'
        wrong=folder/'mismatched_resume_preset.json';atomic_json(wrong,preset)
        attempt=subprocess.run([sys.executable,'-m','training.train_vector_transaction',
            '--run-dir',str(training),'--config-file',str(wrong),'--resume'],
            cwd=ROOT,capture_output=True,text=True,timeout=60)
        assert attempt.returncode!=0 and f'Resume Candidate {kind} backend differs' in attempt.stderr
        assert tree_hashes(training)==before
        resume_rejection=dict(rejected=True,field=mismatch_field,committed_state_config_metadata_unchanged=True,entire_run_file_hashes_unchanged=True)
        atomic_json(folder/'resume_rejection.json',resume_rejection)
    native_rejection=None
    if design.get('verify_native_runtime_resume_rejection'):
        case=folder/'gpu_pairs/r0_bitboard';training=case/'training'
        before=tree_hashes(training)
        preset=json.loads((case/'preset.json').read_text())
        preset['native_feature_runtime']['sha256']='0'*64
        wrong=folder/'wrong_native_runtime_preset.json';atomic_json(wrong,preset)
        attempt=subprocess.run([sys.executable,'-m','training.train_vector_transaction',
            '--run-dir',str(training),'--config-file',str(wrong),'--resume'],
            cwd=ROOT,capture_output=True,text=True,timeout=60)
        assert attempt.returncode!=0 and 'Resume native feature runtime differs' in attempt.stderr
        assert tree_hashes(training)==before
        # Corrupt a copied artifact and cloned checkpoint, preserving all real
        # benchmark files and the frozen compiled binary. No native code runs.
        probe=folder/'corrupt_runtime_probe';probe.mkdir()
        artifact=probe/'artifact';artifact.mkdir()
        identity=design['native_feature_runtime_identity']['artifact'].copy()
        binary=artifact/'candidate_features.so';binary.write_bytes(b'corrupted')
        identity['binary']=str(binary)
        manifest=artifact/'manifest.json';atomic_json(manifest,identity)
        clone=probe/'training';shutil.copytree(training,clone)
        config=json.loads((clone/'config.json').read_text())
        config['native_feature_runtime']=dict(manifest=str(manifest),sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        atomic_json(clone/'config.json',config)
        clone_before=tree_hashes(clone)
        attempt=subprocess.run([sys.executable,'-m','training.train_vector_transaction',
            '--run-dir',str(clone),'--resume'],cwd=ROOT,capture_output=True,text=True,timeout=60)
        assert attempt.returncode!=0 and 'Native feature runtime identity mismatch' in attempt.stderr
        assert tree_hashes(clone)==clone_before and tree_hashes(training)==before
        native_rejection=dict(requested_runtime_change_rejected=True,corrupted_binary_on_cloned_checkpoint_rejected=True,
            entire_real_and_probe_run_file_hashes_unchanged=True,probe_does_not_modify_frozen_artifact=True)
        atomic_json(folder/'native_runtime_resume_rejection.json',native_rejection)
    result=dict(at=now(),status='passed',pairs=pairs,geometric_mean_training_speedup=speedup,
        eligible_for_new_experiments=eligible,minimum_speedup=design['minimum_speedup'],device='cuda',n_envs=8,
        comparison_backends=backends,native_runtime_resume_rejection=native_rejection,
        search_resume_rejection=resume_rejection if mismatch_field == 'candidate_search_backend' else None,
        feature_resume_rejection=resume_rejection if mismatch_field == 'candidate_feature_backend' else None,
        interpretation='Two paired8192-step CUDA8 runs per backend; local throughput evidence, not a universal speedup or policy improvement',
        metrics_timing_fields_excluded=sorted(timing_excluded),model_promotion='none',final_test_used=False)
    atomic_json(folder/'gpu_summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
