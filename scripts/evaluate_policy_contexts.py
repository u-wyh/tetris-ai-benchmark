"""GPU8 natural-greedy diagnostic suite with 256-round durable commits, no training."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from policy_contexts import prepare_run,load_suite_states,activate_states,natural_round
from own_policy_capture import CaptureNativeEnv,save_commit,sha
from restore_own_death_contexts import restore_vector
from restore_rank_contexts import observation_digest
from verify_feature_backend_benchmark import identical
from training.evaluation.evaluate_resumable import durable_json


def run(folder,max_new_chunks):
    import torch
    from sb3_contrib import MaskablePPO
    from training.train_transaction import global_rng_state,restore_global_rng
    from training.train_vector_transaction import training_source_identity
    from training.vector_env import make_vector_env
    from training.evaluation.feature_backend import inference_source_identity,evaluation_runtime_identity
    from resource_guard import now
    design,suite,identity,snapshot,previous,number=prepare_run(folder)
    if not 1<=max_new_chunks<=2:raise ValueError('At most two bounded context evaluation chunks per process')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no context evaluation fallback')
    if (training_source_identity()!=design['evaluation_sources'] or
            inference_source_identity(resumable=True)!=design['inference_sources'] or
            evaluation_runtime_identity(False)!=design['evaluation_runtime']):
        raise ValueError('Context evaluation numerical sources/runtime differ')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    # Identity was checked before creating even this reversible lock file.
    lock=(folder/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=None
    try:
        # Revalidate after the lock: another completed writer must not be overwritten.
        design,suite,identity,snapshot,previous,number=prepare_run(folder)
        reused=number
        model=MaskablePPO.load(design['checkpoint']['path'],device='cuda')
        if model.get_env() is not None or model.device.type!='cuda':raise ValueError('Frozen checkpoint requires CUDA, no attached env')
        counts=[model.num_timesteps,model._n_updates]
        if design['checkpoint'].get('expected_counts',counts)!=counts:raise ValueError('Checkpoint source counters differ')
        model.policy.set_training_mode(False)
        weights=copy.deepcopy(model.policy.state_dict());adam=copy.deepcopy(model.policy.optimizer.state_dict())
        env=make_vector_env(8,max_pieces=20000,env_class=CaptureNativeEnv,
            env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
        if snapshot is None:
            states=load_suite_states(suite)
            views=activate_states(env,states,[entry['observation_digest'] for entry in suite['contexts']])
            initial=[dict(pieces=state['pieces'],score=state['core'].score,lines=state['core'].lines,
                          reward=state['episode_reward']) for state in states]
            snapshot=dict(round=0,environment_steps=0,initial_metrics=initial,local_rewards=[0.0]*8)
        else:
            restore_global_rng(snapshot['rng'])
            views=restore_vector(env,snapshot)
            if not identical(snapshot['rng'],global_rng_state(model)):
                raise ValueError('Restored evaluation model metadata or global sampling RNG differs')
        rng=global_rng_state(model)
        def audit():
            if (not identical(weights,model.policy.state_dict()) or not identical(adam,model.policy.optimizer.state_dict()) or
                    [model.num_timesteps,model._n_updates]!=counts or not identical(rng,global_rng_state(model))):
                raise ValueError('Evaluation changed policy, Adam, counters or global sampling RNG')
            return dict(policy_adam_counts_unchanged=True,global_sampling_rng_unchanged=True,
                        num_timesteps=counts[0],ppo_updates=counts[1],optimizer_steps=0)
        def unchanged():
            if sha(folder/'design.json')!=identity['design_sha256']:raise ValueError('Evaluation design changed')
            for path,expected in design['frozen_files'].items():
                if sha(path)!=expected:raise ValueError('Frozen evaluation input changed: '+path)
        def checkpoint():
            snapshot.update(workers=env.env_method('capture_export'),rng=global_rng_state(model),
                views=[None if view is None else observation_digest(**view) for view in views])
        if number==0:
            checkpoint();previous=save_commit(folder,0,identity,None,snapshot,[],audit());number=1
        computed=[];steps=forward=0
        for _ in range(max_new_chunks):
            if snapshot['round']>=512 or not any(view is not None for view in views):break
            rows=[]
            for _ in range(min(256,512-snapshot['round'])):
                if not any(view is not None for view in views):break
                rows.extend(natural_round(env,views,model.policy,snapshot['round']))
                snapshot['round']+=1;forward+=1
            snapshot['environment_steps']+=len(rows)
            for row in rows:snapshot['local_rewards'][row['lane']]+=row['reward']
            if not rows or snapshot['environment_steps']>4096:raise ValueError('Evaluation step budget or progress differs')
            checkpoint();metadata=audit();unchanged()
            previous=save_commit(folder,number,identity,previous,snapshot,rows,metadata)
            computed.append(number);number+=1;steps+=len(rows)
            durable_json(folder/'progress.json',dict(at=now(),pid=os.getpid(),identity=identity,
                last_manifest_sha256=previous,round=snapshot['round'],environment_steps=snapshot['environment_steps'],
                environment_active_lanes=[lane for lane,view in enumerate(views) if view is not None],
                evaluation_complete=snapshot['round']==512 or not any(view is not None for view in views)))
        checkpoint();metadata=audit();unchanged()
        process=dict(at=now(),pid=os.getpid(),reused_committed_chunks=reused,computed_chunks=computed,
            environment_steps_this_process=steps,policy_forward_calls_this_process=forward,optimizer_steps=0,model_audit=metadata)
        durable_json(folder/f'process_{os.getpid()}.json',process)
        if snapshot['round']==512 or not any(view is not None for view in views):
            endpoints=[]
            for lane,(entry,worker,initial) in enumerate(zip(suite['contexts'],snapshot['workers'],snapshot['initial_metrics'])):
                state=worker['state'];local_steps=state['pieces']-initial['pieces']
                death=state['core'].phase=='over';time_limit=state['finished'] and not death
                horizon=not state['finished'] and local_steps==512
                if (sum([death,time_limit,horizon])!=1 or state['episode_count']!=1 or
                        state['episode_seed']!=entry['episode_seed'] or not 1<=local_steps<=512):
                    raise ValueError('Diagnostic endpoint has invalid episode or censoring metadata')
                if abs((state['episode_reward']-initial['reward'])-snapshot['local_rewards'][lane])>1e-8:
                    raise ValueError('Local reward ledger disagrees with complete worker state')
                endpoints.append(dict(key=entry['key'],lane=lane,seed=state['episode_seed'],
                    initial_pieces=initial['pieces'],steps=local_steps,total_pieces=state['pieces'],
                    reward=snapshot['local_rewards'][lane],score=state['core'].score-initial['score'],
                    cleared_lines=state['core'].lines-initial['lines'],game_over=death,time_limit=time_limit,horizon_censored=horizon))
            if sum(endpoint['steps'] for endpoint in endpoints)!=snapshot['environment_steps']:
                raise ValueError('Full worker endpoint deltas disagree with committed work')
            durable_json(folder/'evaluation_summary.json',dict(at=now(),status='completed',diagnostic_only=True,
                identity=identity,checkpoint=design['checkpoint'],suite=design['suite'],endpoints=endpoints,
                actual_environment_steps=snapshot['environment_steps'],actual_policy_forward_calls=snapshot['round'],
                optimizer_steps=0,model_audit=metadata,last_manifest_sha256=previous,process=process,policy_promotion='none',
                interpretation='Natural greedy from8 frozen contexts,512 local steps; no forced action, global quality verdict or training. '
                    'Environment-active lanes at the evaluation horizon remain unstepped and are horizon-censored.'))
        print(json.dumps(process))
    finally:
        if env is not None:env.close()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--max-new-chunks',type=int,choices=[1,2],default=2)
    args=parser.parse_args();run(args.run_dir.resolve(),args.max_new_chunks)
