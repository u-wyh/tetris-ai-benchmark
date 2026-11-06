"""Fixed groups of two contexts, all eight prior disagreements, 512 placements."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from own_policy_capture import sha
from own_death_factorial_helpers import run_factorial,SourceReproductionError,ARMS
from run_own_death_factorial import source_traces,comparisons,save_group,read_group
from rank_branch_helpers import BranchNativeEnv
from restore_rank_contexts import load_context
from verify_feature_backend_benchmark import identical
from training.evaluation.evaluate_resumable import durable_json


def run(folder,max_new_groups):
    import torch
    from sb3_contrib import MaskablePPO
    from training.train_transaction import global_rng_state
    from training.train_vector_transaction import training_source_identity
    from training.vector_env import make_vector_env
    from training.evaluation.feature_backend import inference_source_identity,evaluation_runtime_identity
    from training.candidate_features import FEATURE_NAMES
    from resource_guard import now
    design=json.loads((folder/'design.json').read_text())
    def unchanged():
        for path,expected in design['frozen_files'].items():
            if sha(path)!=expected:raise ValueError('Frozen prior factorial input changed: '+path)
    unchanged()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no prior factorial fallback')
    if (design['horizon']!=512 or design['n_envs']!=8 or not 1<=max_new_groups<=4 or
            training_source_identity()!=design['training_sources'] or
            inference_source_identity(resumable=True)!=design['inference_sources'] or
            evaluation_runtime_identity(False)!=design['evaluation_runtime']):
        raise ValueError('Prior factorial scope or numerical source/runtime differs')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    context_dir=Path(design['contexts_dir']);summary=json.loads((context_dir/'contexts_summary.json').read_text())
    targets=summary['targets']
    if ([target['step'] for target in targets]!=design['planned_steps'] or len(targets)!=8 or
            any(target['seed']!=100005 or target['actions_to_observed_death_including_selected']>512 for target in targets)):
        raise ValueError('Every predeclared prior target must be retained within the fixed horizon')
    identity=dict(design_sha256=sha(folder/'design.json'),horizon=512,models=design['models'],
        grouping='Fixed chronological pairs of contexts; each A/B/C/D; ordered active lanes per policy',
        precision='highest',tf32=False)
    lock=(folder/'factorials.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=None;models=None;computed=[];reused=[];steps=0;forward=dict(reference=0,candidate=0)
    try:
        for group in range(4):
            chosen=targets[group*2:group*2+2]
            group_identity=dict(identity,group=group,group_keys=[target['key'] for target in chosen])
            directory=folder/f'group_{group:02d}'
            cached=read_group(directory,group_identity)
            if cached is not None:
                if not cached[0]['actual_work']['source_D_reproduction_passed']:raise ValueError('Cached source D failed')
                reused.append(group);continue
            if len(computed)>=max_new_groups:break
            if models is None:
                models={}
                for name,spec in design['models'].items():
                    if sha(spec['path'])!=spec['sha256']:raise ValueError('Prior frozen model changed')
                    model=MaskablePPO.load(spec['path'],device='cuda');model.policy.set_training_mode(False)
                    if (model.get_env() is not None or model.device.type!='cuda' or
                            [model.num_timesteps,model._n_updates]!=spec['counts']):
                        raise ValueError('Prior model counters or device differ')
                    models[name]=model
                frozen={name:(copy.deepcopy(model.policy.state_dict()),copy.deepcopy(model.policy.optimizer.state_dict()),
                    model.num_timesteps,model._n_updates) for name,model in models.items()}
                env=make_vector_env(8,max_pieces=20000,env_class=BranchNativeEnv,
                    env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
                rng=global_rng_state(models['candidate'])
            def model_audit():
                for name,model in models.items():
                    weights,adam,count,updates=frozen[name]
                    if (not identical(weights,model.policy.state_dict()) or
                            not identical(adam,model.policy.optimizer.state_dict()) or
                            (count,updates)!=(model.num_timesteps,model._n_updates)):
                        raise ValueError('Prior factorial changed model, Adam or counts')
                if not identical(rng,global_rng_state(models['candidate'])):raise ValueError('Prior factorial changed sampling RNG')
                return dict(policy_adam_counts_unchanged=True,global_sampling_rng_unchanged=True,
                    model_counts={name:[model.num_timesteps,model._n_updates] for name,model in models.items()})
            specifications=[];clear_index=FEATURE_NAMES.index('cleared_lines')
            for target in chosen:
                state,meta=load_context(context_dir/'contexts'/target['key'],dict(summary['identity'],target=target))
                if state['max_pieces']!=20000 or meta['observation_digest']!=target['observation_digest']:
                    raise ValueError('Prior factorial pre-action context differs')
                specifications.append(dict(key=target['key'],state=state,digest=meta['observation_digest'],
                    actions=[target['shadow_action'],target['candidate_action']],
                    expected_first_clears=[target['shadow_features'][clear_index],target['candidate_features'][clear_index]]))
            selection=json.loads(Path(design['targets_path']).read_text())
            traces=source_traces(Path(selection['capture_dir']),chosen,selection['capture_identity'],design['frozen_files'])
            try:outcomes,work=run_factorial(env,specifications,models,512,traces)
            except SourceReproductionError as error:
                durable_json(folder/f'source_reproduction_failure_group_{group:02d}.json',dict(at=now(),status='failed',
                    identity=group_identity,event=error.event,actual_work=error.work,model_audit=model_audit(),
                    interpretation='Source path not reproduced; keep completed groups, no causal claim for failed group.'))
                raise
            audit=model_audit();unchanged()
            if work['environment_steps']>4096 or not work['source_D_reproduction_passed']:raise ValueError('Prior group scope/gate failed')
            cases=[];states={}
            for target,outcome in zip(chosen,outcomes):
                cases.append(dict(target=target,arms={arm:dict(summary=outcome[arm]['summary'],trace=outcome[arm]['trace']) for arm in ARMS},
                    comparisons=comparisons(outcome)))
                states[target['key']]={arm:outcome[arm]['state'] for arm in ARMS}
            result=dict(identity=group_identity,cases=cases,actual_work=work,model_audit=audit,pid=os.getpid())
            save_group(directory,group_identity,result,states)
            computed.append(group);steps+=work['environment_steps']
            for name in forward:forward[name]+=work['policy_forward_calls'][name]
        process=dict(at=now(),pid=os.getpid(),computed_groups=computed,reused_groups=reused,
            environment_steps_this_process=steps,policy_forward_calls_this_process=forward,optimizer_steps=0)
        durable_json(folder/f'process_{os.getpid()}.json',process)
        if all((folder/f'group_{group:02d}').exists() for group in range(4)):
            cases=[];total=0;total_forward=dict(reference=0,candidate=0)
            for group in range(4):
                group_identity=dict(identity,group=group,group_keys=[target['key'] for target in targets[group*2:group*2+2]])
                result,_=read_group(folder/f'group_{group:02d}',group_identity)
                cases.extend(dict(target=case['target'],arms={arm:case['arms'][arm]['summary'] for arm in ARMS},
                    comparisons=case['comparisons']) for case in result['cases'])
                total+=result['actual_work']['environment_steps']
                for name in total_forward:total_forward[name]+=result['actual_work']['policy_forward_calls'][name]
            if len(cases)!=8 or total>16384:raise ValueError('All-eight prior factorial work budget differs')
            unchanged()
            durable_json(folder/'factorial_summary.json',dict(at=now(),status='passed',identity=identity,cases=cases,
                actual_new_environment_steps=total,policy_forward_calls=total_forward,optimizer_steps=0,process=process,
                source_D_all_eight_deaths_reproduced=True,model_promotion='none',
                interpretation='Eight correlated earlier points in one failed episode, all predeclared. Fixed512 placements. '
                    'Exact source reproduction; local interventions only, not8 independent seeds or a whole-policy quality gate.'))
        print(json.dumps(process))
    finally:
        if env is not None:env.close()
        lock.close()


def verify_restart(folder):
    from resource_guard import now
    if (folder/'restart_receipt.json').exists() or any(folder.glob('group_[0-9][0-9]')):
        raise RuntimeError('Prior groups exist; inspect and continue directly, do not repeat initial verifier')
    def execute(count):
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run-dir',str(folder),'--max-new-groups',str(count)])
        if child.wait():raise RuntimeError('Prior factorial child failed: '+str(child.pid))
        return child.pid
    first=execute(1);before={str(path):sha(path) for path in (folder/'group_00').iterdir() if path.is_file()}
    second=execute(4);process=json.loads((folder/f'process_{second}.json').read_text())
    if (first==second or process['reused_groups']!=[0] or process['computed_groups']!=[1,2,3] or
            any(sha(path)!=expected for path,expected in before.items())):
        raise ValueError('Prior factorial restart changed its first fixed group')
    durable_json(folder/'restart_receipt.json',dict(at=now(),status='passed',first_pid=first,second_pid=second,
        first_group_files_unchanged=True,protected_files=before,second_process=process))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--max-new-groups',type=int,choices=[1,2,3,4],default=4);parser.add_argument('--verify-restart',action='store_true')
    args=parser.parse_args()
    if args.verify_restart:verify_restart(args.run_dir.resolve())
    else:run(args.run_dir.resolve(),args.max_new_groups)
