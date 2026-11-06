"""Fixed first-action interventions, followed by one frozen CUDA greedy policy."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

from training.evaluation.evaluate_resumable import durable_json
from restore_rank_contexts import load_context,sha
from rank_branch_helpers import BranchNativeEnv,run_group,compare_pair


def read_pair(directory,identity):
    if not directory.exists():return None
    manifest=json.loads((directory/'manifest.json').read_text())
    if (manifest['identity']!=identity or set(manifest['files'])!={'result.json','states.pkl'} or
            any(sha(directory/n)!=h for n,h in manifest['files'].items())):
        raise ValueError('Branch pair identity or checksum differs')
    data=json.loads((directory/'result.json').read_text())
    # Own hash-verified cache only; frozen input contexts are checked separately.
    states=pickle.loads((directory/'states.pkl').read_bytes())
    return data,states


def save_pair(directory,identity,pair,audit):
    from verify_feature_backend_benchmark import identical
    data={name:dict(summary=pair[name]['summary'],trace=pair[name]['trace']) for name in ['reference','candidate']}
    data.update(comparison=compare_pair(pair),model_audit=audit)
    states={name:pair[name]['state'] for name in ['reference','candidate']}
    cached=read_pair(directory,identity)
    if cached is not None:
        if data!=cached[0] or not identical(states,cached[1]):
            raise ValueError('Fixed-group replay differs from committed pair')
        return False
    pending=directory.with_name(directory.name+'.pending')
    if pending.exists():
        abandoned=directory.parent/'abandoned';abandoned.mkdir(exist_ok=True)
        pending.rename(abandoned/(pending.name+'.'+str(os.getpid())))
    pending.mkdir(parents=True)
    durable_json(pending/'result.json',data)
    with (pending/'states.pkl').open('xb') as stream:
        pickle.dump(states,stream,protocol=5);stream.flush();os.fsync(stream.fileno())
    durable_json(pending/'manifest.json',dict(identity=identity,
        files={n:sha(pending/n) for n in ['result.json','states.pkl']}))
    pending.rename(directory)
    fd=os.open(directory.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return True


def aggregate(pairs):
    comparisons=[p['comparison'] for p in pairs]
    return dict(pairs=len(pairs),common_reward_delta_sum=sum(p['common_reward_delta'] for p in comparisons),
        common_cleared_delta_sum=sum(p['common_cleared_delta'] for p in comparisons),
        common_four_clear_delta_sum=sum(p['common_four_clear_delta'] for p in comparisons),
        candidate_observed_earlier_deaths=sum(p['candidate_observed_earlier_death'] for p in comparisons),
        reference_observed_earlier_deaths=sum(p['reference_observed_earlier_death'] for p in comparisons),
        candidate_common_clearing_lower=sum(p['common_cleared_delta']<0 for p in comparisons),
        candidate_common_clearing_equal=sum(p['common_cleared_delta']==0 for p in comparisons),
        candidate_common_clearing_higher=sum(p['common_cleared_delta']>0 for p in comparisons),
        endpoints={name:dict(game_overs=sum(p[name]['summary']['game_over'] for p in pairs),
            time_limits=sum(p[name]['summary']['time_limit'] for p in pairs),
            horizon_censored=sum(p[name]['summary']['horizon_censored'] for p in pairs))
            for name in ['reference','candidate']})


def run(folder,through_groups):
    import torch
    from sb3_contrib import MaskablePPO
    from training.train_vector_transaction import training_source_identity
    from training.vector_env import make_vector_env
    from training.evaluation.feature_backend import inference_source_identity,evaluation_runtime_identity
    from verify_feature_backend_benchmark import identical
    from resource_guard import now
    design=json.loads((folder/'design.json').read_text())
    def unchanged():
        for name,digest in design['frozen_files'].items():
            if sha(name)!=digest:raise ValueError('Frozen branch input changed: '+name)
    unchanged()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no branch fallback')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    if (training_source_identity()!=design['training_sources'] or
            inference_source_identity(resumable=True)!=design['inference_sources'] or
            evaluation_runtime_identity(False)!=design['evaluation_runtime']):
        raise ValueError('Branch inference/source runtime differs')
    prefix=Path(design['contexts_dir']);prefix_summary=json.loads((prefix/'contexts_summary.json').read_text())
    targets=prefix_summary['targets']
    if len(targets)!=15 or design['horizon']!=256:raise ValueError('Fixed branch scope differs')
    identity=dict(design_sha256=sha(folder/'design.json'),model=design['model'],
        grouping='fixed consecutive four pairs; active lanes ordered; partial groups replayed whole',
        torch_float32_matmul_precision='highest',tf32=False)
    lock=(folder/'branches.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=model=None;computed=[];reused=[];steps=forward_calls=0
    try:
        for group in range(through_groups):
            chosen=targets[group*4:(group+1)*4];keys=[t['key'] for t in chosen]
            group_identity=dict(identity,group=group,group_keys=keys)
            specs=[];pair_identities=[]
            for target in chosen:
                expected=dict(prefix_summary['identity'],target=target)
                state,metadata=load_context(prefix/'contexts'/target['key'],expected)
                specs.append(dict(state=state,digest=metadata['observation_digest'],
                    actions=[target['reference_action'],target['candidate_action']],expected_first_clears=[4,0]))
                pair_identities.append(dict(group_identity,target=target,context_sha256=metadata['state_sha256']))
            receipt=folder/f'group_{group:02d}.json'
            if receipt.exists():
                cached=json.loads(receipt.read_text())
                if cached['identity']!=group_identity or cached['status']!='passed':raise ValueError('Branch group cache differs')
                for target,pair_identity in zip(chosen,pair_identities):
                    directory=folder/'pairs'/target['key']
                    if sha(directory/'manifest.json')!=cached['pair_manifests'][target['key']]:raise ValueError('Committed pair changed')
                    if read_pair(directory,pair_identity) is None:raise ValueError('Committed pair missing')
                reused.append(group);continue
            if model is None:
                if sha(design['model']['path'])!=design['model']['sha256']:raise ValueError('Frozen model changed')
                model=MaskablePPO.load(design['model']['path'],device='cuda')
                assert model.get_env() is None and model.device.type=='cuda'
                model.policy.set_training_mode(False)
                weights=copy.deepcopy(model.policy.state_dict());adam=copy.deepcopy(model.policy.optimizer.state_dict())
                counts=(model.num_timesteps,model._n_updates)
                rng=(torch.get_rng_state().clone(),[v.clone() for v in torch.cuda.get_rng_state_all()])
                env=make_vector_env(8,max_pieces=10000,env_class=BranchNativeEnv,
                    env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
            outcomes,work=run_group(env,specs,model,design['horizon'])
            assert identical(weights,model.policy.state_dict()) and identical(adam,model.policy.optimizer.state_dict())
            assert counts==(model.num_timesteps,model._n_updates)
            assert torch.equal(rng[0],torch.get_rng_state()) and identical(rng[1],torch.cuda.get_rng_state_all())
            audit=dict(policy_adam_counts_unchanged=True,cpu_cuda_sampling_rng_unchanged=True,
                       num_timesteps=counts[0],ppo_updates=counts[1])
            for target,pair_identity,pair in zip(chosen,pair_identities,outcomes):
                save_pair(folder/'pairs'/target['key'],pair_identity,pair,audit)
            unchanged()
            durable_json(receipt,dict(at=now(),status='passed',identity=group_identity,pid=os.getpid(),work=work,
                pair_manifests={t['key']:sha(folder/'pairs'/t['key']/'manifest.json') for t in chosen}))
            steps+=work['environment_steps'];forward_calls+=work['policy_forward_calls'];computed.append(group)
        process=dict(at=now(),pid=os.getpid(),computed_groups=computed,reused_groups=reused,
            environment_steps_this_process=steps,policy_forward_calls_this_process=forward_calls,optimizer_steps=0)
        durable_json(folder/f'process_{os.getpid()}.json',process)
        if through_groups==4:
            all_pairs=[];total_steps=total_forward=0
            for group in range(4):
                chosen=targets[group*4:(group+1)*4]
                receipt=json.loads((folder/f'group_{group:02d}.json').read_text())
                total_steps+=receipt['work']['environment_steps'];total_forward+=receipt['work']['policy_forward_calls']
                for target in chosen:
                    all_pairs.append(json.loads((folder/'pairs'/target['key']/'result.json').read_text()))
            assert len(all_pairs)==15 and total_steps<=7680
            unchanged()
            durable_json(folder/'branch_summary.json',dict(at=now(),status='passed',identity=identity,
                aggregate=aggregate(all_pairs),planned_environment_step_limit=7680,
                completed_group_environment_steps=total_steps,policy_forward_calls=total_forward,optimizer_steps=0,
                model_promotion='none',process=process,
                interpretation='Selected15 four-clear losses. Only first action differs, then common frozen policy for at most256 placements. '
                    'Dependent local interventions; no whole-candidate-policy or long-term quality claim.'))
        print(json.dumps(process))
    finally:
        if env is not None:env.close()
        lock.close()


def verify_restart(folder):
    from resource_guard import now
    if (folder/'restart_receipt.json').exists():return
    def execute(groups):
        command=[sys.executable,str(Path(__file__).resolve()),'--run-dir',str(folder),'--through-groups',str(groups)]
        child=subprocess.Popen(command);code=child.wait()
        if code:raise subprocess.CalledProcessError(code,command)
        return child.pid
    first=execute(1)
    first_group=json.loads((folder/'group_00.json').read_text())
    protected=[folder/'group_00.json']
    for key in first_group['pair_manifests']:protected.extend(p for p in (folder/'pairs'/key).iterdir() if p.is_file())
    before={str(p):sha(p) for p in protected}
    second=execute(4);summary=json.loads((folder/'branch_summary.json').read_text())
    assert first!=second and summary['process']['pid']==second
    assert summary['process']['reused_groups']==[0] and summary['process']['computed_groups']==[1,2,3]
    assert all(sha(p)==digest for p,digest in before.items())
    durable_json(folder/'restart_receipt.json',dict(at=now(),status='passed',first_pid=first,second_pid=second,
        reused_groups=[0],computed_groups=[1,2,3],first_four_pairs_and_group_receipt_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--through-groups',type=int,choices=[1,2,3,4],default=4);parser.add_argument('--verify-restart',action='store_true')
    args=parser.parse_args()
    if args.verify_restart:verify_restart(args.run_dir.resolve())
    else:run(args.run_dir.resolve(),args.through_groups)
