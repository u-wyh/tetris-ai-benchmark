"""Restore complete worker contexts by replaying saved actions, never resampling."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np

from collect_critic_calibration import array_manifest
from training.evaluation.evaluate_resumable import durable_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_digest(observation,mask):
    return dict(observation=hashlib.sha256(b''.join(observation[k].tobytes() for k in sorted(observation))).hexdigest(),
                mask=hashlib.sha256(np.asarray(mask,dtype=bool).tobytes()).hexdigest())


def load_workers(folder):
    metadata=json.loads((folder/'checkpoint.json').read_text())
    if sha(folder/'state.pkl')!=metadata['state_sha256']:
        raise ValueError('Worker snapshot checksum differs')
    # Only this project's hash-verified, frozen local snapshot is accepted.
    state=pickle.loads((folder/'state.pkl').read_bytes())
    if (len(state['workers'])!=len(metadata['worker_digests']) or
            state['trainer']['num_timesteps']!=metadata['num_timesteps']):
        raise ValueError('Worker snapshot receipt differs')
    if any('get_legal_placements' in w['core'].__dict__ for w in state['workers']):
        raise ValueError('Temporary placement override in worker snapshot')
    return state['workers'],metadata


def restore_vector(env,folder):
    workers,metadata=load_workers(folder)
    if len(workers)!=env.num_envs:raise ValueError('Worker count differs')
    for i,worker in enumerate(workers):env.env_method('set_state',worker,indices=[i])
    observations=env.env_method('get_observation');masks=env.env_method('action_masks')
    if [observation_digest(o,m) for o,m in zip(observations,masks)]!=metadata['worker_digests']:
        raise ValueError('Restored worker observation or mask differs')
    return {key:np.stack([o[key] for o in observations]) for key in observations[0]}


def save_context(directory,identity,state,observation,mask):
    from verify_feature_backend_benchmark import identical
    if directory.exists():
        old,metadata=load_context(directory,identity)
        if not identical(old,state) or metadata['observation_digest']!=observation_digest(observation,mask):
            raise ValueError('Replayed context differs from committed context')
        return False
    pending=directory.with_name(directory.name+'.pending')
    if pending.exists():
        abandoned=directory.parent/'abandoned';abandoned.mkdir(exist_ok=True)
        pending.rename(abandoned/(pending.name+'.'+str(os.getpid())))
    pending.mkdir(parents=True)
    payload=pickle.dumps(state,protocol=5)
    with (pending/'state.pkl').open('xb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    if not identical(pickle.loads(payload),state):raise ValueError('Context snapshot round trip differs')
    durable_json(pending/'manifest.json',dict(identity=identity,state_sha256=sha(pending/'state.pkl'),
        observation_digest=observation_digest(observation,mask)))
    pending.rename(directory)
    fd=os.open(directory.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return True


def load_context(directory,identity):
    metadata=json.loads((directory/'manifest.json').read_text())
    if metadata['identity']!=identity or sha(directory/'state.pkl')!=metadata['state_sha256']:
        raise ValueError('Rank context identity or checksum differs')
    state=pickle.loads((directory/'state.pkl').read_bytes())
    if 'get_legal_placements' in state['core'].__dict__:raise ValueError('Temporary placement override')
    return state,metadata


def replay_segment(env,observations,arrays,on_before_step):
    """SubprocVecEnv auto-reset is intentional here: reproduce original capture."""
    steps=len(arrays['actions']);width=env.num_envs
    for time in range(steps):
        for name in observations:
            np.testing.assert_array_equal(observations[name],arrays['obs_'+name][time])
        masks=np.stack(env.env_method('action_masks'))
        np.testing.assert_array_equal(masks,arrays['action_masks'][time].astype(bool))
        on_before_step(time,observations,masks)
        observations,rewards,dones,infos=env.step(arrays['actions'][time,:,0])
        np.testing.assert_array_equal(rewards,arrays['raw_env_rewards'][time])
        np.testing.assert_array_equal(dones,arrays['step_dones'][time])
        truncated=np.array([i.get('TimeLimit.truncated',False) for i in infos],dtype=bool)
        np.testing.assert_array_equal(truncated,arrays['step_truncated'][time])
        np.testing.assert_array_equal(dones & ~truncated,arrays['step_terminated'][time])
        for name in ['episode_seed','pieces','cleared_lines','holes_before','holes_after','height_after','new_holes']:
            np.testing.assert_array_equal([i[name] for i in infos],arrays['info_'+name][time])
    return observations,steps*width


def selected_targets(geometry):
    metadata=json.loads((geometry/'rows_manifest.json').read_text())
    if sha(geometry/'flipped_geometry.npz')!=metadata['file_sha256']:
        raise ValueError('Geometry ledger identity differs')
    with np.load(geometry/'flipped_geometry.npz',allow_pickle=False) as data:rows={n:data[n] for n in data.files}
    if array_manifest(rows)!=metadata['arrays']:raise ValueError('Geometry ledger arrays differ')
    targets=[]
    for i in np.flatnonzero(rows['delta_cleared']==-4):
        row={n:int(rows[n][i]) for n in ['chunk','worker','global_step','reference_action','candidate_action']}
        row['local_step']=row['global_step']-(row['chunk']-1)*512
        if not 1<=row['chunk']<=4 or not 0<=row['worker']<8 or not 0<=row['local_step']<512:
            raise ValueError('Target row axes differ')
        row['key']=f"c{row['chunk']:02d}_t{row['local_step']:03d}_w{row['worker']}"
        targets.append(row)
    if len({r['key'] for r in targets})!=len(targets):raise ValueError('Duplicate target contexts')
    return targets


def run(folder,through_chunk):
    import torch
    from resource_guard import now
    from training.train_vector_transaction import TransactionNativeSearchCandidateEnv,training_source_identity
    from training.vector_env import make_vector_env
    from verify_feature_backend_benchmark import identical
    design=json.loads((folder/'design.json').read_text())
    def unchanged():
        for name,digest in design['frozen_files'].items():
            if sha(name)!=digest:raise ValueError('Frozen replay input changed: '+name)
    unchanged()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refuse replay fallback')
    if training_source_identity()!=design['training_sources']:raise ValueError('Replay source identity differs')
    targets=selected_targets(Path(design['geometry_dir']))
    if len(targets)!=design['expected_contexts']:raise ValueError('Target selection differs')
    identity=dict(design_sha256=sha(folder/'design.json'),training_sources=design['training_sources'])
    lock=(folder/'replay.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=None;computed=[];reused=[];total_steps=0
    try:
        for index in range(1,through_chunk+1):
            chunk_targets=[t for t in targets if t['chunk']==index]
            receipt=folder/f'chunk_{index:04d}.json'
            expected_identity=dict(identity,chunk=index)
            if receipt.exists():
                record=json.loads(receipt.read_text())
                if record['identity']!=expected_identity or record['status']!='passed':
                    raise ValueError('Completed replay identity differs')
                for target in chunk_targets:load_context(folder/'contexts'/target['key'],dict(identity,target=target))
                reused.append(index);continue
            if env is None:
                env=make_vector_env(8,max_pieces=10000,env_class=TransactionNativeSearchCandidateEnv,
                    env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
            dataset=Path(design['dataset_dir']);segment=dataset/f'chunk_{index:04d}'
            before=dataset/('initial' if index==1 else f'chunk_{index-1:04d}')
            observations=restore_vector(env,before)
            meta=json.loads((segment/'dataset.json').read_text())
            if sha(segment/'rollout.npz')!=meta['file_sha256']:raise ValueError('Rollout checksum differs')
            fields=['obs_state','obs_candidate','action_masks','actions','raw_env_rewards','step_dones',
                'step_terminated','step_truncated']+['info_'+n for n in
                ['episode_seed','pieces','cleared_lines','holes_before','holes_after','height_after','new_holes']]
            with np.load(segment/'rollout.npz',allow_pickle=False) as data:arrays={n:data[n] for n in fields}
            if array_manifest(arrays)!={n:meta['arrays'][n] for n in fields}:
                raise ValueError('Rollout array identity differs')
            def capture(time,obs,masks):
                for target in chunk_targets:
                    if target['local_step']!=time:continue
                    worker=target['worker'];a,b=target['reference_action'],target['candidate_action']
                    if not masks[worker,a] or not masks[worker,b]:raise ValueError('Target action not legal')
                    if int(obs['candidate'][worker,a,5])!=4 or int(obs['candidate'][worker,b,5])!=0:
                        raise ValueError('Target four-clear selection differs')
                    state=env.env_method('get_state',indices=[worker])[0]
                    observation={n:v[worker] for n,v in obs.items()}
                    save_context(folder/'contexts'/target['key'],dict(identity,target=target),state,observation,masks[worker])
            observations,count=replay_segment(env,observations,arrays,capture)
            expected,_=load_workers(segment);actual=env.env_method('get_state')
            if not identical(expected,actual):raise ValueError('Replayed final complete worker states differ')
            total_steps+=count;unchanged()
            durable_json(receipt,dict(at=now(),status='passed',identity=expected_identity,pid=os.getpid(),
                environment_steps=count,all_observations_masks_rewards_info_exact=True,
                final_complete_worker_core_rng_states_exact=True,contexts=[t['key'] for t in chunk_targets]))
            computed.append(index)
            del arrays
        result=dict(at=now(),pid=os.getpid(),computed_chunks=computed,reused_chunks=reused,
            environment_steps_this_process=total_steps,policy_forward_calls=0,optimizer_steps=0)
        durable_json(folder/f'process_{os.getpid()}.json',result)
        if through_chunk==4:
            for target in targets:load_context(folder/'contexts'/target['key'],dict(identity,target=target))
            durable_json(folder/'contexts_summary.json',dict(at=now(),status='passed',identity=identity,targets=targets,
                context_count=len(targets),total_original_prefix_environment_steps=16384,process=result,
                no_training=True,interpretation='Exact replay of saved stochastic actions into complete worker states. No resampled action or new policy trajectory.'))
        print(json.dumps(result))
    finally:
        if env is not None:env.close()
        lock.close()


def verify_restart(folder):
    from resource_guard import now
    if (folder/'restart_receipt.json').exists():return
    def execute(count):
        command=[sys.executable,str(Path(__file__).resolve()),'--run-dir',str(folder),'--through-chunk',str(count)]
        child=subprocess.Popen(command);code=child.wait()
        if code:raise subprocess.CalledProcessError(code,command)
        return child.pid
    first=execute(2)
    protected=[folder/f'chunk_{i:04d}.json' for i in [1,2]]
    for path in (folder/'contexts').iterdir():
        if path.is_dir() and path.name.startswith(('c01_','c02_')):
            protected.extend(p for p in path.iterdir() if p.is_file())
    before={str(p):sha(p) for p in protected}
    second=execute(4);summary=json.loads((folder/'contexts_summary.json').read_text())
    assert first!=second and summary['process']['pid']==second
    assert summary['process']['reused_chunks']==[1,2] and summary['process']['computed_chunks']==[3,4]
    assert all(sha(p)==digest for p,digest in before.items())
    durable_json(folder/'restart_receipt.json',dict(at=now(),status='passed',first_pid=first,second_pid=second,
        reused_chunks=[1,2],computed_chunks=[3,4],prior_receipts_and_context_files_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--through-chunk',type=int,choices=[1,2,3,4],default=4);parser.add_argument('--verify-restart',action='store_true')
    args=parser.parse_args()
    if args.verify_restart:verify_restart(args.run_dir.resolve())
    else:run(args.run_dir.resolve(),args.through_chunk)
