"""Recover all eight predeclared prior points, grouped by original saved-action chunk."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from own_policy_capture import CaptureNativeEnv,sha
from restore_own_death_contexts import load_verified_chunk,restore_vector,replay_chunk,sampling_rng
from restore_rank_contexts import load_context,save_context,observation_digest
from verify_feature_backend_benchmark import identical
from training.evaluation.evaluate_resumable import durable_json


def run(folder,max_new_chunks):
    import torch
    from training.train_transaction import restore_global_rng
    from training.train_vector_transaction import training_source_identity
    from training.vector_env import make_vector_env
    from training.evaluation.feature_backend import inference_source_identity,evaluation_runtime_identity
    from resource_guard import now
    design=json.loads((folder/'design.json').read_text())
    def unchanged():
        for path,expected in design['frozen_files'].items():
            if sha(path)!=expected:raise ValueError('Frozen prior context input changed: '+path)
    unchanged()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no prior recovery fallback')
    if (training_source_identity()!=design['training_sources'] or
            inference_source_identity(resumable=True)!=design['inference_sources'] or
            evaluation_runtime_identity(False)!=design['evaluation_runtime']):
        raise ValueError('Prior recovery numerical source/runtime differs')
    selection=json.loads(Path(design['targets_path']).read_text());targets=selection['targets']
    chunks=sorted({target['source_chunk'] for target in targets})
    if (len(targets)!=8 or chunks!=design['planned_source_chunks'] or not 1<=max_new_chunks<=len(chunks) or
            any(target['seed']!=100005 or target['step']>=6921 for target in targets)):
        raise ValueError('All-eight prior recovery scope differs')
    torch.set_num_threads(1)
    identity=dict(design_sha256=sha(folder/'design.json'),targets_sha256=sha(design['targets_path']),
                  capture_identity=selection['capture_identity'])
    capture=Path(selection['capture_dir']);lock=(folder/'recovery.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=None;computed=[];reused=[];steps_this_process=0
    def expected_identity(target):return dict(identity,target=target)
    def load(number):
        directory=capture/f'chunk_{number:04d}'
        return load_verified_chunk(directory,design['frozen_files'][str(directory/'manifest.json')],selection['capture_identity'])
    try:
        for number in chunks:
            chosen=[target for target in targets if target['source_chunk']==number]
            receipt=folder/f'replay_{number:04d}.json'
            if receipt.exists():
                cached=json.loads(receipt.read_text())
                if cached['status']!='passed' or cached['identity']!=identity:raise ValueError('Prior recovery receipt differs')
                for target in chosen:
                    directory=folder/'contexts'/target['key']
                    if sha(directory/'manifest.json')!=cached['context_manifests'][target['key']]:
                        raise ValueError('Recovered prior context changed')
                    load_context(directory,expected_identity(target))
                reused.append(number);continue
            if len(computed)>=max_new_chunks:break
            initial,_,_=load(number-1);expected,data,manifest=load(number)
            if len(initial['workers'])!=8 or len(expected['workers'])!=8:raise ValueError('All8 workers required')
            if env is None:
                env=make_vector_env(8,max_pieces=20000,env_class=CaptureNativeEnv,
                    env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
            restore_global_rng(initial['rng']);rng=sampling_rng();views=restore_vector(env,initial)
            recovered,views,steps=replay_chunk(env,views,data,initial['round'],expected['round'],chosen)
            if (not identical(env.env_method('capture_export'),expected['workers']) or
                    [None if view is None else observation_digest(**view) for view in views]!=expected['views'] or
                    not identical(rng,sampling_rng()) or steps!=manifest['rows'] or steps>2048):
                raise ValueError('Prior chunk full endpoint, sampling RNG or work differs')
            for target in chosen:
                context=recovered[target['key']]
                if context['state']['pieces']!=target['step']-1:raise ValueError('Prior pre-action offset differs')
                save_context(folder/'contexts'/target['key'],expected_identity(target),context['state'],**context['view'])
            unchanged()
            durable_json(receipt,dict(at=now(),status='passed',identity=identity,pid=os.getpid(),chunk=number,
                actual_replayed_environment_steps=steps,policy_forward_calls=0,optimizer_steps=0,
                full_eight_worker_end_state_and_views_exact=True,parent_sampling_rng_unchanged=True,
                context_manifests={target['key']:sha(folder/'contexts'/target['key']/'manifest.json') for target in chosen}))
            computed.append(number);steps_this_process+=steps
        process=dict(at=now(),pid=os.getpid(),computed_chunks=computed,reused_chunks=reused,
            actual_environment_steps_this_process=steps_this_process,policy_forward_calls=0,optimizer_steps=0)
        durable_json(folder/f'process_{os.getpid()}.json',process)
        if all((folder/f'replay_{number:04d}.json').exists() for number in chunks):
            total=sum(json.loads((folder/f'replay_{number:04d}.json').read_text())['actual_replayed_environment_steps'] for number in chunks)
            if total>design['maximum_replayed_environment_steps']:raise ValueError('Prior recovery budget exceeded')
            durable_json(folder/'contexts_summary.json',dict(at=now(),status='passed',identity=identity,targets=targets,
                source_chunks=chunks,actual_replayed_environment_steps=total,policy_forward_calls=0,optimizer_steps=0,
                process=process,interpretation='All8 prior pre-action contexts reconstructed from saved actions. No new policy sampling/training.'))
        print(json.dumps(process))
    finally:
        if env is not None:env.close()
        lock.close()


def verify_restart(folder):
    from resource_guard import now
    if (folder/'restart_receipt.json').exists() or any(folder.glob('replay_*.json')) or (folder/'contexts').exists():
        raise RuntimeError('Prior recovery outputs exist; inspect and continue directly, do not repeat initial verifier')
    def execute(count):
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run-dir',str(folder),'--max-new-chunks',str(count)])
        if child.wait():raise RuntimeError('Prior recovery child failed: '+str(child.pid))
        return child.pid
    first=execute(1)
    protected=[*folder.glob('replay_*.json'),*folder.glob('contexts/*/*')]
    before={str(path):sha(path) for path in protected if path.is_file()}
    second=execute(2);process=json.loads((folder/f'process_{second}.json').read_text())
    if (first==second or process['reused_chunks']!=[26] or process['computed_chunks']!=[27] or
            any(sha(path)!=expected for path,expected in before.items())):
        raise ValueError('Prior actual restart did not preserve its first committed chunk and contexts')
    durable_json(folder/'restart_receipt.json',dict(at=now(),status='passed',first_pid=first,second_pid=second,
        first_chunk_four_context_files_unchanged=True,protected_files=before,second_process=process))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--max-new-chunks',type=int,choices=[1,2],default=2);parser.add_argument('--verify-restart',action='store_true')
    args=parser.parse_args()
    if args.verify_restart:verify_restart(args.run_dir.resolve())
    else:run(args.run_dir.resolve(),args.max_new_chunks)
