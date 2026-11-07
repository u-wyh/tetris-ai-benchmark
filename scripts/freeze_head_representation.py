"""Export public-observation score-head features with one immutable CUDA source."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np

MODEL_SHA='eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34'
CACHE_VERSION='public-last-head640-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unchanged(frozen):
    for path,want in frozen.items():
        if sha(path)!=want:raise ValueError('Frozen representation input changed: '+path)


def validate_arrays(arrays,metadata):
    n=len(metadata['states'])
    specs={'h':np.float32,'raw_score':np.float32,'actions':np.int64,'offsets':np.int64,
           'original_weight':np.float32,'original_bias':np.float32}
    if (set(arrays)!=set(specs) or any(arrays[k].dtype!=dtype or not np.isfinite(arrays[k]).all()
            for k,dtype in specs.items()) or arrays['offsets'].shape!=(n+1,) or
            arrays['offsets'][0]!=0 or np.any(np.diff(arrays['offsets'])<=0) or
            arrays['h'].shape!=(int(arrays['offsets'][-1]),64) or
            arrays['actions'].shape!=arrays['raw_score'].shape or
            arrays['actions'].shape!=(len(arrays['h']),) or
            arrays['original_weight'].shape!=(64,) or arrays['original_bias'].shape!=(1,)):
        raise ValueError('Public representation array schema differs')
    for i,state in enumerate(metadata['states']):
        start,stop=arrays['offsets'][i:i+2];actions=arrays['actions'][start:stop]
        if (np.any(actions<0) or np.any(actions>=1840) or np.any(np.diff(actions)<=0) or
                state['reference_action'] not in actions):
            raise ValueError('Invalid legal action rows or reference')
        mask=np.zeros(1840,dtype=bool);mask[actions]=True
        if hashlib.sha256(mask.tobytes()).hexdigest()!=state['input_digest']['mask']:
            raise ValueError('Exported action set differs from sealed public mask')


def load_cache(folder,metadata_sha,arrays_sha):
    if sha(folder/'cache_metadata.json')!=metadata_sha or sha(folder/'features.npz')!=arrays_sha:
        raise ValueError('External representation seal differs before reading arrays')
    metadata=json.loads((folder/'cache_metadata.json').read_text())
    if (metadata['version']!=CACHE_VERSION or metadata['arrays_sha256']!=arrays_sha or
            metadata['status']!='passed' or metadata['source_model_sha256']!=MODEL_SHA or
            metadata['source_model_counts']!=[30003200,36625] or len(metadata['states'])!=640 or
            metadata['work']['gpu_forward_calls']!=80 or
            metadata['source_policy_adam_counts_rng_unchanged'] is not True or
            metadata['eligible_for_single_joint_lp'] is not True):
        raise ValueError('Frozen original CUDA representation scope differs')
    with np.load(folder/'features.npz',allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    validate_arrays(arrays,metadata)
    for i,state in enumerate(metadata['states']):
        if i<8:
            if (state['kind']!='pair' or state['pair']!=i//2 or state['split']!='train' or
                    state['role']!=('base' if i%2==0 else 'alternative') or
                    state['seed']!=[992000,992004,992006,992008][i//2]):
                raise ValueError('Four predetermined training pair order differs')
            actions=arrays['actions'][arrays['offsets'][i]:arrays['offsets'][i+1]]
            if state['good_action'] not in actions:raise ValueError('Preferred action is masked')
        else:
            lane=(i-8)%8;chunk=(i-8)//8
            if (state['kind']!='normal' or state['lane']!=lane or state['chunk']!=chunk or
                    state['seed']!=100000+lane or state['split']!=('train' if lane<4 else 'validation')):
                raise ValueError('Normal whole-episode train/validation order differs')
    return arrays,metadata


def public_view(state):
    from training.candidate_features import candidate_features
    from training.env.tetris_env import TetrisEnv
    public=state['core'].get_public_observation();matrix,mask=candidate_features(public)
    return dict(observation=dict(state=TetrisEnv._encode_observation(public),candidate=matrix),mask=mask)


def run(folder):
    import torch
    from sb3_contrib import MaskablePPO
    from restore_own_death_contexts import load_verified_chunk
    from restore_rank_contexts import observation_digest
    from paired_branch_capture import specs,validate_snapshot
    from paired_branch_guard import require_authorization
    from training.train_transaction import global_rng_state
    from training.train_vector_transaction import training_source_identity
    from training.evaluation.feature_backend import inference_source_identity,evaluation_runtime_identity
    from training.evaluation.evaluate_resumable import durable_json
    from verify_feature_backend_benchmark import identical
    from collect_critic_calibration import array_manifest
    from resource_guard import now
    design=json.loads((folder/'representation_design.json').read_text());unchanged(design['frozen_files'])
    require_authorization(design,folder)
    if (folder/'execution.json').exists() or (folder/'cache_metadata.json').exists():
        raise FileExistsError('Representation attempt exists; inspect actual work instead of repeating CUDA')
    receipt=dict(at=now(),pid=os.getpid(),status='started',gpu_forward_attempts=0,gpu_forward_calls=0,
                 environment_steps=0,optimizer_steps=0,model_loads=0)
    durable_json(folder/'execution.json',{**receipt,'gpu_forward_calls':None,'gpu_forward_attempts':None,
        'scope':'Started only; CUDA totals unverified until final receipt. No blind retry.'})
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; preserve without CPU/host fallback')
        if (training_source_identity()!=design['training_sources'] or
                inference_source_identity(resumable=True)!=design['inference_sources'] or
                evaluation_runtime_identity(False)!=design['evaluation_runtime'] or sha(design['model'])!=MODEL_SHA):
            raise ValueError('Original numerical source/runtime/model differs')
        evidence=json.loads(Path(design['window_result']).read_text())
        paired=Path(design['paired_job']);paired_design=json.loads((paired/'design.json').read_text())
        if (evidence['status']!='passed' or evidence['label_counts']['beneficial']!=4 or
                evidence['model']['sha256']!=MODEL_SHA or
                [p['case'] for p in evidence['comparisons']]!=paired_design['cases']):
            raise ValueError('Four fixed current-GPU preference labels differ')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        source=MaskablePPO.load(design['model'],device='cuda',env=None);receipt['model_loads']=1
        if source.device.type!='cuda' or source.get_env() is not None or [source.num_timesteps,source._n_updates]!=[30003200,36625]:
            raise ValueError('Original source CUDA/counts differ')
        source.policy.set_training_mode(False)
        before=copy.deepcopy((source.policy.state_dict(),source.policy.optimizer.state_dict(),global_rng_state(source),source._n_updates))
        head=source.policy.score_head[-1]
        if not isinstance(head,torch.nn.Linear) or (head.in_features,head.out_features)!=(64,1):
            raise ValueError('Expected final linear64 score head')
        w=head.weight.detach().cpu().numpy().reshape(-1).copy();bias=head.bias.detach().cpu().numpy().copy()
        views_data=[];states=[];max_error=0.
        def observe(views,records):
            nonlocal max_error
            require_authorization(design,folder)
            if len(views)!=8 or receipt['gpu_forward_attempts']>=80:
                raise ValueError('Fixed80 CUDA8 forward limit differs')
            obs={k:np.stack([v['observation'][k] for v in views]) for k in views[0]['observation']}
            masks=np.stack([v['mask'] for v in views]);captured=[]
            def hook(module,inputs,output):
                captured.append((inputs[0].detach().cpu().numpy().copy(),output.detach().cpu().numpy().reshape(-1).copy()))
            handle=head.register_forward_hook(hook);receipt['gpu_forward_attempts']+=1
            try:
                with torch.inference_mode():
                    tensors={k:torch.as_tensor(v,device='cuda') for k,v in obs.items()}
                    distribution=source.policy.get_distribution(tensors,action_masks=masks)
                    modes=distribution.distribution.probs.argmax(dim=1).cpu().numpy()
                receipt['gpu_forward_calls']+=1
            finally:handle.remove()
            if len(captured)!=1:raise ValueError('Exactly one linear head invocation required per batch')
            latent,raw=captured[0]
            if latent.shape!=(int(masks.sum()),64) or not np.isfinite(latent).all() or not np.isfinite(raw).all():
                raise ValueError('Nonfinite or invalid public head output')
            error=float(np.max(np.abs(latent.astype(np.float64)@w.astype(np.float64)+float(bias[0])-raw)))
            max_error=max(error,max_error)
            if error>1e-4:raise ValueError('Raw score reconstruction exceeds predeclared1e-4')
            offset=0
            for index,(view,record) in enumerate(zip(views,records)):
                actions=np.flatnonzero(view['mask']).astype(np.int64);stop=offset+len(actions)
                if int(modes[index])!=record['reference_action']:
                    raise ValueError('Actual GPU probability mode differs from sealed teacher')
                states.append({**record,'input_digest':observation_digest(**view)})
                views_data.append(dict(h=latent[offset:stop].copy(),raw_score=raw[offset:stop].copy(),actions=actions))
                offset=stop
        directory=paired/'chunk_0000';identity=json.loads((directory/'manifest.json').read_text())['identity']
        if identity['design_sha256']!=sha(paired/'design.json'):raise ValueError('Paired source design identity differs')
        initial,_,_=load_verified_chunk(directory,design['frozen_files'][str(directory/'manifest.json')],identity)
        validate_snapshot(initial,paired_design);ordered=specs(paired_design['cases'])
        views=[public_view(worker['state']) for worker in initial['workers']]
        if [observation_digest(**view) for view in views]!=initial['views']:
            raise ValueError('Paired public reconstruction differs')
        records=[dict(kind='pair',pair=i//2,branch_id=spec['branch_id'],seed=spec['case']['seed'],
            role=spec['role'],reference_action=spec['case']['baseline_action'],good_action=spec['case']['action'],split='train')
            for i,spec in enumerate(ordered)]
        observe(views,records)
        duplicates=[]
        for i in range(0,8,2):
            left,right=views_data[i:i+2]
            if not np.array_equal(left['actions'],right['actions']) or states[i]['input_digest']!=states[i+1]['input_digest']:
                raise ValueError('Paired public inputs/action sets differ')
            duplicates.append(dict(pair=i//2,h_max_difference=float(np.max(np.abs(left['h']-right['h']))),
                                   score_max_difference=float(np.max(np.abs(left['raw_score']-right['raw_score'])))))
        base=Path(design['normal_dir']);previous=None;previous_sha=None
        for index in range(80):
            directory=base/f'chunk_{index:04d}'
            saved,rows,manifest=load_verified_chunk(directory,design['frozen_files'][str(directory/'manifest.json')],design['normal_identity'])
            if (manifest['number']!=index or manifest['previous_manifest_sha256']!=previous_sha or
                    saved['round']!=min(index*256,20000) or saved['environment_steps']!=saved['round']*8):
                raise ValueError('Normal complete chain/counters differ')
            if previous is not None:
                if list(map(int,rows['lane'][:8]))!=list(range(8)) or not (rows['round'][:8]==previous['round']).all():
                    raise ValueError('Normal next-round teacher rows differ')
                views=[];records=[]
                for lane,worker in enumerate(previous['workers']):
                    state=worker['state'];view=public_view(state);digest=observation_digest(**view)
                    if (not worker['active'] or state['finished'] or state['episode_count']!=1 or
                            state['episode_seed']!=100000+lane or int(rows['episode_seed'][lane])!=100000+lane or
                            digest!=previous['views'][lane] or digest!=dict(observation=str(rows['observation_sha256'][lane]),mask=str(rows['mask_sha256'][lane]))):
                        raise ValueError('Normal source episode/public tensors or teacher inputs differ')
                    views.append(view);records.append(dict(kind='normal',chunk=index-1,round=previous['round'],lane=lane,
                        seed=100000+lane,split='train' if lane<4 else 'validation',reference_action=int(rows['action'][lane])))
                observe(views,records)
            previous=saved;previous_sha=design['frozen_files'][str(directory/'manifest.json')]
        if len(states)!=640 or receipt['gpu_forward_calls']!=80:raise ValueError('Expected8paired+632normal states/80forward')
        arrays={key:np.concatenate([v[key] for v in views_data]) for key in ('h','raw_score','actions')}
        arrays.update(offsets=np.r_[0,np.cumsum([len(v['actions']) for v in views_data])].astype(np.int64),
                      original_weight=w,original_bias=bias)
        validate_arrays(arrays,dict(states=states))
        source_unchanged=identical(before,(source.policy.state_dict(),source.policy.optimizer.state_dict(),global_rng_state(source),source._n_updates))
        if not source_unchanged:raise ValueError('Original source weights/Adam/counts/sampling RNG changed')
        unchanged(design['frozen_files'])
        with (folder/'features.npz').open('xb') as stream:
            np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        metadata=dict(at=now(),version=CACHE_VERSION,status='passed',source_model_sha256=MODEL_SHA,
            source_model_counts=[30003200,36625],states=states,arrays=array_manifest(arrays),arrays_sha256=sha(folder/'features.npz'),
            source_policy_adam_counts_rng_unchanged=True,max_score_reconstruction_error=max_error,
            duplicate_pair_differences=duplicates,eligible_for_single_joint_lp=all(p['h_max_difference']<=1e-6 for p in duplicates),
            eligible_for_training=False,work=dict(gpu_forward_calls=80,environment_steps=0,optimizer_steps=0,model_loads=1),
            scope='Public observation derived latents only,4 current-GPU verified training pairs and632 correlated normal states. No model update, normal/validation quality or historical identity claim.')
        durable_json(folder/'cache_metadata.json',metadata)
        receipt.update(at=now(),status='completed',original_state_unchanged=True,cache_metadata_sha256=sha(folder/'cache_metadata.json'))
    except BaseException as exc:
        receipt.update(at=now(),status='failed',error=repr(exc));raise
    finally:durable_json(folder/'execution.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    run(p.parse_args().run_dir.resolve())
