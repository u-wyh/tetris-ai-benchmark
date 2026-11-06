"""Recoverable same-state policy ranking diagnostic on immutable saved rollouts."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from collect_critic_calibration import array_manifest
from training.evaluation.evaluate_resumable import durable_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def masked_log_probabilities(logits, masks):
    """Float64 renormalization of float32 policy logits; no probability floor.

    Masked entries are -inf and contribute zero to KL. Underflow of exp at
    extreme legal logits is allowed, while their finite log-prob remains usable.
    """
    logits, masks = np.asarray(logits, dtype=np.float64), np.asarray(masks)
    if (logits.ndim != 2 or logits.shape != masks.shape or masks.dtype != np.bool_ or
            not masks.any(axis=1).all() or not np.isfinite(logits[masks]).all()):
        raise ValueError('Invalid masked policy logits')
    selected = np.where(masks, logits, -np.inf)
    shifted = selected-selected.max(axis=1, keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def compare_forward(reference, candidate, masks, features):
    """Actions/probabilities come from the actual float32 distribution mode.

    Float64 KL describes the distributions induced by its finite legal logits.
    This preserves float32 probability ties that need not match a logit argmax.
    """
    masks, features = np.asarray(masks), np.asarray(features)
    if (masks.ndim != 2 or masks.dtype != np.bool_ or not masks.any(axis=1).all() or
            features.shape != (*masks.shape, 16) or features.dtype != np.uint8):
        raise ValueError('Feature rows or dtype differ')
    rows = np.arange(len(masks))
    for result in (reference, candidate):
        actions, probabilities = result['actions'], result['probabilities']
        if (actions.shape != (len(masks),) or actions.dtype.kind not in 'iu' or
                np.any(actions < 0) or np.any(actions >= masks.shape[1]) or
                not masks[rows, actions].all() or probabilities.shape != masks.shape or
                not np.isfinite(probabilities).all() or np.any(probabilities < 0) or
                np.any(probabilities[~masks] != 0) or
                not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-5, atol=1e-6) or
                not np.array_equal(actions, probabilities.argmax(axis=1))):
            raise ValueError('Actual policy mode/probabilities differ from legal mask')
    log_p = masked_log_probabilities(reference['logits'], masks)
    log_q = masked_log_probabilities(candidate['logits'], masks)
    delta = np.zeros_like(log_p)
    np.subtract(log_p, log_q, out=delta, where=masks)
    kl = (np.exp(log_p)*delta).sum(axis=1)
    if np.any(kl < -1e-12) or not np.isfinite(kl).all():
        raise ValueError('Invalid masked KL')
    kl = np.maximum(kl, 0)  # Roundoff only, never clip the probabilities.
    legal_logits = np.where(masks, reference['logits'], -np.inf).astype(np.float64)
    ranked = np.sort(legal_logits, axis=1)
    counts = masks.sum(axis=1)
    margin = np.full(len(masks), np.inf)
    multiple = counts > 1
    margin[multiple] = ranked[multiple, -1]-ranked[multiple, -2]
    a, b = reference['actions'], candidate['actions']
    return dict(reference_action=a, candidate_action=b, disagrees=(a != b), legal_count=counts,
        reference_top2_logit_margin=margin, masked_kl_reference_candidate=kl,
        reference_greedy_probability=reference['probabilities'][rows, a],
        candidate_greedy_probability=candidate['probabilities'][rows, b],
        candidate_probability_of_reference=candidate['probabilities'][rows, a],
        reference_probability_of_candidate=reference['probabilities'][rows, b],
        reference_mode_differs_from_logit_argmax=(a != legal_logits.argmax(axis=1)),
        reference_features=features[rows, a], candidate_features=features[rows, b])


def forward(policy, observations, masks):
    import torch
    with torch.no_grad():
        distribution = policy.get_distribution(observations, action_masks=masks)
        return dict(logits=distribution.distribution.logits.detach().cpu().numpy().copy(),
            probabilities=distribution.distribution.probs.detach().cpu().numpy().copy(),
            actions=distribution.get_actions(deterministic=True).detach().cpu().numpy().copy())


def terminal_distances(terminated, dones):
    """Future termination distance in the SAME episode; -1 means censored.

    A timeout or another episode boundary never borrows a later terminal label.
    The current action's termination has distance zero.
    """
    terminated, dones = np.asarray(terminated), np.asarray(dones)
    if terminated.dtype != np.bool_ or dones.dtype != np.bool_ or terminated.shape != dones.shape:
        raise ValueError('Invalid terminal markers')
    if terminated.ndim != 2 or np.any(terminated & ~dones):
        raise ValueError('Terminal without an episode boundary')
    result = np.full(terminated.shape, -1, dtype=np.int32)
    for worker in range(terminated.shape[1]):
        future = -1
        for step in range(len(terminated)-1, -1, -1):
            if dones[step, worker]: future = step if terminated[step, worker] else -1
            if future >= 0: result[step, worker] = future-step
    return result


def load_chunk(segment):
    metadata = json.loads((segment/'dataset.json').read_text())
    if sha(segment/'rollout.npz') != metadata['file_sha256']:
        raise ValueError('Saved rollout file identity differs')
    fields = ['obs_state','obs_candidate','action_masks','actions','episode_starts',
              'step_dones','step_terminated','step_truncated','info_episode_seed']
    with np.load(segment/'rollout.npz', allow_pickle=False) as data:
        arrays = {name:data[name] for name in fields}
    if array_manifest(arrays) != {name:metadata['arrays'][name] for name in fields}:
        raise ValueError('Saved rollout array identity differs')
    masks = arrays['action_masks']
    if (arrays['obs_state'].shape != (512,8,237) or arrays['obs_candidate'].shape != (512,8,1840,16) or
            arrays['obs_state'].dtype != np.float32 or arrays['obs_candidate'].dtype != np.uint8 or
            masks.shape != (512,8,1840) or not np.isin(masks, [0,1]).all() or
            not np.isin(arrays['episode_starts'], [0,1]).all() or
            arrays['actions'].shape != (512,8,1) or not np.isfinite(arrays['obs_state']).all()):
        raise ValueError('Saved observation axes or masks differ')
    if any(arrays[name].shape != (512,8) for name in fields if name not in
           ['obs_state','obs_candidate','action_masks','actions']):
        raise ValueError('Saved marker axes differ')
    if not np.take_along_axis(masks, arrays['actions'], axis=2).all():
        raise ValueError('Saved sampled action was not legal')
    np.testing.assert_array_equal(arrays['step_terminated'], arrays['step_dones'] & ~arrays['step_truncated'])
    np.testing.assert_array_equal(arrays['episode_starts'][1:].astype(bool), arrays['step_dones'][:-1])
    return arrays


def read_cache(directory, identity):
    if not directory.exists(): return None
    metadata = json.loads((directory/'manifest.json').read_text())
    if metadata['identity'] != identity or sha(directory/'rows.npz') != metadata['file_sha256']:
        raise ValueError('Rank cache identity differs')
    with np.load(directory/'rows.npz', allow_pickle=False) as data:
        arrays = {name:data[name] for name in data.files}
    if array_manifest(arrays) != metadata['arrays']:
        raise ValueError('Rank cache arrays differ')
    return arrays


def commit_cache(directory, identity, arrays, model_audit):
    if directory.exists(): raise FileExistsError('Preserve committed rank chunk')
    pending = directory.with_name(directory.name+'.pending')
    if pending.exists():
        abandoned = directory.parent/'abandoned'; abandoned.mkdir(exist_ok=True)
        pending.rename(abandoned/(pending.name+'.'+str(os.getpid())))
    pending.mkdir(parents=True)
    with (pending/'rows.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    with np.load(pending/'rows.npz', allow_pickle=False) as data:
        assert array_manifest({n:data[n] for n in data.files}) == array_manifest(arrays)
    durable_json(pending/'manifest.json',dict(identity=identity,arrays=array_manifest(arrays),
        file_sha256=sha(pending/'rows.npz'),model_audit=model_audit))
    pending.rename(directory)
    fd=os.open(directory.parent, os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def summarize(arrays, bins):
    from training.candidate_features import FEATURE_NAMES
    def group(selected):
        count=int(selected.sum())
        if not count:return dict(samples=0)
        disagree=selected & arrays['disagrees']
        kl=arrays['masked_kl_reference_candidate'][selected]
        delta=arrays['candidate_features'].astype(np.int16)-arrays['reference_features'].astype(np.int16)
        return dict(samples=count,greedy_disagreements=int(disagree.sum()),
            disagreement_fraction=float(disagree.sum()/count),mean_kl=float(kl.mean()),max_kl=float(kl.max()),
            mean_candidate_probability_of_reference=float(arrays['candidate_probability_of_reference'][selected].mean()),
            flipped_feature_delta_means={name:float(delta[disagree,index].mean())
                for index,name in enumerate(FEATURE_NAMES)} if disagree.any() else None)
    margin=arrays['reference_top2_logit_margin']
    groups=dict(overall=group(np.ones(len(margin),dtype=bool)),
        per_worker={str(w):group(arrays['worker']==w) for w in range(8)},
        margin_bins={f'[{low},{high})':group((margin>=low)&(margin<high))
            for low,high in zip(bins[:-1], bins[1:])},
        single_legal_action=group(arrays['legal_count']==1),
        terminal_within32=group((arrays['terminal_distance']>=0)&(arrays['terminal_distance']<32)),
        outside_observed_terminal32=group((arrays['terminal_distance']<0)|(arrays['terminal_distance']>=32)))
    groups['reference_float32_mode_logit_argmax_mismatches']=int(arrays['reference_mode_differs_from_logit_argmax'].sum())
    return groups


def run(folder, through_chunk):
    import torch
    from sb3_contrib import MaskablePPO
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    from verify_feature_backend_benchmark import identical
    from resource_guard import now
    design=json.loads((folder/'design.json').read_text())
    def unchanged():
        for name,digest in design['frozen_files'].items():
            if sha(name)!=digest:raise RuntimeError('Frozen rank input changed: '+name)
    unchanged()
    runtime=evaluation_runtime_identity(False)
    if runtime!=design['evaluation_runtime'] or inference_source_identity(resumable=True)!=design['inference_sources']:
        raise ValueError('Policy inference runtime differs')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    lock=(folder/'rank.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        identity=dict(design_sha256=sha(folder/'design.json'),runtime=runtime,device='cuda',float32_policy_float64_kl=True)
        dataset=Path(design['dataset_dir']);models={}; snapshots={}; counts={}
        caches={index:read_cache(folder/f'rows_{index:04d}',dict(identity,chunk=index)) for index in range(1,5)}
        pending=[i for i in range(1,through_chunk+1) if caches[i] is None]
        markers=[]
        for i in range(1,5):
            metadata=json.loads((dataset/f'chunk_{i:04d}'/'dataset.json').read_text())
            if metadata['model_sha256']!=design['models']['reference']['sha256'] or metadata['chunk']!=i:
                raise ValueError('Dataset provenance differs from reference model/chunk')
            with np.load(dataset/f'chunk_{i:04d}'/'rollout.npz',allow_pickle=False) as data:
                markers.append({name:data[name] for name in ['step_terminated','step_dones','episode_starts','info_episode_seed']})
        for previous,current in zip(markers,markers[1:]):
            np.testing.assert_array_equal(previous['step_dones'][-1],current['episode_starts'][0].astype(bool))
        distance=terminal_distances(np.concatenate([m['step_terminated'] for m in markers]),
                                    np.concatenate([m['step_dones'] for m in markers]))
        if pending:
            for label,spec in design['models'].items():
                assert sha(spec['path'])==spec['sha256']
                model=MaskablePPO.load(spec['path'],device='cuda')
                assert model.get_env() is None and model.device.type=='cuda'
                model.policy.set_training_mode(False)
                models[label]=model
                snapshots[label]=(copy.deepcopy(model.policy.state_dict()),copy.deepcopy(model.policy.optimizer.state_dict()))
                counts[label]=(model.num_timesteps,model._n_updates)
        rng=(torch.get_rng_state().clone(),[s.clone() for s in torch.cuda.get_rng_state_all()])
        computed=[]
        for index in pending:
            unchanged();arrays=load_chunk(dataset/f'chunk_{index:04d}')
            states=arrays['obs_state'].reshape(-1,237);features=arrays['obs_candidate'].reshape(-1,1840,16)
            masks=arrays['action_masks'].reshape(-1,1840).astype(bool);outputs=[]
            for start in range(0,len(states),design['batch_size']):
                end=min(start+design['batch_size'],len(states))
                obs=dict(state=torch.as_tensor(states[start:end],device='cuda'),
                         candidate=torch.as_tensor(features[start:end],device='cuda'))
                observed={name:forward(model.policy,obs,masks[start:end]) for name,model in models.items()}
                outputs.append(compare_forward(observed['reference'],observed['candidate'],masks[start:end],features[start:end]))
            rows={name:np.concatenate([o[name] for o in outputs]) for name in outputs[0]}
            rows.update(worker=np.tile(np.arange(8,dtype=np.int16),512),
                global_step=np.repeat(np.arange((index-1)*512,index*512,dtype=np.int32),8),
                chunk=np.full(4096,index,dtype=np.int16),episode_seed=arrays['info_episode_seed'].reshape(-1),
                terminal_distance=distance[(index-1)*512:index*512].reshape(-1))
            assert torch.equal(rng[0],torch.get_rng_state()) and identical(rng[1],torch.cuda.get_rng_state_all())
            audit={}
            for label,model in models.items():
                assert identical(snapshots[label][0],model.policy.state_dict())
                assert identical(snapshots[label][1],model.policy.optimizer.state_dict())
                assert counts[label]==(model.num_timesteps,model._n_updates)
                audit[label]=dict(timesteps=counts[label][0],ppo_updates=counts[label][1],policy_adam_counts_unchanged=True)
            unchanged();commit_cache(folder/f'rows_{index:04d}',dict(identity,chunk=index),rows,audit)
            caches[index]=rows;computed.append(index)
            durable_json(folder/'rank_progress.json',dict(at=now(),pid=os.getpid(),completed=[i for i in caches if caches[i] is not None],computed_this_process=computed))
            del arrays,states,features,masks
        unchanged()
        complete=all(v is not None for v in caches.values())
        receipt=dict(at=now(),pid=os.getpid(),computed_chunks=computed,reused_chunks=[i for i in range(1,through_chunk+1) if i not in computed],
                     policy_optimizer_counts_unchanged=True,cpu_cuda_rng_unchanged=True,environment_steps=0,optimizer_steps=0)
        durable_json(folder/f'process_{os.getpid()}.json',receipt)
        if complete:
            all_rows={name:np.concatenate([caches[i][name] for i in range(1,5)]) for name in caches[1]}
            bins=[*design['margin_bin_edges'],float('inf')]
            durable_json(folder/'rank_summary.json',dict(at=now(),status='passed',identity=identity,samples=16384,
                diagnostic=summarize(all_rows,bins),process=receipt,models=design['models'],model_promotion='none',
                interpretation='Same saved states from frozen reference-policy sampling. Descriptive ranking shift; no trajectory causality or quality improvement claim.'))
        print(json.dumps(dict(receipt,complete=complete)))
    finally:lock.close()


def verify_restart(folder):
    """Two actual processes: commit 2 chunks, then reuse them and finish 3/4."""
    from resource_guard import now
    if (folder/'restart_receipt.json').exists(): return
    def execute(count):
        command=[sys.executable,str(Path(__file__).resolve()),'--run-dir',str(folder),
                 '--through-chunk',str(count)]
        process=subprocess.Popen(command)
        code=process.wait()
        if code:raise subprocess.CalledProcessError(code,command)
        return process.pid
    first_pid=execute(2)
    before={str(p):sha(p) for i in (1,2) for p in (folder/f'rows_{i:04d}').iterdir() if p.is_file()}
    second_pid=execute(4)
    result=json.loads((folder/'rank_summary.json').read_text())
    assert result['process']['reused_chunks']==[1,2] and result['process']['computed_chunks']==[3,4]
    assert first_pid!=second_pid and result['process']['pid']==second_pid
    assert all(sha(p)==digest for p,digest in before.items())
    durable_json(folder/'restart_receipt.json',dict(at=now(),status='passed',committed_first_two_chunk_files_unchanged=True,
        second_process_reused=[1,2],second_process_computed=[3,4],first_pid=first_pid,second_pid=second_pid))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--through-chunk',type=int,choices=[1,2,3,4],default=4)
    parser.add_argument('--verify-restart',action='store_true')
    args=parser.parse_args()
    if args.verify_restart:verify_restart(args.run_dir.resolve())
    else:run(args.run_dir.resolve(),args.through_chunk)
