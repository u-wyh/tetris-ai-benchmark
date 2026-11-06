"""Two own-death contexts, fixed four arms each, frozen CUDA policies, 256 steps."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from own_policy_capture import sha
from own_death_factorial_helpers import run_factorial, SourceReproductionError, ARMS
from rank_branch_helpers import BranchNativeEnv, compare_pair
from restore_rank_contexts import load_context
from training.evaluation.evaluate_resumable import durable_json
from verify_feature_backend_benchmark import identical


def source_traces(capture, targets, identity, frozen):
    arrays = {}
    traces = []
    for target in targets:
        rows = []
        for number in range(target['source_chunk'], (target['death_step']+255)//256+1):
            if number not in arrays:
                directory = capture/f'chunk_{number:04d}'
                if sha(directory/'manifest.json') != frozen[str(directory/'manifest.json')]:
                    raise ValueError('Frozen source trace manifest differs')
                manifest = json.loads((directory/'manifest.json').read_text())
                if manifest['identity'] != identity or sha(directory/'rows.npz') != manifest['files']['rows.npz']:
                    raise ValueError('Source trace identity or array checksum differs')
                with np.load(directory/'rows.npz', allow_pickle=False) as archive:
                    arrays[number] = {key: archive[key] for key in archive.files}
            data = arrays[number]
            selected = np.flatnonzero((data['lane'] == target['lane']) &
                (data['pieces'] >= target['step']) & (data['pieces'] <= target['death_step']))
            for row in selected:
                outcome = {key: data[key][row].item() for key in
                    ['action','reward','terminated','truncated','cleared_lines','holes_after','height_after','pieces','score']}
                rows.append(dict(action=int(data['action'][row]), features=data['candidate_features'][row].copy(),
                    digest=dict(observation=str(data['observation_sha256'][row]), mask=str(data['mask_sha256'][row])),
                    outcome=outcome))
        expected = target['actions_to_observed_death_including_selected']
        if (len(rows) != expected or
                [row['outcome']['pieces'] for row in rows] != list(range(target['step'], target['death_step']+1)) or
                not rows[-1]['outcome']['terminated'] or any(row['outcome']['truncated'] for row in rows) or
                any(row['outcome']['terminated'] for row in rows[:-1])):
            raise ValueError('Original source death trace is not complete and contiguous')
        traces.append(rows)
    return traces


def comparisons(outcomes):
    return {name: compare_pair(dict(reference=outcomes[left], candidate=outcomes[right]))
            for name, left, right in [('first_action_with_reference','A','B'),
                ('first_action_with_candidate','C','D'),('continuation_with_shadow_first','A','C'),
                ('continuation_with_candidate_first','B','D')]}


def read_group(directory, identity):
    if not directory.exists():
        return None
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest['identity'] != identity or set(manifest['files']) != {'results.json','states.pkl'} or
            any(sha(directory/name) != expected for name, expected in manifest['files'].items())):
        raise ValueError('Committed factorial group identity or checksum differs')
    result = json.loads((directory/'results.json').read_text())
    # Own group, hash-verified before pickle; all eight lanes form one commit.
    states = pickle.loads((directory/'states.pkl').read_bytes())
    return result, states


def save_group(directory, identity, results, states):
    cached = read_group(directory, identity)
    if cached is not None:
        if cached[0] != results or not identical(cached[1], states):
            raise ValueError('Whole fixed-group replay differs from committed factorial outputs')
        return
    pending = directory.with_name(directory.name+'.pending')
    if pending.exists():
        abandoned = directory.parent/'abandoned'; abandoned.mkdir(exist_ok=True)
        pending.rename(abandoned/(pending.name+'.'+str(os.getpid())))
    pending.mkdir()
    durable_json(pending/'results.json', results)
    with (pending/'states.pkl').open('xb') as stream:
        pickle.dump(states, stream, protocol=5); stream.flush(); os.fsync(stream.fileno())
    durable_json(pending/'manifest.json', dict(identity=identity,
        files={name:sha(pending/name) for name in ['results.json','states.pkl']}))
    pending.rename(directory)
    descriptor = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(folder):
    import torch
    from sb3_contrib import MaskablePPO
    from training.train_vector_transaction import training_source_identity
    from training.train_transaction import global_rng_state
    from training.vector_env import make_vector_env
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    from resource_guard import now
    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for path, expected in design['frozen_files'].items():
            if sha(path) != expected:
                raise ValueError('Frozen factorial input changed: '+path)
    unchanged()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no factorial host/CPU fallback')
    if (design['horizon'] != 256 or design['n_envs'] != 8 or
            training_source_identity() != design['training_sources'] or
            inference_source_identity(resumable=True) != design['inference_sources'] or
            evaluation_runtime_identity(False) != design['evaluation_runtime']):
        raise ValueError('Factorial bounded protocol or numerical runtime differs')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    context_dir = Path(design['contexts_dir'])
    contexts = json.loads((context_dir/'contexts_summary.json').read_text())
    targets = contexts['targets']
    if [target['seed'] for target in targets] != [100002,100005]:
        raise ValueError('Both original true deaths must be included in fixed order')
    identity = dict(design_sha256=sha(folder/'design.json'), horizon=256,
        lanes='100002:A/B/C/D then 100005:A/B/C/D; each policy active lanes ordered',
        precision='highest', tf32=False, models=design['models'])
    lock = (folder/'factorial.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = None
    try:
        cached = read_group(folder/'group', identity)
        if cached is None:
            selection = json.loads(Path(design['targets_path']).read_text())
            traces = source_traces(Path(selection['capture_dir']), targets, selection['capture_identity'], design['frozen_files'])
            specifications = []
            for target in targets:
                expected = dict(contexts['identity'], target=target)
                state, meta = load_context(context_dir/'contexts'/target['key'], expected)
                if state['max_pieces'] != 20000 or meta['observation_digest'] != target['observation_digest']:
                    raise ValueError('Recovered context limits or pre-action digest differs')
                specifications.append(dict(key=target['key'], state=state, digest=meta['observation_digest'],
                    actions=[target['shadow_action'],target['candidate_action']], expected_first_clears=[0,0]))
            models = {}
            for name, spec in design['models'].items():
                if sha(spec['path']) != spec['sha256']:
                    raise ValueError('Frozen factorial model changed')
                model = MaskablePPO.load(spec['path'], device='cuda')
                if (model.get_env() is not None or model.device.type != 'cuda' or
                        [model.num_timesteps,model._n_updates] != spec['counts']):
                    raise ValueError('Factorial model source counters or device differ')
                model.policy.set_training_mode(False); models[name] = model
            frozen = {name:(copy.deepcopy(m.policy.state_dict()), copy.deepcopy(m.policy.optimizer.state_dict()),
                       m.num_timesteps,m._n_updates) for name,m in models.items()}
            env = make_vector_env(8, max_pieces=20000, env_class=BranchNativeEnv,
                env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
            rng = global_rng_state(models['candidate'])
            def model_audit():
                for name, model in models.items():
                    weights, adam, steps, updates = frozen[name]
                    if (not identical(weights, model.policy.state_dict()) or
                            not identical(adam, model.policy.optimizer.state_dict()) or
                            (steps,updates) != (model.num_timesteps,model._n_updates)):
                        raise ValueError('Frozen factorial model, Adam or source counters changed')
                if not identical(rng, global_rng_state(models['candidate'])):
                    raise ValueError('Greedy factorial changed global CPU/CUDA sampling RNG')
                return dict(policy_adam_counts_unchanged=True, global_sampling_rng_unchanged=True,
                    model_counts={name:[m.num_timesteps,m._n_updates] for name,m in models.items()})
            try:
                outcomes, work = run_factorial(env, specifications, models, 256, traces)
            except SourceReproductionError as error:
                durable_json(folder/'source_reproduction_failure.json', dict(at=now(),status='failed',
                    identity=identity,event=error.event,actual_work=error.work,model_audit=model_audit(),
                    interpretation='New numerical/batch protocol did not reproduce original D; no causal claim.'))
                raise
            audit = model_audit(); unchanged()
            if work['environment_steps'] > 2048 or not work['source_D_reproduction_passed']:
                raise ValueError('Factorial scope or source reproduction failed')
            cases = []; states = {}
            for target, outcome in zip(targets,outcomes):
                cases.append(dict(target=target, arms={arm:dict(summary=outcome[arm]['summary'],
                    trace=outcome[arm]['trace']) for arm in ARMS}, comparisons=comparisons(outcome)))
                states[target['key']] = {arm:outcome[arm]['state'] for arm in ARMS}
            result = dict(identity=identity,cases=cases,actual_work=work,model_audit=audit,pid=os.getpid())
            save_group(folder/'group', identity, result, states)
        else:
            result, _ = cached
        unchanged()
        durable_json(folder/'factorial_summary.json', dict(at=now(),status='passed',identity=identity,
            cases=[dict(target=case['target'],arms={arm:case['arms'][arm]['summary'] for arm in ARMS},
                comparisons=case['comparisons']) for case in result['cases']],
            actual_work=result['actual_work'],model_audit=result['model_audit'],pid=result['pid'],
            group_manifest_sha256=sha(folder/'group/manifest.json'), model_promotion='none',
            interpretation='Two selected last-disagreement own-death states; 256 local placements only. '
                'D source paths reproduced exactly. Different first actions also change hold/piece/placement. '
                'Horizon-censored local outcomes do not establish whole-policy or new-seed quality.'))
        print(json.dumps(result['actual_work']))
    finally:
        if env is not None:
            env.close()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir.resolve())
