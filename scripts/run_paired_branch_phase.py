"""Execute one frozen, live-guarded paired GPU phase; inspect before any retry."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paired_branch_capture import (prepare, boundary, specs, phase_plan, install, advance,
    validate_snapshot, stored_boundary, PairedNativeEnv, sha, identical, observation_digest)
from own_policy_capture import save_commit
from training.evaluation.evaluate_resumable import durable_json
from paired_branch_guard import verify_live_guard, require_authorization
from resource_guard import now


def run(attempt):
    request = json.loads((attempt/'request.json').read_text())
    folder = Path(request['job_folder']).resolve()
    if (request.get('version') != 'paired-phase-v1' or request.get('mode') not in ('probe', 'capture') or
            sha(folder/'design.json') != request['job_design_sha256']):
        raise ValueError('Paired phase request/schema or job seal differs')
    request_sha = sha(attempt/'request.json')
    lock = (attempt/'attempt.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    job_lock = None;env = None
    work = dict(environment_steps=0, gpu_forward_calls=0, optimizer_steps=0, model_loads=0)
    receipt = dict(at=now(), pid=os.getpid(), request_sha256=request_sha, status='started')
    computing = False
    try:
        job = json.loads((folder/'design.json').read_text())
        require_authorization(job, attempt)
        if (attempt/'receipt.json').exists():
            cached = json.loads((attempt/'receipt.json').read_text())
            if cached['request_sha256'] != request_sha or cached['status'] not in ('completed', 'probe_passed', 'terminal_reused'):
                raise ValueError('Existing attempt did not finish; inspect without blind retry')
            _, _, snapshot, previous, count = prepare(folder)
            end = cached['end']
            if end['chunks'] and (count < end['chunks'] or
                    sha(folder/f"chunk_{end['chunks']-1:04d}"/'manifest.json') != end['manifest_sha256']):
                raise ValueError('Completed attempt committed prefix changed')
            reused = dict(at=now(), pid=os.getpid(), status='reused', original_pid=cached['pid'],
                          original_receipt_sha256=sha(attempt/'receipt.json'), work=work)
            durable_json(attempt/f'reuse_{os.getpid()}.json', reused)
            print(json.dumps(reused));return
        if (attempt/'started.json').exists():
            raise RuntimeError('Started attempt has no final receipt; audit actual cache, never retry blindly')
        proof = verify_live_guard(request, attempt, job)
        job_lock = (folder/'capture.lock').open('a');fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stored_boundary(folder) != request['expected_start']:
            raise ValueError('Requested paired checkpoint boundary is stale before pickle')
        design, identity, snapshot, previous, count = prepare(folder)
        start = boundary(snapshot, previous, count)
        if start != request['expected_start']:
            raise ValueError('Requested paired checkpoint boundary is stale')
        if request['mode'] == 'probe' and snapshot is not None:
            raise ValueError('Initialization probe requires a fresh job')
        plan = phase_plan(snapshot, request['max_chunks'])
        receipt.update(start=start, end=start, guard_proof=proof, plan=plan)
        durable_json(attempt/'started.json', {**receipt, 'work':None,
            'scope':'Actual work unknown until final receipt; after interruption audit cache before new request'})
        if request['mode'] == 'capture' and plan['chunks'] == 0:
            receipt.update(status='terminal_reused', work=work, at=now())
            durable_json(attempt/'receipt.json', receipt);return
        import torch
        from sb3_contrib import MaskablePPO
        from training.vector_env import make_vector_env
        from training.train_transaction import global_rng_state
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = None
        if request['mode'] == 'capture':
            model = MaskablePPO.load(design['model']['path'], env=None, device='cuda')
            work['model_loads'] = 1
            if model.get_env() is not None or model.device.type != 'cuda' or [model.num_timesteps, model._n_updates] != [30003200, 36625]:
                raise ValueError('Paired original CUDA model counts differ')
            model.policy.set_training_mode(False)
            frozen_model = copy.deepcopy((model.policy.state_dict(), model.policy.optimizer.state_dict(),
                                          model.num_timesteps, model._n_updates))
        env = make_vector_env(8, base_seed=0, max_pieces=100000, env_class=PairedNativeEnv,
            env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
        # The common cap can exceed100K only under the already frozen historical cap rule.
        for lane, spec in enumerate(specs(design['cases'])):
            cap = max(100000, spec['case']['step']+5001)
            env.set_attr('max_pieces', cap, indices=[lane])
        views = install(env, design, snapshot)
        if request['mode'] == 'probe':
            workers = env.env_method('paired_export')
            receipt.update(status='probe_passed', work=work, at=now(),
                source_full_state_restored=True, paired_full_core_rng_equal=True,
                schema_normalization='Legacy raw reward-v1 missing reward_profile maps to verified default metadata only; no RNG synthesis',
                branch_ids=[w['branch_id'] for w in workers],
                public_digests=[None if v is None else observation_digest(**v) for v in views])
            durable_json(attempt/'receipt.json', receipt);return
        rng = copy.deepcopy(global_rng_state(model))
        def audit():
            if (not identical(frozen_model, (model.policy.state_dict(), model.policy.optimizer.state_dict(),
                                             model.num_timesteps, model._n_updates)) or
                    not identical(rng, global_rng_state(model))):
                raise ValueError('Frozen original policy/Adam/counts or full sampling RNG changed')
            return dict(policy_adam_counts_rng_unchanged=True, model_sha256=design['model']['sha256'],
                        model_counts=[30003200,36625], optimizer_steps=0)
        def capture(decision_round, steps):
            return dict(round=decision_round, environment_steps=steps, workers=env.env_method('paired_export'),
                        views=[None if v is None else observation_digest(**v) for v in views])
        if snapshot is None:
            snapshot = capture(0,0);validate_snapshot(snapshot,design)
            previous = save_commit(folder,0,identity,None,snapshot,[],audit());count = 1
        ordered = specs(design['cases'])
        for _ in range(plan['chunks']):
            require_authorization(design, attempt)
            if not any(v is not None for v in views):break
            rows = [];stop = min(5000,snapshot['round']+256)
            decision_round = snapshot['round'];steps = snapshot['environment_steps']
            while decision_round < stop and any(v is not None for v in views):
                require_authorization(design, attempt)
                upper = sum(v is not None for v in views)
                if steps+upper > 40000 or work['environment_steps']+upper > 4096:
                    raise RuntimeError('Paired job/phase environment budget exhausted')
                computing = True
                new_rows = advance(env,views,model.policy,ordered,decision_round)
                computing = False
                work['gpu_forward_calls'] += 1;work['environment_steps'] += len(new_rows)
                rows.extend(new_rows);steps += len(new_rows);decision_round += 1
            snapshot = capture(decision_round, steps);validate_snapshot(snapshot,design)
            previous = save_commit(folder,count,identity,previous,snapshot,rows,audit());count += 1
            receipt['end'] = boundary(snapshot,previous,count)
        for path, expected in design['frozen_files'].items():
            if sha(path) != expected:raise ValueError('Frozen paired input changed during phase: '+path)
        receipt.update(status='completed', at=now(), work=work, source_audit=audit(),
            terminal=not any(w['active'] for w in snapshot['workers']))
        durable_json(attempt/'receipt.json',receipt)
        print(json.dumps(receipt))
    except BaseException as exc:
        if (attempt/'started.json').exists() and not (attempt/'receipt.json').exists():
            receipt.update(status='failed',at=now(),error=repr(exc),
                work=None if computing else work,
                counts_exact=not computing, scope='Committed prefix retained; inspect before a new boundary request')
            durable_json(attempt/'receipt.json',receipt)
        raise
    finally:
        if env is not None:env.close()
        if job_lock is not None:job_lock.close()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-dir',required=True,type=Path)
    run(parser.parse_args().attempt_dir.resolve())
