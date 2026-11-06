"""Inspect complete endpoints and replay just three saved actions of seed100005."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from own_policy_capture import sha
from own_death_factorial_helpers import ARMS
from run_own_death_factorial import read_group
from rank_branch_helpers import BranchNativeEnv
from restore_rank_contexts import load_context, observation_digest
from restore_own_death_contexts import sampling_rng
from training.evaluation.evaluate_resumable import durable_json
from verify_feature_backend_benchmark import identical
from resource_guard import now


def run(folder):
    import torch
    from training.vector_env import make_vector_env
    from training.train_vector_transaction import training_source_identity
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    design = json.loads((folder/'design.json').read_text())
    def unchanged():
        for path, expected in design['frozen_files'].items():
            if sha(path) != expected:
                raise ValueError('Frozen convergence input changed: '+path)
    unchanged()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no convergence host/CPU fallback')
    if (training_source_identity() != design['training_sources'] or
            inference_source_identity(resumable=True) != design['inference_sources'] or
            evaluation_runtime_identity(False) != design['evaluation_runtime']):
        raise ValueError('Convergence numerical source/runtime differs')
    torch.set_num_threads(1)
    source = Path(design['factorial_dir'])
    summary = json.loads((source/'factorial_summary.json').read_text())
    if summary['status'] != 'passed' or sha(source/'group/manifest.json') != summary['group_manifest_sha256']:
        raise ValueError('Original complete factorial result required')
    data, endpoints = read_group(source/'group', summary['identity'])
    static = []
    for case in data['cases']:
        key = case['target']['key']; arms = case['arms']
        pairs = {left+'_'+right:identical(endpoints[key][left], endpoints[key][right])
                 for i,left in enumerate(ARMS) for right in ARMS[i+1:]}
        differences = [i+1 for i,(a,b) in enumerate(zip(arms['A']['trace'],arms['B']['trace']))
                       if a['action'] != b['action']]
        static.append(dict(seed=case['target']['seed'], complete_endpoint_state_pairs_exact=pairs,
            reference_vs_candidate_first_action_disagreement_steps=differences,
            B_D_complete_trace_exact=arms['B']['trace']==arms['D']['trace']))
    chosen = [case for case in data['cases'] if case['target']['seed'] == 100005]
    if len(chosen) != 1 or design['probe_saved_action_rounds'] != 3 or design['probe_max_environment_steps'] != 12:
        raise ValueError('Fixed seed100005 three-action convergence probe scope differs')
    case = chosen[0]; target = case['target']
    contexts = Path(design['contexts_dir'])
    context_summary = json.loads((contexts/'contexts_summary.json').read_text())
    state, meta = load_context(contexts/'contexts'/target['key'],dict(context_summary['identity'],target=target))
    env = make_vector_env(8,max_pieces=20000,env_class=BranchNativeEnv,
        env_kwargs=dict(native_feature_runtime=design['native_feature_runtime']))
    steps = 0; probe = []
    try:
        rng = sampling_rng()
        views = []
        for lane in range(8):
            view = env.env_method('branch_activate',state if lane < 4 else None,indices=[lane])[0]
            if lane < 4 and (observation_digest(**view) != meta['observation_digest'] or
                            not identical(env.env_method('get_state',indices=[lane])[0],state)):
                raise ValueError('Convergence initial state clone differs')
            views.append(view)
        for time in range(3):
            for lane, arm in enumerate(ARMS):
                row = case['arms'][arm]['trace'][time]
                if views[lane] is None or not views[lane]['mask'][row['action']]:
                    raise ValueError('Saved convergence action is illegal or lane is stopped')
                outcome = env.env_method('branch_step_active',row['action'],indices=[lane])[0]
                steps += 1
                if outcome['row'] != row:
                    raise ValueError('Three-step convergence replay differs from original factorial trace')
                views[lane] = outcome['view']
            states = [env.env_method('get_state',indices=[lane])[0] for lane in range(4)]
            pairs = {left+'_'+right:identical(states[i],states[j])
                     for i,left in enumerate(ARMS) for j,right in enumerate(ARMS) if j > i}
            digests = [None if view is None else observation_digest(**view) for view in views[:4]]
            probe.append(dict(after_saved_actions=time+1,complete_worker_state_pairs_exact=pairs,
                              all_four_active_view_digests_exact=all(digest == digests[0] for digest in digests)))
        if not identical(rng,sampling_rng()) or steps != 12:
            raise ValueError('Convergence parent RNG or actual step accounting differs')
        unchanged()
        durable_json(folder/'convergence_summary.json',dict(at=now(),status='passed',static_endpoints=static,
            probe_seed=100005,probe=probe,actual_replayed_environment_steps=steps,
            policy_forward_calls=0,optimizer_steps=0,parent_sampling_rng_unchanged=True,
            interpretation='Three fixed saved actions replayed on four cloned lanes, four padding lanes inactive. '
                'Exact full-state convergence is stronger than equal score or geometry summaries. '
                'No additional policy inference, intervention choice, training or global quality claim.'))
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    run(parser.parse_args().run_dir.resolve())
