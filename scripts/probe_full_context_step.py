"""Direct continuation reference and exact comparison for full-context rebasing."""

import argparse
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def direct(folder):
    import torch
    from stable_baselines3.common.logger import configure
    from training.train_vector_transaction import restore_task, TransactionCandidateEnv
    from training.vector_env import make_vector_env
    from training.train_transaction import global_rng_state

    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    source=folder/'source_context';config=json.loads((source/'config.json').read_text())
    assert config['device']=='cuda' and config['ppo']['n_envs']==8
    spec=json.loads((folder/'preset.json').read_text())['initial_context']
    env=make_vector_env(8,config['run_seed'],config['max_pieces'],env_class=TransactionCandidateEnv)
    try:
        model,_,_=restore_task(source,spec['task'],config,env)
        for number in (1,2):
            task=folder/'direct'/f'task_{number:06d}';task.mkdir(parents=True,exist_ok=False)
            model.set_logger(configure(str(task),[]))
            model.learn(total_timesteps=4096,reset_num_timesteps=False,use_masking=True)
            model.save(str(task/'model.zip'))
            (task/'trainer_state.pkl').write_bytes(pickle.dumps(global_rng_state(model),protocol=5))
            (task/'vector_env_state.pkl').write_bytes(pickle.dumps(env.env_method('get_state'),protocol=5))
            model.logger.close()
    finally:env.close()


def verify(folder):
    from sb3_contrib import MaskablePPO
    from training.train_vector_transaction import verify_task,verify_training_source_identity,sha256
    from training.train_ppo import atomic_json,now
    from verify_feature_backend_benchmark import identical

    training=folder/'training';config=json.loads((training/'config.json').read_text())
    state=json.loads((training/'transaction_state.json').read_text())
    verify_training_source_identity(json.loads((training/'metadata.json').read_text()))
    assert state['committed_steps']==8192 and state['pid'] is None
    offset=config['source_steps_offset'];records=[]
    for number in (1,2):
        reference=folder/'direct'/f'task_{number:06d}'
        candidate=training/'committed'/f'task_{number:06d}'
        meta=verify_task(candidate,number,config)
        a=MaskablePPO.load(str(reference/'model.zip'),device='cpu')
        b=MaskablePPO.load(str(candidate/'model.zip'),device='cpu')
        assert identical(a.policy.state_dict(),b.policy.state_dict()),'Policy differs'
        assert identical(a.policy.optimizer.state_dict(),b.policy.optimizer.state_dict()),'Adam differs'
        assert a._n_updates==b._n_updates,'PPO update count differs'
        trainers=[pickle.loads((p/'trainer_state.pkl').read_bytes()) for p in (reference,candidate)]
        assert trainers[0]['num_timesteps']==offset+trainers[1]['num_timesteps']
        trainers[0]['num_timesteps']-=offset
        assert identical(*trainers),'Rebased trainer and full RNG differ'
        workers=[pickle.loads((p/'vector_env_state.pkl').read_bytes()) for p in (reference,candidate)]
        assert identical(*workers),'Complete worker states differ'
        assert meta['lifetime_steps']==offset+number*4096
        records.append(dict(task=number,policy_adam_rng_worker_states_exact=True,
                            additional_steps=number*4096,lifetime_steps=meta['lifetime_steps']))
    source=folder/'source_context';original=ROOT/'runs/ppo_candidate_raw_30m_seed42'
    for relative,expected in json.loads((folder/'source_files.json').read_text()).items():
        assert sha256(source/relative)==sha256(original/relative)==expected,'Original source copy changed'
    result=dict(at=now(),status='passed',device='cuda',n_envs=8,source_steps_offset=offset,
                source_original_files_unchanged=True,tasks=records,
                interpretation='Current-code direct official continuation equals rebased bitboard continuation across process resume. Original source identity remains unproven; no policy-quality claim.',
                final_test_used=False,model_promotion='none')
    atomic_json(folder/'probe_summary.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['direct','verify'])
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();(direct if args.mode=='direct' else verify)(args.run_dir.resolve())
