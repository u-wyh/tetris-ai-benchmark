import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from training.candidate_policy import CandidateScoringPolicy
from training.env.fast_candidate_env import FastCandidateTetrisEnv
from training.kl_maskable_ppo import KLMaskablePPO


def test_score_reward_and_snapshot_preserve_official_game():
    a = FastCandidateTetrisEnv(max_pieces=100, reward_profile="bc_score_v1")
    b = FastCandidateTetrisEnv(max_pieces=100)
    a.reset(seed=930); b.reset(seed=930)
    for index in range(40):
        if index == 10:
            a.core.lines = b.core.lines = 10
            a.core.level = b.core.level = 2
        act = int(np.flatnonzero(a.action_masks())[0])
        score, level = a.core.score, a.core.level
        _, reward, over, cap, info = a.step(act)
        b.step(act)
        assert a.core.get_public_observation() == b.core.get_public_observation()
        assert reward == 0.02 + (info['score'] - score) / (100 * level) - 20 * over
        if over or cap:
            break
    c = FastCandidateTetrisEnv(max_pieces=100, reward_profile="bc_score_v1")
    c.reset(seed=999); c.set_state(a.get_state())
    assert c.core.get_public_observation() == a.core.get_public_observation()
    assert c.episode_reward == a.episode_reward
    a.close(); b.close(); c.close()


def test_critic_warmup_freezes_actor_then_releases_it(tmp_path):
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: FastCandidateTetrisEnv(reward_profile="bc_score_v1")])
    ref = MaskablePPO(CandidateScoringPolicy, env, n_steps=16, batch_size=16, n_epochs=1, seed=20)
    model = KLMaskablePPO(CandidateScoringPolicy, env, n_steps=16, batch_size=16,
                          n_epochs=1, seed=21, critic_warmup_steps=16)
    model.policy.load_state_dict(ref.policy.state_dict())
    model.set_reference(ref)
    initial = {k:v.clone() for k,v in model.policy.state_dict().items()}
    model.learn(16)
    assert all(torch.equal(v, initial[k]) for k,v in model.policy.state_dict().items()
               if not k.startswith('value_net.'))
    assert any(not torch.equal(v, initial[k]) for k,v in model.policy.state_dict().items()
               if k.startswith('value_net.'))
    model.save(tmp_path / 'warmup.zip')
    loaded = KLMaskablePPO.load(tmp_path / 'warmup.zip', env=env, force_reset=False)
    assert loaded.critic_warmup_steps == 16
    loaded.set_reference(ref)
    loaded.learn(16, reset_num_timesteps=False)
    assert any(not torch.equal(v, initial[k]) for k,v in loaded.policy.state_dict().items()
               if not k.startswith(('value_net.', 'candidate_scale')))
    assert all(torch.equal(v, initial[k]) for k,v in ref.policy.state_dict().items())
    env.close()


def test_checksum_and_common_window_exactness(tmp_path):
    import pytest
    import json
    from training.bc_score_rl import paired, save_record, load_record
    path = tmp_path / 'result.json'
    save_record(path, {'seed': 1, 'score': 10})
    assert load_record(path)['score'] == 10
    data = json.loads(path.read_text()); data['data']['score'] = 11
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match='checksum'):
        load_record(path)
    a = [{'seed':1,'pieces':3,'scores':[0,10,20,30]}]
    b = [{'seed':1,'pieces':2,'scores':[0,11,22]}]
    result = paired(a,b)
    assert result['pairs'][0]['common_pieces'] == 2
    assert result['common_score_ratio'] == 1.1
    assert result['pairs'][0]['survival_delta'] == -1


def test_gate_rejects_early_death_despite_score_gain():
    from training.bc_score_rl import gate
    base = dict(mean_pieces=100000, p10=100000, early=0, capped=16, games=16,
                high_fraction=0.0, mean_max_high_streak=0.0, new_holes_per_100=2.0)
    candidate = dict(base, early=1)
    result = {'summary':{'bc':base,'candidate':candidate},
              'paired_to_bc':{'candidate':dict(common_score_ratio=1.1,positive_seeds=16,ratio_ci95=[1.01,1.2])}}
    assert not gate(result,long=True,confirm=True)['passed']
    result['summary']['candidate']['early']=0
    assert gate(result,long=True,confirm=True)['passed']


def test_evaluation_checkpoint_resumes_exact_score_and_board_statistics(tmp_path):
    import hashlib, pickle
    import pytest
    import training.bc_score_rl as queue
    if not queue.BC.exists():
        pytest.skip('Frozen BC artifact unavailable')
    queue.eval_init({'bc':str(queue.BC)})
    identity={'smoke':True}
    uninterrupted=queue.eval_game('bc',93,64,identity,tmp_path/'missing.pkl')
    env=queue.LongCandidateTetrisEnv(max_pieces=64)
    env.reset(seed=93)
    state=dict(seed=93,variant='bc',cap=64,identity=identity,pieces=0,score=0,scores=[0],
               clears_0_to_4=[0]*5,new_holes=0,high_steps=0,high_streak=0,max_high_streak=0,
               inference_seconds=0.0,wall_seconds=0.0)
    for _ in range(16):
        action=queue.fast_action(queue.MODELS['bc'],env.get_observation(),env.action_masks())
        _,_,over,truncated,info=env.step(action)
        assert not over and not truncated
        state['pieces']=info['pieces'];state['score']=info['score'];state['scores'].append(info['score'])
        state['clears_0_to_4'][info['cleared_lines']]+=1;state['new_holes']+=info['new_holes']
        high=info['height_after']>=17
        state['high_steps']+=int(high)
        state['high_streak']=state['high_streak']+1 if high else 0
        state['max_high_streak']=max(state['max_high_streak'],state['high_streak'])
    payload=pickle.dumps({'state':state,'env':env.get_state(),'controller':None},protocol=5)
    checkpoint=tmp_path/'resume.pkl'
    queue.atomic_pickle(checkpoint,{'payload':payload,'sha256':hashlib.sha256(payload).hexdigest()})
    env.close()
    resumed=queue.eval_game('bc',93,64,identity,checkpoint)
    for name in ('pieces','score','scores','clears_0_to_4','new_holes','high_steps','max_high_streak',
                 'game_over','capped','lines','final_height','final_holes'):
        assert resumed[name]==uninterrupted[name]
