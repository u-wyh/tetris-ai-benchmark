import copy
import pickle

import numpy as np
import pytest
from sb3_contrib import MaskablePPO

from training.public_lookahead import PublicSimulator,PublicLookaheadController
from training.tetris_core.core import TetrisCore
from training.evaluation.fast_legal import get_legal_placements
from training.evaluation.fast_candidate_features import candidate_features
from training.env.tetris_env import TetrisEnv
from training.bc_sequence_safety import STUDENT
from training.evaluation.hybrid_queue import BC

def observation(public):
    placements=get_legal_placements(public);features,mask=candidate_features(public,placements)
    return dict(state=TetrisEnv._encode_observation(public),candidate=features),mask

def test_public_simulation_exact_official_transition_including_hold():
    core=TetrisCore(53)
    for turn in range(18):
        public=core.get_public_observation();choices=core.get_legal_placements()
        #Exercise empty Hold, occupied Hold, rotations, differing path scores and unknown boundaries.
        for p in choices[::max(1,len(choices)//5)]:
            real=copy.deepcopy(core);sim=PublicSimulator(public);real.step(p['actionId']);sim.step(p['actionId'])
            actual=real.get_public_observation();pred=sim.get_public_observation()
            for key in ('board','score','lines','level','hold','holdAvailable','currentPiece','gameOver','phase'):
                assert pred[key]==actual[key],key
            assert actual['next'][:len(pred['next'])]==pred['next']
            assert not hasattr(sim,'bag') and not hasattr(sim,'rng')
        action=choices[(turn*7)%len(choices)]['actionId'];core.step(action)
        if core.phase=='over':break

def test_unknown_piece_is_a_frontier_not_a_fictitious_piece():
    sim=PublicSimulator(TetrisCore(53).get_public_observation())
    count=0
    while sim.phase=='playing':
        choices=sim.get_legal_placements();sim.step(choices[0]['actionId']);count+=1
    assert sim.phase=='unknown_frontier' and not sim.get_public_observation()['gameOver']
    assert sim.current is None and sim.queue==[] and count<=4
    with pytest.raises(ValueError):PublicSimulator(dict(TetrisCore(53).get_public_observation(),next=list('IJLOS')))

def test_hidden_rng_bag_tail_cannot_change_decision_or_mutate_core():
    import torch
    torch.set_num_threads(1)
    student=MaskablePPO.load(STUDENT,device='cpu');bc=MaskablePPO.load(BC,device='cpu')
    controller=PublicLookaheadController(student,bc);core=TetrisCore(53);other=copy.deepcopy(core)
    other.rng.state=781;other.bag=list('ZZZZZZZ');other.queue[3:]=list('II')
    assert core.get_public_observation()==other.get_public_observation()
    before=pickle.dumps(core);obs,mask=observation(core.get_public_observation())
    a,trace=controller.decide(obs,mask,core.get_public_observation())
    b,trace2=controller.decide(obs,mask,other.get_public_observation())
    assert (a,trace)==(b,trace2) and mask[a]
    assert pickle.dumps(core)==before
    assert all(len(r['path'])<=4 and r['known_tokens_consumed']<=3 for r in trace['alternatives'])
    #The controller rebuilds policy observations from public fields, ignoring injected inputs.
    assert controller.decide(dict(state=np.zeros(237),candidate=np.zeros((1840,16))),mask,core.get_public_observation())==(a,trace)

def test_known_blockout_and_lockout_are_not_unknown_frontier():
    public=TetrisCore(53).get_public_observation();sim=PublicSimulator(public)
    sim.board[0]=['X']*10;sim.queue=['O'];sim._spawn()
    assert sim.phase=='over' and sim.get_public_observation()['gameOver']
