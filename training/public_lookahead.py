"""A frozen-policy hybrid controller; bounded search of publicly known pieces only."""
import copy
from dataclasses import dataclass

import numpy as np

from training.tetris_core.core import TetrisCore
from training.tetris_core.pieces import make_piece
from training.evaluation.fast_legal import get_legal_placements
from training.candidate_features import board_features,candidate_board
from training.evaluation.failure_curriculum_eval import fast_action
from training.evaluation.fast_candidate_features import candidate_features
from training.env.tetris_env import TetrisEnv

@dataclass(frozen=True)
class PlanningConfig:
    depth:int=4
    roots:int=4
    override_margin:float=12.0

    def __post_init__(self):
        if (self.depth,self.roots,self.override_margin)!=(4,4,12.0):
            raise ValueError('Only one preregistered planning configuration is supported')

class PublicSimulator(TetrisCore):
    """Inherited official transition, with no hidden queue generation or RNG object."""
    def __init__(self,public):
        if len(public['next'])>3:raise ValueError('Only Next3 is public')
        self.board=copy.deepcopy(public['board']);self.current=copy.deepcopy(public['currentPiece'])
        self.hold=public['hold'];self.hold_used=not public['holdAvailable'];self.queue=list(public['next'])
        self.score=public['score'];self.lines=public['lines'];self.level=public['level'];self.phase=public['phase']
        self._placements=None;self.consumed=0

    def reset(self,*_args,**_kwargs):raise RuntimeError('Public simulator cannot initialize a hidden game')

    def _fill_queue(self):raise RuntimeError('Public simulator cannot generate unknown pieces')

    def _spawn(self,reset_hold=True):
        if reset_hold:self.hold_used=False
        if not self.queue:
            self.current=None;self.phase='unknown_frontier';return
        kind=self.queue.pop(0);self.consumed+=1;self._activate(kind)

    def get_legal_placements(self):
        if self._placements is None:
            public=self.get_public_observation()
            #Empty Hold requires the next token; it cannot be expanded at an unknown boundary.
            if not self.hold and not self.queue:public['holdAvailable']=False
            self._placements=get_legal_placements(public)
        return self._placements

    def step(self,action):
        result=super().step(action);self._placements=None;return result

    def clone(self):
        other=PublicSimulator(self.get_public_observation());other.consumed=self.consumed
        return other

def geometry(board):
    holes,aggregate,height,bump,wells=board_features(board)
    top=sum(board[y][x] is not None for y in range(4) for x in range(3,7))
    cost=8*holes+.12*aggregate+.15*bump+3*max(0,height-11)**2+120*top+max(0,wells-14)
    return cost,dict(holes=holes,height=height,central_top4=top,wells=wells)

def placement_cost(public,placement):
    board,cleared,lock_out=candidate_board(public['board'],placement['occupiedCells'])
    cost,risk=geometry(board)
    return cost-(0,1,3,5,8)[cleared]*2+1000000*lock_out,risk

class PublicLookaheadController:
    """Not a trained model and not a height-triggered expert switch."""
    def __init__(self,student,bc,config=PlanningConfig()):
        self.student,self.bc,self.config=student,bc,config

    def decide(self,observation,mask,public,placements=None):
        if len(public['next'])!=3:raise ValueError('A real decision must expose exactly Next3')
        sim=PublicSimulator(public);legal=sim.get_legal_placements()
        ids={p['actionId'] for p in legal}
        if ids!=set(np.flatnonzero(mask)):raise RuntimeError('Public legal set differs from official mask')
        features,_=candidate_features(public,legal)
        observation=dict(state=TetrisEnv._encode_observation(public),candidate=features)
        student=fast_action(self.student,observation,mask);bc=fast_action(self.bc,observation,mask)
        by_id={p['actionId']:p for p in legal}
        estimates={p['actionId']:placement_cost(public,p) for p in legal}
        safe=min(ids,key=lambda a:(estimates[a][0],a))
        score=max(ids,key=lambda a:(int(observation['candidate'][a,5]),-estimates[a][0],-a))
        roots=list(dict.fromkeys((student,bc,safe,score)))[:self.config.roots]
        results=[]
        for action in roots:
            node=sim.clone();node._placements=legal;node.step(action)
            reward=(node.score-public['score'])/max(100,100*public['level']);max_cost,risk=geometry(node.board)
            path=[action];legal_counts=[];terminal=node.phase
            for _ in range(1,self.config.depth):
                if node.phase!='playing':break
                options=node.get_legal_placements();legal_counts.append(len(options))
                if not options:terminal='no_mechanical_action';break
                pub=node.get_public_observation()
                #A deterministic width-one beam; unexpanded actions remain uncertainty, not infeasibility.
                best=min(options,key=lambda p:(placement_cost(pub,p)[0],p['actionId']))
                before=node.score;node.step(best['actionId']);path.append(best['actionId'])
                reward+=(node.score-before)/max(100,100*public['level'])
                cost,risk=geometry(node.board);max_cost=max(max_cost,cost);terminal=node.phase
            final_cost,risk=geometry(node.board)
            value=final_cost+.25*max_cost-2*reward
            if terminal in ('over','no_mechanical_action'):value+=1000000
            results.append(dict(action=action,value=value,terminal=terminal,path=path,known_tokens_consumed=node.consumed,remaining_known=len(node.queue),legal_counts=legal_counts,final_risk=risk,reward_normalized=reward))
        preferred=results[0];best=min(results,key=lambda r:(r['value'],roots.index(r['action'])))
        selected=best if best['value']+self.config.override_margin<preferred['value'] else preferred
        return selected['action'],dict(student_action=student,bc_action=bc,selected=selected,alternatives=results,override=selected['action']!=student,reason='public_rollout_advantage' if selected['action']!=student else 'retain_student_within_margin',bounded_search=True)
