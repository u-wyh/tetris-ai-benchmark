"""Five fixed ablations; production planner and its public simulator stay untouched."""
import numpy as np
from training.public_lookahead import PublicSimulator, geometry, placement_cost
from training.evaluation.failure_curriculum_eval import fast_action
from training.evaluation.fast_candidate_features import candidate_features
from training.env.tetris_env import TetrisEnv

SPECS = {'A': (1, 0), 'B': (2, 1), 'C': (2, 4), 'D': (4, 1), 'E': (4, 4)}
LEGACY_BRANCH_KEYS = ('action','value','terminal','path','known_tokens_consumed','remaining_known','legal_counts','final_risk','reward_normalized')

def legacy_trace(trace):
    def branch(row): return {k: row[k] for k in LEGACY_BRANCH_KEYS}
    return {**{k:trace[k] for k in ('student_action','bc_action','override','reason','bounded_search')},
            'selected':branch(trace['selected']),'alternatives':[branch(r) for r in trace['alternatives']]}

class AblationController:
    def __init__(self, student, bc, variant):
        if variant not in SPECS: raise ValueError('Only A/B/C/D/E are frozen')
        self.student,self.bc,self.variant=student,bc,variant

    def decide(self, observation, mask, public):
        if len(public['next']) != 3: raise ValueError('Exactly public Next3 is required')
        if self.variant == 'A':
            action=fast_action(self.student,observation,mask)
            return action, dict(student_action=action,bc_action=-1,geometry_action=-1,clear_action=-1,
                nominal_depth=0,roots=[action],selected=None,alternatives=[],override=False,
                reason='student_only',bounded_search=False)
        count,depth=SPECS[self.variant]
        sim=PublicSimulator(public);legal=sim.get_legal_placements()
        ids={p['actionId'] for p in legal}
        if ids != set(np.flatnonzero(mask)): raise RuntimeError('Official legal mask differs')
        features,_=candidate_features(public,legal)
        observation=dict(state=TetrisEnv._encode_observation(public),candidate=features)
        student=fast_action(self.student,observation,mask);bc=fast_action(self.bc,observation,mask)
        estimates={p['actionId']:placement_cost(public,p) for p in legal}
        safe=min(ids,key=lambda a:(estimates[a][0],a))
        score=max(ids,key=lambda a:(int(observation['candidate'][a,5]),-estimates[a][0],-a))
        roots=list(dict.fromkeys((student,bc) if count==2 else (student,bc,safe,score)))
        results=[]
        for action in roots:
            node=sim.clone();node._placements=legal;node.step(action)
            reward=(node.score-public['score'])/max(100,100*public['level']);max_cost,risk=geometry(node.board)
            immediate_reward=reward;immediate_geometry=max_cost
            path=[action];legal_counts=[];terminal=node.phase
            for _ in range(1,depth):
                if node.phase!='playing':break
                options=node.get_legal_placements();legal_counts.append(len(options))
                if not options:terminal='no_mechanical_action';break
                pub=node.get_public_observation()
                best=min(options,key=lambda p:(placement_cost(pub,p)[0],p['actionId']))
                before=node.score;node.step(best['actionId']);path.append(best['actionId'])
                reward+=(node.score-before)/max(100,100*public['level'])
                cost,risk=geometry(node.board);max_cost=max(max_cost,cost);terminal=node.phase
            final_cost,risk=geometry(node.board)
            value=final_cost+.25*max_cost-2*reward
            if terminal in ('over','no_mechanical_action'):value+=1000000
            results.append(dict(action=action,value=value,terminal=terminal,path=path,
                known_tokens_consumed=node.consumed,remaining_known=len(node.queue),legal_counts=legal_counts,
                final_risk=risk,reward_normalized=reward,final_geometry=final_cost,max_geometry=max_cost,
                score_contribution=-2*reward,immediate_reward=immediate_reward,
                immediate_geometry=immediate_geometry,immediate_proxy=estimates[action][0],
                death_penalty=1000000 if terminal in ('over','no_mechanical_action') else 0,
                stop_reason='depth_limit' if terminal=='playing' else terminal))
        preferred=results[0];best=min(results,key=lambda r:(r['value'],roots.index(r['action'])))
        selected=best if best['value']+12<preferred['value'] else preferred
        return selected['action'],dict(student_action=student,bc_action=bc,geometry_action=safe,clear_action=score,
            nominal_depth=depth,roots=roots,selected=selected,alternatives=results,
            override=selected['action']!=student,
            reason='public_rollout_advantage' if selected['action']!=student else 'retain_student_within_margin',bounded_search=True)
