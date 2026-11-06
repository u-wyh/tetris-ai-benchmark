"""Diagnostic-only active lanes: stop at death/cap, never auto-reset a branch."""
import copy

import numpy as np

from training.train_vector_transaction import (TransactionCompactSearchCandidateEnv,
    TransactionNativeSearchCandidateEnv)


class BranchControls:
    def branch_activate(self,state=None):
        self._branch_active=False
        self._branch_trace=[]
        self._branch_initial=None
        if state is None:return None
        if state['finished']:raise ValueError('Cannot activate a finished context')
        self.set_state(copy.deepcopy(state))
        self._branch_initial=dict(lines=self.core.lines,score=self.core.score,pieces=self.pieces,
            episode_seed=self.episode_seed,episode_count=self.episode_count)
        self._branch_active=True
        return self.branch_view()

    def branch_view(self):
        if not getattr(self,'_branch_active',False):return None
        return dict(observation=self.get_observation(),mask=self.action_masks())

    def branch_step_active(self,action):
        if not getattr(self,'_branch_active',False):return None
        observation,reward,terminated,truncated,info=self.step(action)
        if (self.episode_seed!=self._branch_initial['episode_seed'] or
                self.episode_count!=self._branch_initial['episode_count']):
            raise RuntimeError('Branch unexpectedly reset its episode')
        row=dict(action=int(action),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),
            cleared_lines=int(info['cleared_lines']),holes_after=int(info['holes_after']),
            height_after=int(info['height_after']),pieces=int(info['pieces']),score=int(info['score']))
        self._branch_trace.append(row)
        if terminated or truncated:self._branch_active=False
        view=None if not self._branch_active else dict(observation=observation,mask=self.action_masks())
        return dict(row=row,view=view)

    def branch_finish(self,horizon):
        if self._branch_initial is None:return None
        self._branch_active=False
        steps=len(self._branch_trace)
        if not 1<=steps<=horizon:raise RuntimeError('Branch exceeded fixed horizon')
        last=self._branch_trace[-1]
        if steps<horizon and not (last['terminated'] or last['truncated']):
            raise RuntimeError('Branch stopped before terminal outcome or horizon')
        return dict(summary=dict(steps=steps,reward=sum(r['reward'] for r in self._branch_trace),
            cleared_lines=self.core.lines-self._branch_initial['lines'],score_delta=self.core.score-self._branch_initial['score'],
            game_over=last['terminated'],time_limit=last['truncated'],
            horizon_censored=bool(steps==horizon and not last['terminated'] and not last['truncated']),
            episode_seed=self.episode_seed,episode_count=self.episode_count,
            final_board=copy.deepcopy(self.core.board)),trace=self._branch_trace,state=self.get_state())


class BranchCompactEnv(BranchControls,TransactionCompactSearchCandidateEnv):
    pass


class BranchNativeEnv(BranchControls,TransactionNativeSearchCandidateEnv):
    pass


def run_group(env,specifications,policy,horizon):
    """Fixed group of <=4 pairs; GPU inference sees only active lanes in order."""
    from verify_feature_backend_benchmark import identical
    from restore_rank_contexts import observation_digest
    if not 1<=len(specifications)<=env.num_envs//2 or horizon<1:raise ValueError('Invalid branch group')
    views=[None]*env.num_envs
    for lane in range(env.num_envs):
        state=specifications[lane//2]['state'] if lane<2*len(specifications) else None
        views[lane]=env.env_method('branch_activate',state,indices=[lane])[0]
        if state is not None:
            if not identical(env.env_method('get_state',indices=[lane])[0],state):
                raise ValueError('Branch clone changed complete worker state')
            spec=specifications[lane//2]
            if observation_digest(**views[lane])!=spec['digest']:
                raise ValueError('Branch initial observation/mask differs')
    steps=forward_calls=0
    for time in range(horizon):
        active=[i for i,v in enumerate(views) if v is not None]
        if not active:break
        if time==0:
            actions=[specifications[i//2]['actions'][i%2] for i in active]
        else:
            observations={key:np.stack([views[i]['observation'][key] for i in active])
                          for key in views[active[0]]['observation']}
            masks=np.stack([views[i]['mask'] for i in active])
            predicted,_=policy.predict(observations,deterministic=True,action_masks=masks)
            actions=list(map(int,predicted));forward_calls+=1
        for lane,action in zip(active,actions):
            if not views[lane]['mask'][action]:raise ValueError('Branch selected an illegal action')
            result=env.env_method('branch_step_active',action,indices=[lane])[0]
            if result is None:raise RuntimeError('Active lane did not advance')
            if time==0:
                expected=specifications[lane//2].get('expected_first_clears')
                if expected is not None and result['row']['cleared_lines']!=expected[lane%2]:
                    raise ValueError('Forced first action cleared an unexpected number of rows')
            steps+=1;views[lane]=result['view']
    pairs=[]
    for pair in range(len(specifications)):
        outcomes=[env.env_method('branch_finish',horizon,indices=[2*pair+side])[0] for side in range(2)]
        pairs.append(dict(reference=outcomes[0],candidate=outcomes[1]))
    return pairs,dict(environment_steps=steps,policy_forward_calls=forward_calls)


def compare_pair(pair):
    left,right=pair['reference'],pair['candidate'];a,b=left['summary'],right['summary']
    common=min(a['steps'],b['steps'])
    def total(outcome,name):return sum(row[name] for row in outcome['trace'][:common])
    return dict(common_steps=common,common_reward_delta=total(right,'reward')-total(left,'reward'),
        common_cleared_delta=total(right,'cleared_lines')-total(left,'cleared_lines'),
        common_four_clear_delta=sum(r['cleared_lines']==4 for r in right['trace'][:common])-
            sum(r['cleared_lines']==4 for r in left['trace'][:common]),
        candidate_observed_earlier_death=bool(b['game_over'] and a['steps']>b['steps']),
        reference_observed_earlier_death=bool(a['game_over'] and b['steps']>a['steps']))
