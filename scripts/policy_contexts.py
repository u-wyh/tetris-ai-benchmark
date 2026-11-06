"""Inputs and natural greedy stepping for resumable diagnostic context evaluations."""
import json
from pathlib import Path

import numpy as np

from own_policy_capture import sha,read_chain,greedy_details
from restore_rank_contexts import load_context,observation_digest
from verify_feature_backend_benchmark import identical


def prepare_run(folder):
    """Read-only preflight. Wrong inputs/cache identity fail before a lock or output is written."""
    design=json.loads((folder/'design.json').read_text())
    for path,expected in design['frozen_files'].items():
        if sha(path)!=expected:raise ValueError('Frozen context evaluation input changed: '+path)
    checkpoint=design['checkpoint'];suite_spec=design['suite']
    if sha(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Checkpoint checksum differs')
    if sha(suite_spec['path'])!=suite_spec['sha256']:raise ValueError('Suite checksum differs')
    suite=json.loads(Path(suite_spec['path']).read_text())
    entries=suite['contexts']
    if (suite['version']!=1 or suite['n_envs']!=8 or suite['horizon']!=512 or suite['max_pieces']!=20000 or
            len(entries)!=8 or len({entry['key'] for entry in entries})!=8):
        raise ValueError('Diagnostic suite fixed GPU8/512 context protocol differs')
    for entry in entries:
        directory=Path(entry['directory'])
        if (sha(directory/'manifest.json')!=entry['manifest_sha256'] or
                sha(directory/'state.pkl')!=entry['state_sha256']):
            raise ValueError('Suite context manifest or state checksum differs')
        metadata=json.loads((directory/'manifest.json').read_text())
        if (metadata['identity']!=entry['expected_identity'] or
                metadata['state_sha256']!=entry['state_sha256'] or
                metadata['observation_digest']!=entry['observation_digest']):
            raise ValueError('Suite context identity or observation differs')
    identity=dict(version=1,design_sha256=sha(folder/'design.json'),checkpoint_sha256=checkpoint['sha256'],
        suite_sha256=suite_spec['sha256'],chunk_rounds=256,horizon=512,device='cuda',precision='highest',tf32=False,
        ordered_context_keys=[entry['key'] for entry in entries],batching='only active lanes, original suite order; no forced action')
    snapshot,previous,count=read_chain(folder,identity)
    return design,suite,identity,snapshot,previous,count


def load_suite_states(suite):
    states=[]
    for entry in suite['contexts']:
        directory=Path(entry['directory'])
        if sha(directory/'manifest.json')!=entry['manifest_sha256'] or sha(directory/'state.pkl')!=entry['state_sha256']:
            raise ValueError('Frozen suite context changed before loading')
        state,metadata=load_context(directory,entry['expected_identity'])
        if (state['finished'] or state['max_pieces']!=suite['max_pieces'] or
                state['pieces']!=entry['initial_pieces'] or state['episode_seed']!=entry['episode_seed'] or
                metadata['observation_digest']!=entry['observation_digest']):
            raise ValueError('Suite initial episode or pre-action state differs')
        states.append(state)
    return states


def activate_states(env,states,expected_digests):
    if len(states)!=env.num_envs or len(expected_digests)!=env.num_envs:
        raise ValueError('Every evaluation lane requires its complete state')
    views=[]
    for lane,state in enumerate(states):
        saved=dict(state=state,active=True)
        view=env.env_method('capture_restore',saved,indices=[lane])[0]
        if (not identical(saved,env.env_method('capture_export',indices=[lane])[0]) or
                observation_digest(**view)!=expected_digests[lane]):
            raise ValueError('Evaluation initial clone, observation or mask differs')
        views.append(view)
    return views


def natural_round(env,views,policy,decision_round):
    active=[lane for lane,view in enumerate(views) if view is not None]
    if not active:return []
    observations={key:np.stack([views[lane]['observation'][key] for lane in active])
                  for key in views[active[0]]['observation']}
    masks=np.stack([views[lane]['mask'] for lane in active])
    decisions=greedy_details(policy,observations,masks)
    rows=[]
    for batch,lane in enumerate(active):
        action=int(decisions['action'][batch])
        digest=observation_digest(**views[lane])
        row=dict(round=decision_round,local_action_step=decision_round+1,lane=lane,action=action,
            observation_sha256=digest['observation'],mask_sha256=digest['mask'],
            selected_features=views[lane]['observation']['candidate'][action].copy(),
            greedy_probability=float(decisions['probability'][batch]),top2_logit_margin=float(decisions['margin'][batch]))
        if not views[lane]['mask'][action]:raise ValueError('Natural policy selected an illegal action')
        outcome=env.env_method('capture_step',action,indices=[lane])[0]
        if outcome is None:raise ValueError('Active context lane did not advance')
        row.update(outcome['row']);rows.append(row);views[lane]=outcome['view']
    return rows
