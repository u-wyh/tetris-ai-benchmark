"""Sealed eight-lane paired interventions; frozen CUDA continuation, no resets."""
import copy
import gzip
import io
import json
import pickle
from pathlib import Path

import numpy as np

from own_policy_capture import CaptureNativeEnv, CaptureCompactEnv, sha, greedy_details
from restore_rank_contexts import observation_digest
from verify_feature_backend_benchmark import identical
from audit_topk_preference_evidence import sealed_bytes, public_hash

VERSION = 'paired-original30m-gpu8-v1'
SEEDS = [992000, 992004, 992006, 992008]


def selected_cases(preferences):
    chosen = []
    for seed in SEEDS:
        pool = [p for p in preferences['preferences'] if p['seed'] == seed and
                p['label'] == 'beneficial' and any(s.startswith('ppo_top') for s in p['sources'])]
        if not pool:
            raise ValueError('Missing predetermined training seed')
        pair = min(pool, key=lambda p: (p['step'], p['state_id'], p['action']))
        chosen.append({k: pair[k] for k in ('state_id', 'seed', 'step', 'action', 'baseline_action',
                                           'public_hash', 'snapshot_sha256')})
    return chosen


def specs(cases):
    return [dict(case=case, branch_id=f"s{case['seed']}_{role}", first_action=case[action],
                 role=role, horizon=5000)
            for case in cases for role, action in [('base', 'baseline_action'), ('alternative', 'action')]]


def load_start(case, design):
    """External and legacy hashes precede bounded decompression and unpickling."""
    source = Path(design['source_root']);frozen = design['frozen_files']
    prefix = source/'counterfactual_states'/f"seed_{case['seed']}_step_{case['step']}.pkl"
    meta = json.loads(sealed_bytes(prefix.with_suffix('.pkl.json'), frozen))
    payload = sealed_bytes(prefix.with_suffix('.pkl.gz'), frozen)
    if sha(prefix.with_suffix('.pkl.gz')) != meta['sha256'] or meta['sha256'] != case['snapshot_sha256']:
        raise ValueError('Legacy snapshot seal differs')
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        raw = stream.read(4*1024**2+1)
    if len(raw) > 4*1024**2:
        raise ValueError('Snapshot decompression bound exceeded')
    state = pickle.loads(raw)
    required = {'core', 'max_pieces', 'reward_version', 'reward_profile', 'hole_penalty_coef',
        'height_penalty_coef', 'episode_seed', 'episode_count', 'pieces', 'episode_reward', 'finished',
        'numpy_generator', 'action_generator', 'numpy_seed', 'candidate_feature_version'}
    # The sealed b382a44 legacy get_state predates reward_profile. Its raw v1
    # formula is identical to current default. Add only this documented metadata;
    # never manufacture missing gameplay/NumPy/action RNG or trajectory fields.
    if (set(state) == required-{'reward_profile'} and state['reward_version'] == 'placement-reward-v1' and
            state['hole_penalty_coef'] == 0. and state['height_penalty_coef'] == 0.):
        state['reward_profile'] = 'default'
    if (set(state) != required or state['episode_seed'] != case['seed'] or state['pieces'] != case['step'] or
            state['episode_count'] != 1 or state['finished'] or state['max_pieces'] != 100000 or
            meta['seed'] != case['seed'] or meta['step'] != case['step'] or
            state['core'].rng.state != meta['rng_state'] or
            'get_legal_placements' in state['core'].__dict__ or
            public_hash(state['core'].get_public_observation()) != case['public_hash'] or
            meta['public_hash'] != case['public_hash']):
        raise ValueError('Full source environment/core/RNG/metadata differs')
    # Same explicit cap rule as historical branches, with no cap inside the window.
    state['max_pieces'] = max(100000, case['step']+5001)
    return state


class PairedControls:
    def paired_install(self, spec, state, saved=None):
        self._paired_spec = copy.deepcopy(spec)
        self.set_state(copy.deepcopy(state))
        self._capture_seed = spec['case']['seed']
        self._paired_steps = self.pieces-spec['case']['step']
        self._capture_active = not self._finished and self._paired_steps < spec['horizon']
        if (self.episode_seed != self._capture_seed or self.episode_count != 1 or
                not 0 <= self._paired_steps <= spec['horizon'] or
                (saved is not None and not identical(saved, self.paired_export()))):
            raise ValueError('Paired restore differs from complete saved worker')
        return self.capture_view()

    def paired_export(self):
        return dict(state=self.get_state(), active=self._capture_active,
                    branch_id=self._paired_spec['branch_id'], steps=self._paired_steps)

    def paired_step_map(self, actions):
        if not self._capture_active:
            return None
        key = self._paired_spec['branch_id']
        if (type(actions) is not dict or key not in actions or type(actions[key]) is not int or
                not 0 <= actions[key] < 1840):
            raise ValueError('Missing or invalid independent branch action')
        action = actions[key]
        if self._paired_steps == 0 and action != self._paired_spec['first_action']:
            raise ValueError('First intervention differs')
        result = self.capture_step(action)
        self._paired_steps += 1
        if self._paired_steps == self._paired_spec['horizon']:
            self._capture_active = False
            result['view'] = None
        result['row'].update(branch_id=key, relative_step=self._paired_steps)
        return result


class PairedNativeEnv(PairedControls, CaptureNativeEnv):
    pass


class PairedCompactEnv(PairedControls, CaptureCompactEnv):
    pass


def validate_snapshot(snapshot, design):
    if snapshot is None:
        return
    ordered = specs(design['cases'])
    if (len(snapshot['workers']) != 8 or len(snapshot['views']) != 8 or
            not 0 <= snapshot['round'] <= 5000):
        raise ValueError('Paired snapshot lanes/round differ')
    total = 0
    for spec, worker, view in zip(ordered, snapshot['workers'], snapshot['views']):
        state = worker['state'];steps = worker['steps']
        active = not state['finished'] and steps < 5000
        if (worker['branch_id'] != spec['branch_id'] or type(worker['active']) is not bool or
                worker['active'] != active or (view is not None) != active or
                not 0 <= steps <= snapshot['round'] or (active and steps != snapshot['round']) or
                state['episode_seed'] != spec['case']['seed'] or state['episode_count'] != 1 or
                state['pieces'] != spec['case']['step']+steps or
                state['max_pieces'] != max(100000, spec['case']['step']+5001)):
            raise ValueError('Paired identity, full state, horizon or stopped lane differs')
        total += steps
    if total != snapshot['environment_steps'] or total > design['environment_step_limit']:
        raise ValueError('Paired unique-work accounting differs')


def prepare(folder):
    from training.train_vector_transaction import training_source_identity
    from training.evaluation.feature_backend import inference_source_identity, evaluation_runtime_identity
    from training.native_feature_runtime import load_runtime
    from own_policy_capture import read_chain
    design = json.loads((folder/'design.json').read_text())
    if (design.get('version') != VERSION or design.get('device') != 'cuda' or design.get('n_envs') != 8 or
            design.get('horizon') != 5000 or design.get('chunk_rounds') != 256 or
            design.get('environment_step_limit') != 40000 or design.get('phase_environment_step_limit') != 4096 or
            design['model']['counts'] != [30003200, 36625] or
            design['model']['sha256'] != 'eb859b8989efe8f3332393ed57110036979b03bb4d3d8d8316d080cf04b2da34'):
        raise ValueError('Fixed paired GPU8 design differs')
    for path, expected in design['frozen_files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen paired source/input changed: '+path)
    preferences = json.loads(sealed_bytes(Path(design['preferences']), design['frozen_files']))
    if selected_cases(preferences) != design['cases']:
        raise ValueError('Fixed pre-GPU pair selection differs')
    if (training_source_identity() != design['training_sources'] or
            inference_source_identity(resumable=True) != design['inference_sources'] or
            evaluation_runtime_identity(False) != design['evaluation_runtime']):
        raise ValueError('Paired numerical source/runtime differs')
    if sha(design['model']['path']) != design['model']['sha256']:
        raise ValueError('Paired original source model changed')
    load_runtime(design['native_feature_runtime'])
    identity = dict(version=VERSION, design_sha256=sha(folder/'design.json'), chunk_rounds=256,
        ordered_branches=[s['branch_id'] for s in specs(design['cases'])],
        device='cuda', precision='highest', tf32=False, batching='ordered active branches, no auto-reset')
    snapshot, previous, count = read_chain(folder, identity)
    validate_snapshot(snapshot, design)
    return design, identity, snapshot, previous, count


def boundary(snapshot, previous, count):
    return dict(manifest_sha256=previous, chunks=count, round=0 if snapshot is None else snapshot['round'],
                environment_steps=0 if snapshot is None else snapshot['environment_steps'])


def stored_boundary(folder):
    """Read only JSON/SHA before rejecting a stale request; never deserialize state."""
    chunks = sorted(folder.glob('chunk_[0-9][0-9][0-9][0-9]'))
    if not chunks:
        return boundary(None,None,0)
    if [p.name for p in chunks] != [f'chunk_{i:04d}' for i in range(len(chunks))]:
        raise ValueError('Paired checkpoint sequence differs')
    path = chunks[-1]/'manifest.json';manifest = json.loads(path.read_text())
    return dict(manifest_sha256=sha(path),chunks=len(chunks),round=manifest['round'],
                environment_steps=manifest['environment_steps'])


def phase_plan(snapshot, requested):
    if type(requested) is not int or not 1 <= requested <= 2:
        raise ValueError('Each phase allows only one or two new256-round chunks')
    active = 8 if snapshot is None else sum(w['active'] for w in snapshot['workers'])
    decision_round = 0 if snapshot is None else snapshot['round']
    return dict(chunks=min(requested, (5000-decision_round+255)//256) if active else 0,
                maximum_new_environment_steps=min(requested*256, 5000-decision_round)*active)


def install(env, design, snapshot):
    ordered = specs(design['cases']);views = []
    if snapshot is None:
        initial = [load_start(case, design) for case in design['cases']]
    for lane, spec in enumerate(ordered):
        saved = None if snapshot is None else snapshot['workers'][lane]
        state = initial[lane//2] if saved is None else saved['state']
        view = env.env_method('paired_install', spec, state, saved, indices=[lane])[0]
        exported = env.env_method('paired_export', indices=[lane])[0]
        if saved is None:
            if not identical(state, exported['state']):
                raise ValueError('Native restoration changed source full state/RNG')
            if not view['mask'][spec['first_action']]:
                raise ValueError('Fixed first intervention is illegal')
        else:
            if (None if view is None else observation_digest(**view)) != snapshot['views'][lane]:
                raise ValueError('Restored branch public input differs')
        views.append(view)
    workers = env.env_method('paired_export')
    if snapshot is None:
        for lane in range(0, 8, 2):
            if not identical(workers[lane]['state'], workers[lane+1]['state']) or not identical(views[lane], views[lane+1]):
                raise ValueError('Paired initial full core/RNG or public views differ')
    return views


def advance(env, views, policy, ordered, decision_round):
    active = [i for i, view in enumerate(views) if view is not None]
    if not active:
        return []
    observations = {k: np.stack([views[i]['observation'][k] for i in active]) for k in views[active[0]]['observation']}
    masks = np.stack([views[i]['mask'] for i in active])
    details = greedy_details(policy, observations, masks)
    actions = {};rows = []
    for batch, lane in enumerate(active):
        spec = ordered[lane];natural = int(details['action'][batch])
        if decision_round == 0 and natural != spec['case']['baseline_action']:
            raise ValueError('New GPU start mode differs from the sealed baseline action')
        action = spec['first_action'] if decision_round == 0 else natural
        if not masks[batch, action]:
            raise ValueError('Selected paired action is masked')
        actions[spec['branch_id']] = action
        digest = observation_digest(**views[lane])
        rows.append(dict(round=decision_round, lane=lane, action=action, natural_action=natural,
            forced_first=decision_round == 0, observation_sha256=digest['observation'],
            mask_sha256=digest['mask'], candidate_features=observations['candidate'][batch, action].copy()))
    outcomes = env.env_method('paired_step_map', actions, indices=active)
    if len(outcomes) != len(active):
        raise RuntimeError('Missing branch worker response')
    for lane, row, outcome in zip(active, rows, outcomes):
        if (outcome is None or outcome['row']['episode_seed'] != ordered[lane]['case']['seed'] or
                outcome['row']['branch_id'] != ordered[lane]['branch_id'] or
                outcome['row']['relative_step'] != decision_round+1):
            raise RuntimeError('Paired response identity or episode differs')
        row.update(outcome['row']);views[lane] = outcome['view']
    return rows
