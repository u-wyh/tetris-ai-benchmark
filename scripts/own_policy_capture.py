"""Diagnostic lanes, compact greedy observations and chained durable captures."""
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np

from training.train_vector_transaction import (TransactionCompactSearchCandidateEnv,
    TransactionNativeSearchCandidateEnv)
from training.evaluation.evaluate_resumable import durable_json
from restore_rank_contexts import observation_digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CaptureControls:
    def capture_reset(self, seed):
        self.reset(seed=seed)
        if self.episode_seed != seed or self.episode_count != 1 or self.pieces != 0:
            raise ValueError('Capture reset did not preserve its explicit first episode seed')
        self._capture_active = True
        self._capture_seed = seed
        return self.capture_view()

    def capture_view(self):
        if not getattr(self, '_capture_active', False):
            return None
        return dict(observation=self.get_observation(), mask=self.action_masks())

    def capture_step(self, action):
        if not getattr(self, '_capture_active', False):
            return None
        observation, reward, terminated, truncated, info = self.step(action)
        if self.episode_seed != self._capture_seed or self.episode_count != 1:
            raise RuntimeError('Capture episode unexpectedly reset')
        self._capture_active = not (terminated or truncated)
        fields = ['pieces', 'score', 'cleared_lines', 'holes_before', 'holes_after', 'height_after']
        row = {key: int(info[key]) for key in fields}
        row.update(reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                   episode_seed=int(self.episode_seed))
        return dict(row=row, view=None if not self._capture_active else
                    dict(observation=observation, mask=self.action_masks()))

    def capture_export(self):
        return dict(state=self.get_state(), active=self._capture_active)

    def capture_restore(self, saved):
        state, active = saved['state'], saved['active']
        if (type(active) is not bool or active == state['finished'] or
                state['episode_count'] != 1):
            raise ValueError('Capture active/finished/episode metadata differs')
        self.set_state(copy.deepcopy(state))
        self._capture_active = active
        self._capture_seed = state['episode_seed']
        return self.capture_view()


class CaptureNativeEnv(CaptureControls, TransactionNativeSearchCandidateEnv):
    pass


class CaptureCompactEnv(CaptureControls, TransactionCompactSearchCandidateEnv):
    pass


def greedy_details(policy, observations, masks, executed=None):
    """Use the actual float32 probability mode; transfer only selected scalars."""
    import torch
    from stable_baselines3.common.utils import obs_as_tensor
    with torch.no_grad():
        obs = obs_as_tensor(observations, policy.device)
        distribution = policy.get_distribution(obs, action_masks=masks)
        categorical = distribution.distribution
        actions = distribution.get_actions(deterministic=True)
        rows = torch.arange(len(actions), device=actions.device)
        mask = torch.as_tensor(masks, dtype=torch.bool, device=actions.device)
        if not mask[rows, actions].all():
            raise ValueError('Greedy capture selected an illegal action')
        legal_logits = categorical.logits.masked_fill(~mask, -torch.inf)
        top = legal_logits.topk(2, dim=1).values
        margin = torch.where(mask.sum(dim=1) > 1, top[:, 0]-top[:, 1], torch.inf)
        selected = actions if executed is None else torch.as_tensor(executed, device=actions.device)
        if not mask[rows, selected].all():
            raise ValueError('Observed executed action is illegal')
        return {key: value.detach().cpu().numpy().copy() for key, value in dict(
            action=actions, probability=categorical.probs[rows, actions], margin=margin,
            probability_of_executed=categorical.probs[rows, selected]).items()}


def advance_round(env, views, candidate, shadow, round_index):
    active = [lane for lane, view in enumerate(views) if view is not None]
    if not active:
        return []
    observations = {key: np.stack([views[lane]['observation'][key] for lane in active])
                    for key in views[active[0]]['observation']}
    masks = np.stack([views[lane]['mask'] for lane in active])
    chosen = greedy_details(candidate.policy, observations, masks)
    observed = greedy_details(shadow.policy, observations, masks, chosen['action'])
    rows = []
    for batch, lane in enumerate(active):
        action, other = int(chosen['action'][batch]), int(observed['action'][batch])
        digest = observation_digest(**views[lane])
        row = dict(round=round_index, lane=lane, action=action, shadow_action=other,
            observation_sha256=digest['observation'], mask_sha256=digest['mask'],
            candidate_features=observations['candidate'][batch, action].copy(),
            shadow_features=observations['candidate'][batch, other].copy(),
            candidate_probability=float(chosen['probability'][batch]),
            candidate_margin=float(chosen['margin'][batch]),
            shadow_probability=float(observed['probability'][batch]),
            shadow_margin=float(observed['margin'][batch]),
            shadow_probability_of_executed=float(observed['probability_of_executed'][batch]))
        outcome = env.env_method('capture_step', action, indices=[lane])[0]
        if outcome is None:
            raise RuntimeError('An active capture lane did not advance')
        row.update(outcome['row'])
        rows.append(row)
        views[lane] = outcome['view']
    return rows


def save_commit(folder, number, identity, previous_sha, snapshot, rows, audit):
    """Rename is the commit point. Existing committed chunks are never replaced."""
    directory = folder / f'chunk_{number:04d}'
    if directory.exists():
        raise FileExistsError('Capture chunk already committed')
    pending = directory.with_name(directory.name+'.pending')
    if pending.exists():
        abandoned = folder/'abandoned'
        abandoned.mkdir(exist_ok=True)
        pending.rename(abandoned/(pending.name+'.'+str(os.getpid())))
    pending.mkdir(parents=True)
    with (pending/'state.pkl').open('xb') as stream:
        pickle.dump(snapshot, stream, protocol=5)
        stream.flush(); os.fsync(stream.fileno())
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]} if rows else {}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError('Capture trace contains an object array')
    with (pending/'rows.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    durable_json(pending/'manifest.json', dict(identity=identity, number=number,
        previous_manifest_sha256=previous_sha, round=snapshot['round'],
        environment_steps=snapshot['environment_steps'], rows=len(rows), audit=audit,
        files={name: sha(pending/name) for name in ['state.pkl', 'rows.npz']}))
    pending.rename(directory)
    descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha(directory/'manifest.json')


def read_chain(folder, identity):
    """Validate contiguous manifests and ALL file hashes before unpickling our last state."""
    chunks = sorted(path for path in folder.glob('chunk_[0-9][0-9][0-9][0-9]') if path.is_dir())
    previous = None
    last_round = last_steps = 0
    for number, directory in enumerate(chunks):
        manifest = json.loads((directory/'manifest.json').read_text())
        if (directory.name != f'chunk_{number:04d}' or manifest['number'] != number or
                manifest['identity'] != identity or manifest['previous_manifest_sha256'] != previous or
                set(manifest['files']) != {'state.pkl', 'rows.npz'} or
                any(sha(directory/name) != expected for name, expected in manifest['files'].items())):
            raise ValueError('Capture chain identity, link, sequence or file checksum differs')
        if ((number == 0 and (manifest['round'] != 0 or manifest['rows'] != 0)) or
                (number > 0 and not last_round < manifest['round'] <= last_round+identity['chunk_rounds']) or
                manifest['environment_steps'] != last_steps+manifest['rows']):
            raise ValueError('Capture chain work accounting differs')
        previous = sha(directory/'manifest.json')
        last_round, last_steps = manifest['round'], manifest['environment_steps']
    if not chunks:
        return None, None, 0
    # Only this project's own local cache after the external manifest SHA checks above.
    snapshot = pickle.loads((chunks[-1]/'state.pkl').read_bytes())
    if snapshot['round'] != last_round or snapshot['environment_steps'] != last_steps:
        raise ValueError('Capture snapshot counters differ from manifest')
    return snapshot, previous, len(chunks)
