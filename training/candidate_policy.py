"""Masked PPO policy that scores every legal placement with shared weights."""

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from training.candidate_features import FEATURE_SCALE
from training.tetris_core import ACTION_COUNT


class PublicStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=237)

    def forward(self, observations):
        return observations["state"]


class CandidateScoringPolicy(MaskableActorCriticPolicy):
    """Shared actor scores legal rows; critic reads only public state."""

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        if not isinstance(observation_space, spaces.Dict) or not isinstance(action_space, spaces.Discrete):
            raise ValueError("Candidate policy requires Dict observation and Discrete action")
        if action_space.n != ACTION_COUNT:
            raise ValueError("Candidate policy requires 1840 actions")
        kwargs.setdefault("features_extractor_class", PublicStateExtractor)
        kwargs.setdefault("ortho_init", False)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        self.state_encoder = nn.Sequential(nn.Linear(237, 128), nn.Tanh(),
                                           nn.Linear(128, 64), nn.Tanh())
        self.candidate_encoder = nn.Sequential(nn.Linear(16, 64), nn.Tanh(),
                                               nn.Linear(64, 64), nn.Tanh())
        self.score_head = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.value_net = nn.Sequential(nn.Linear(237, 256), nn.Tanh(),
                                       nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))
        self.register_buffer("candidate_scale", torch.as_tensor(FEATURE_SCALE.copy()))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _distribution(self, obs, action_masks):
        if action_masks is None:
            raise ValueError("Candidate policy requires the official legal action mask")
        mask = torch.as_tensor(action_masks, dtype=torch.bool, device=obs["state"].device)
        mask = mask.reshape(-1, ACTION_COUNT)
        if torch.any(~mask.any(dim=1)):
            raise ValueError("Candidate policy received a state without legal actions")
        selected = mask.nonzero(as_tuple=False)
        state_latent = self.state_encoder(obs["state"])
        candidates = obs["candidate"][selected[:, 0], selected[:, 1]].float()
        candidate_latent = self.candidate_encoder(candidates / self.candidate_scale)
        joined = torch.cat((state_latent[selected[:, 0]], candidate_latent), dim=1)
        scores = self.score_head(joined).squeeze(-1)
        logits = torch.full(mask.shape, -1e8, dtype=scores.dtype, device=scores.device)
        logits[selected[:, 0], selected[:, 1]] = scores
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        distribution.apply_masking(mask)
        return distribution

    def forward(self, obs, deterministic=False, action_masks=None):
        distribution = self._distribution(obs, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self.predict_values(obs)
        return actions.reshape((-1, *self.action_space.shape)), values, distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions, action_masks=None):
        distribution = self._distribution(obs, action_masks)
        return self.predict_values(obs), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs, action_masks=None):
        return self._distribution(obs, action_masks)

    def predict_values(self, obs):
        return self.value_net(obs["state"])
