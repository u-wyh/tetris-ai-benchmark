"""Risk of PPO continuation after a public, legal candidate action."""
import numpy as np
import torch
from torch import nn

from training.candidate_features import FEATURE_SCALE


class RiskPredictor(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(237, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.action = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width + 32, width), nn.ReLU(), nn.Linear(width, 2))

    def forward(self, state, action):
        action = action / torch.as_tensor(FEATURE_SCALE, device=action.device)
        return self.head(torch.cat((self.state(state), self.action(action)), dim=-1))


def labels(step, pieces, game_over, horizon):
    """Return 1=death, 0=observed survival, None=right censored."""
    remaining = pieces - step
    if game_over and remaining <= horizon:
        return 1
    if remaining >= horizon:
        return 0
    return None


def probabilities(model, state, actions):
    with torch.no_grad():
        states = torch.as_tensor(np.asarray(state), dtype=torch.float32).reshape(1, 237)
        candidates = torch.as_tensor(np.asarray(actions), dtype=torch.float32).reshape(-1, 16)
        return torch.sigmoid(model(states.expand(len(candidates), -1), candidates)).cpu().numpy()
