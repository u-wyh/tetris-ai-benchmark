"""Mask-aware PPO regularized toward a frozen, probabilistic BC actor."""
import numpy as np
import torch
from gymnasium import spaces
from torch.nn import functional as F
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance


def masked_policy_kl(current_policy, reference_policy, observations, action_masks):
    """Exact categorical KL over the official legal actions; gradients reach current only."""
    mask = action_masks.bool()
    current = current_policy.get_distribution(observations, action_masks=mask).distribution
    with torch.no_grad():
        reference = reference_policy.get_distribution(observations, action_masks=mask).distribution
    log_p = current.logits.masked_fill(~mask, 0.0)
    log_q = reference.logits.masked_fill(~mask, 0.0)
    return (current.probs.masked_fill(~mask, 0.0) * (log_p - log_q)).sum(-1).mean()


class KLMaskablePPO(MaskablePPO):
    """Original MaskablePPO losses plus beta KL(current || frozen BC)."""

    def __init__(self, *args, kl_beta=0.02, **kwargs):
        self.kl_beta = float(kl_beta)
        if self.kl_beta < 0:
            raise ValueError("KL coefficient must be nonnegative")
        self.reference_policy = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["reference_policy"]

    def set_reference(self, source_model):
        self.reference_policy = source_model.policy
        self.reference_policy.to(self.device).eval()
        for parameter in self.reference_policy.parameters():
            parameter.requires_grad_(False)

    def train(self):
        if self.reference_policy is None:
            raise RuntimeError("KL-PPO requires a frozen reference policy")
        self.policy.set_training_mode(True)
        self.reference_policy.eval()
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (self.clip_range_vf(self._current_progress_remaining)
                         if self.clip_range_vf is not None else None)
        entropy_losses, pg_losses, value_losses, clip_fractions, bc_kls = [], [], [], [], []
        continue_training = True
        loss = None
        approx_kl_divs = []
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions, action_masks=rollout_data.action_masks)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -torch.min(advantages * ratio,
                                         advantages * torch.clamp(ratio, 1 - clip_range,
                                                                   1 + clip_range)).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())
                values_pred = (values if clip_range_vf is None else
                               rollout_data.old_values + torch.clamp(values - rollout_data.old_values,
                                                                      -clip_range_vf, clip_range_vf))
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = -torch.mean(-log_prob) if entropy is None else -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                bc_kl = masked_policy_kl(self.policy, self.reference_policy,
                                         rollout_data.observations, rollout_data.action_masks)
                bc_kls.append(bc_kl.item())
                loss = (policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                        + self.kl_beta * bc_kl)
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())
        for key, value in (("train/entropy_loss", np.mean(entropy_losses)),
                           ("train/policy_gradient_loss", np.mean(pg_losses)),
                           ("train/value_loss", np.mean(value_losses)),
                           ("train/approx_kl", np.mean(approx_kl_divs)),
                           ("train/clip_fraction", np.mean(clip_fractions)),
                           ("train/bc_kl", np.mean(bc_kls)),
                           ("train/loss", loss.item()),
                           ("train/explained_variance", explained_var)):
            self.logger.record(key, value)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
