"""Gradient groups keep sample weights and correct episode coordinates."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_critic_tail_gradients import contribution_summary, group_masks, partition_gradients


@pytest.mark.parametrize('selected', [[True, False, False], [False]*3, [True]*3])
def test_partition_uses_original_denominator_and_preserves_parameters(selected):
    p = torch.nn.Parameter(torch.tensor(2.))
    loss = (p-torch.tensor([0., 1., 3.])).square()
    mask = torch.tensor(selected)
    full, left, right, error = partition_gradients(loss, mask, [p])
    expected = (2*(2-torch.tensor([0., 1., 3.]))*mask).sum()/3
    assert left.item() == pytest.approx(expected.item())
    torch.testing.assert_close(full, left+right)
    assert error < 1e-6 and p.item() == 2 and p.grad is None


def test_opposing_gradients_are_not_additive_norm_fractions():
    row = contribution_summary(torch.tensor([1.]), torch.tensor([3.]), torch.tensor([-2.]))
    assert row['selected_to_full_norm_ratio'] == 3
    assert row['selected_projection_on_full'] == 3
    assert row['selected_other_cosine'] == -1


def test_terminal_window_never_crosses_episode_start_and_ties_are_stable():
    arrays = dict(values=np.zeros((5, 2)), returns=np.ones((5, 2)),
        episode_starts=np.array([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]]),
        step_terminated=np.array([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]], dtype=bool))
    groups = group_masks(arrays, terminal_window=4, top_fraction=.2)
    assert np.argwhere(groups['terminal_window']).tolist() == [[1, 1], [2, 1], [3, 1]]
    assert np.argwhere(groups['top_residual']).tolist() == [[4, 0], [4, 1]]
