"""Run under resource_guard, serially after the live development pipeline."""

import json
import os
from pathlib import Path

import pytest
import torch
from sb3_contrib import MaskablePPO

from training.initialization import validate_context_model_recipe


def test_real_cuda_checkpoint_matches_recorded_recipe_and_rejects_changed_gamma():
    source=Path(os.environ.get('TETRIS_ORIGINAL_CONTEXT_DIR',
        str(Path(__file__).resolve().parents[1]/'runs/ppo_candidate_raw_30m_seed42')))
    if not (source/'final/model.zip').exists():
        pytest.skip('Original CUDA checkpoint unavailable')
    assert torch.cuda.is_available(), 'This validation requires CUDA; no CPU fallback'
    config=json.loads((source/'config.json').read_text())
    model=MaskablePPO.load(str(source/'final/model.zip'),device='cuda')
    assert str(next(model.policy.parameters()).device)=='cuda:0'
    assert len(model.policy.optimizer.state)==18
    validate_context_model_recipe(model,config)
    config['ppo']['gamma']=0.999
    with pytest.raises(ValueError,match='gamma'):validate_context_model_recipe(model,config)
