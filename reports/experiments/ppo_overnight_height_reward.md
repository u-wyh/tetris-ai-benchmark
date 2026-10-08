# PPO overnight height-risk experiments

This queue uses the fixed 32 validation seeds and 5,000-piece cap. Existing
Raw, Shaped V1 (λ hole 0.10), and Shaped V2 (λ hole 0.02) are evaluated first;
it then trains V3 (hole 0.02 + height-risk 0.02) and V4 (height-risk 0.02)
from independent Step-0 initializations. Human rules, observation, and action
space are unchanged. Results and stage state are written under
`runs/ppo_overnight_height_reward/`.

The queue will select a candidate only after paired validation results are
available. It will not start a 10M run automatically until the comparison is
complete and the candidate passes the survival and reward-exploitation checks.
