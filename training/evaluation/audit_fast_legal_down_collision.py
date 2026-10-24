"""Read-only consistency audit of the archived downward-collision benchmark."""

import json

from training.evaluation.audit_fast_legal_landing_sort import audit


if __name__ == "__main__":
    print(json.dumps(audit("ppo_fast_legal_down_collision_20261024", "03a2513"), sort_keys=True))
