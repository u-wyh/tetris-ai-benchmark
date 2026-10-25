"""Audit saved clear-cadence records without replaying the PPO policy."""

import hashlib
import gzip
import json
import pickle
from pathlib import Path
import math

import torch
from sb3_contrib import MaskablePPO

from training.evaluation.source_provenance import verify as verify_source
from training.evaluation.hybrid_queue import PPO, HASHES
from training.evaluation.failure_curriculum_eval import LongCandidateTetrisEnv
from training.evaluation.failure_curriculum_eval import fast_action
from training.tetris_core.legal_placements import get_legal_placements
from training.candidate_features import board_features
from training.traditional.v2 import simulate_public
from training.env.tetris_env import board_max_height


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/ppo_clear_cadence_counterfactual"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_provenance(row):
    """Validate provenance for new records; legacy records have no manifest."""
    if "execution_provenance" in row:
        verify_source(ROOT, row["execution_provenance"])


def audit():
    check(digest(PPO) == HASHES["ppo"], "frozen PPO file")
    torch.set_num_threads(1)
    model = MaskablePPO.load(str(PPO), device="cpu")
    episodes = states = branches = rescues = harms = 0
    instability_seeds = control_seeds = 0
    for seed in range(1550000, 1550008):
        episode = read(RUN / "episodes" / f"seed_{seed}.json")
        check_provenance(episode)
        check(episode["seed"] == seed and episode["cap"] == 20000, f"episode {seed}")
        check(episode["ppo_sha256"] == HASHES["ppo"], f"frozen PPO identity {seed}")
        check(episode["design_sha256"] == digest(ROOT / "reports/experiments/ppo_clear_cadence_counterfactual_design.md"), f"design {seed}")
        check(0 < episode["pieces"] <= 20000 and len(episode["states"]) <= 4, f"bounds {seed}")
        check(episode["game_over"] != episode["capped"] and
              episode["capped"] == (episode["pieces"] == 20000), f"episode terminal {seed}")
        check(episode["score"] >= 0 and len(episode["clears"]) == 5 and
              sum(episode["clears"]) <= episode["pieces"], f"episode totals {seed}")
        episodes += 1
        instability_seeds += any(s["stratum"] == "instability" for s in episode["states"])
        control_seeds += any(s["stratum"] == "recovered_control" for s in episode["states"])
        steps = [s["step"] for s in episode["states"]]
        check(all(b - a >= 500 for a, b in zip(sorted(steps), sorted(steps)[1:])), f"spacing {seed}")
        for state in episode["states"]:
            step = state["step"]
            check(state["stratum"] in ("instability", "recovered_control") and
                  250 <= step < episode["pieces"] and step % 25 == 0,
                  f"selection bounds {seed}:{step}")
            prefix = f"seed_{seed}_step_{step}"
            snapshot = RUN / "states" / f"{prefix}.pkl.gz"
            choice_path = RUN / "candidates" / f"{prefix}.json"
            choice = read(choice_path)
            check(digest(snapshot) == state["snapshot_sha256"] == choice["snapshot_sha256"], f"snapshot {prefix}")
            saved_state = pickle.loads(gzip.decompress(snapshot.read_bytes()))
            check(saved_state["episode_seed"] == seed and saved_state["pieces"] == step, f"snapshot state {prefix}")
            env = LongCandidateTetrisEnv(max_pieces=saved_state["max_pieces"])
            try:
                env.reset(seed=seed)
                env.set_state(saved_state)
                public = env.core.get_public_observation()
                legal = {p["actionId"] for p in get_legal_placements(public)}
                check(state["height"] == board_max_height(public["board"]) and
                      state["holes"] == board_features(public["board"])[0],
                      f"selection features {prefix}")
                mask = env.action_masks()
                check(legal == {i for i, allowed in enumerate(mask) if allowed}, f"Core mask {prefix}")
                obs = env.get_observation()
                predicted = fast_action(model, obs, mask)
                tensor = {k: torch.as_tensor(v, device="cpu").unsqueeze(0) for k, v in obs.items()}
                with torch.no_grad():
                    probs = model.policy.get_distribution(
                        tensor, action_masks=torch.as_tensor(mask).unsqueeze(0)
                    ).distribution.probs[0].cpu().numpy()
            finally:
                env.close()
            check(choice["ppo_sha256"] == episode["ppo_sha256"] and choice["design_sha256"] == episode["design_sha256"], f"candidate identity {prefix}")
            actions = (choice["original_action"], choice["alternative_action"])
            check(None not in actions and len(set(actions)) == 2, f"actions {prefix}")
            check(actions[0] == predicted and math.isclose(float(sum(probs)), 1.0, abs_tol=1e-5),
                  f"PPO action and probability normalization {prefix}")
            ranked = choice["ranked_legal_actions"]
            check({r["action"] for r in ranked} == legal and len(ranked) == len(legal), f"candidate legality {prefix}")
            placements = {p["actionId"]: p for p in get_legal_placements(public)}
            for row in ranked:
                child, clear = simulate_public(public, placements[row["action"]])
                holes, _, height, _, _ = board_features(child["board"])
                expected_key = [int(child["phase"] == "over"), -int(clear),
                                int(height), int(holes), -row["ppo_probability"], row["action"]]
                check(row["rank_key"] == expected_key and
                      0 <= row["ppo_probability"] <= 1 and
                      math.isclose(row["ppo_probability"], float(probs[row["action"]]), abs_tol=1e-7),
                      f"candidate features and PPO probability {prefix}:{row['action']}")
            check(ranked == sorted(ranked, key=lambda r: r["rank_key"]), f"ranking {prefix}")
            check(actions[0] in [r["action"] for r in ranked] and actions[1] == next(r["action"] for r in ranked if r["action"] != actions[0]), f"choice {prefix}")
            outcomes = []
            for action in actions:
                branch = read(RUN / "branches" / f"{prefix}_action_{action}.json")
                check_provenance(branch)
                outcome = branch["outcome"]
                check(branch["candidate_sha256"] == digest(choice_path) and branch["snapshot_sha256"] == state["snapshot_sha256"] and branch["ppo_sha256"] == episode["ppo_sha256"], f"branch identity {prefix}:{action}")
                n = outcome["pieces"]
                check(branch["action"] == outcome["action"] == action and branch["horizon"] == outcome["horizon"] == 3000, f"branch action {prefix}:{action}")
                check(0 < n <= 3000 and len(outcome["scores"]) == len(outcome["fours"]) == n, f"series length {prefix}:{action}")
                check(all(b >= a and b - a <= 1 for a, b in zip([0] + outcome["fours"][:-1], outcome["fours"])), f"fours {prefix}:{action}")
                check(all(b >= a >= 0 for a, b in zip([0] + outcome["scores"][:-1], outcome["scores"])), f"scores {prefix}:{action}")
                check(outcome["fours"][-1] <= n and outcome["game_over"] != outcome["censored"],
                      f"branch terminal {prefix}:{action}")
                check(outcome["censored"] == (not outcome["game_over"] and n == 3000), f"censoring {prefix}:{action}")
                outcomes.append(outcome)
                branches += 1
            base, alt = outcomes
            common = min(base["pieces"], alt["pieces"])
            expected = {"seed": seed, "step": step, "stratum": state["stratum"],
                        "original_action": actions[0], "alternative_action": actions[1],
                        "original_pieces": base["pieces"], "alternative_pieces": alt["pieces"],
                        "original_game_over": base["game_over"], "alternative_game_over": alt["game_over"],
                        "common_pieces": common,
                        "original_score": base["scores"][common - 1], "alternative_score": alt["scores"][common - 1],
                        "original_fours": base["fours"][common - 1], "alternative_fours": alt["fours"][common - 1],
                        "rescue": base["game_over"] and alt["pieces"] - base["pieces"] >= 256,
                        "harm": alt["game_over"] and base["pieces"] - alt["pieces"] >= 256}
            check(read(RUN / "comparisons" / f"{prefix}.json") == expected, f"comparison {prefix}")
            rescues += expected["rescue"]
            harms += expected["harm"]
            states += 1
    check((episodes, states, branches, rescues, harms, instability_seeds, control_seeds) == (8, 22, 44, 0, 1, 4, 8), "aggregate mismatch")
    return {"episodes": episodes, "states": states, "branches": branches,
            "rescues": rescues, "harms": harms,
            "instability_seeds": instability_seeds, "control_seeds": control_seeds}


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True))
