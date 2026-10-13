"""Background report generation remains reproducible after long experiments."""
import json

from training import imitation_next_queue as queue


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_generate_reports_from_completed_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "RUN", tmp_path / "runs/candidate_bc_next_stage")
    monkeypatch.setattr(queue, "DAGGER", tmp_path / "runs/candidate_bc_dagger_v1")
    names = ("candidate_bc_generalization.md", "candidate_bc_ppo_degradation.md",
             "candidate_kl_ppo_2m.md", "candidate_dagger_v1.md",
             "candidate_imitation_next_stage.md")
    for name in names:
        path = tmp_path / "reports/experiments" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Report\n\n状态：**等待正式结果。**\n")
    sample = {"mean_pieces": 20000, "median_pieces": 20000, "p10_pieces": 20000,
              "p90_pieces": 20000, "capped_count": 30, "mean_lines": 7999,
              "mean_score": 333000000, "mean_wall_seconds": 800}
    save(queue.RUN / "generalization/summary.json",
         {"bc": sample, "v2_hold": sample,
          "paired": {"both_capped": 30, "bc_wins": 0, "v2_wins": 0}})
    row = {"mean_pieces": 20000, "p10_pieces": 20000, "capped_count": 32,
           "mean_lines": 7999, "new_holes_per_100_pieces": 2.1,
           "mean_height_at_end": 3.0}
    labels = ("pure_30m", "bc", "bc_ppo_2m", "kl_ppo_2m", "dagger_selected")
    save(queue.RUN / "comparison_summary.json",
         {"existing": {label: row for label in labels[:3]},
          "new": {label: row for label in labels[3:]},
          "v2_action_agreement": {"models": {
              label: {"v2_action_agreement": .95} for label in labels}}})
    save(queue.RUN / "long_validation_protocol.json", {"models": ["bc"]})
    save(queue.RUN / "comparisons/long_100k_bc/summary.json",
         {"mean_pieces": 100000, "p10_pieces": 100000, "capped_count": 32})
    save(queue.DAGGER / "data/seed_1.json",
         {"samples": 1000, "height_ge_12": 2, "height_ge_16": 0,
          "holes_ge_4": 3, "learner_teacher_agreement": 900})
    queue.generate_reports()
    assert "正式实测结果" in (tmp_path / "reports/experiments/candidate_bc_generalization.md").read_text()
    final = (tmp_path / "reports/experiments/candidate_imitation_next_stage.md").read_text()
    assert "95.0%" in final and "100k 长验证模型：bc" in final
    assert json.loads((queue.RUN / "report_status.json").read_text())["state"] == "generated"
