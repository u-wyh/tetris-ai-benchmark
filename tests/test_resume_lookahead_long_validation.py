from datetime import date, datetime
from types import SimpleNamespace

import pytest

from scripts import resume_lookahead_long_validation as resume


@pytest.fixture
def protocol():
    return {
        "groups": {"main": [123], "extension": [123]},
        "main_cap": 6,
        "extension_cap": 12,
    }


def publish(run, label, variant, seed):
    path = resume.result_path(run, label, variant, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".npz", ".end.pkl", ".json"):
        path.with_suffix(suffix).write_text(f"{label}-{variant}-{seed}{suffix}")
    return path


def test_window_uses_explicit_beijing_date_and_closeout_margin():
    day = date(2026, 11, 24)
    start, end = resume.window_deadlines(day)
    assert resume.seconds_left(day, start) == 8 * 60 * 60 - resume.CLOSEOUT_SECONDS
    assert resume.seconds_left(day, datetime(2026, 11, 24, 8, 29, 29, tzinfo=resume.BEIJING)) == 1
    for outside in (start.replace(minute=29), end.replace(minute=29, second=30), end):
        with pytest.raises(resume.WindowStopped):
            resume.seconds_left(day, outside)
    with pytest.raises(resume.WindowStopped):
        resume.seconds_left(date(2026, 11, 25), start)


def test_history_links_and_checkpoint_copies_are_idempotent(tmp_path, protocol):
    archive, output = tmp_path / "archive", tmp_path / "output"
    output.mkdir()
    saved = publish(archive, "100k", "A", 123)
    resume.prepare_workspace(archive, output, protocol, b"protocol")
    resume.prepare_workspace(archive, output, protocol, b"protocol")
    linked = resume.result_path(output, "100k", "A", 123)
    assert linked.is_symlink() and linked.resolve() == saved
    assert linked.with_suffix(".end.pkl").is_file()

    old_partial = resume.result_path(archive, "100k", "E", 123)
    old_checkpoint = old_partial.with_suffix(".checkpoint.pkl")
    old_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    old_checkpoint.write_bytes(b"old checkpoint")
    new_partial = resume.result_path(output, "100k", "E", 123)
    new_checkpoint = new_partial.with_suffix(".checkpoint.pkl")
    resume.copy_checkpoint(old_partial, new_partial)
    assert new_checkpoint.read_bytes() == old_checkpoint.read_bytes()
    new_checkpoint.write_bytes(b"new checkpoint")
    resume.copy_checkpoint(old_partial, new_partial)
    assert new_checkpoint.read_bytes() == b"new checkpoint"
    assert old_checkpoint.read_bytes() == b"old checkpoint"

    with pytest.raises(RuntimeError, match="protocol"):
        resume.prepare_workspace(archive, output, protocol, b"modified protocol")
    assert saved.exists()


def test_resume_completes_main_before_extension_and_skips_results(tmp_path, protocol):
    archive, output = tmp_path / "archive", tmp_path / "output"
    output.mkdir()
    for variant in ("A", "B", "C", "BC"):
        publish(archive, "100k", variant, 123)
    resume.prepare_workspace(archive, output, protocol, b"protocol")
    attempts = []

    def game(variant, seed, cap, label):
        attempts.append((label, variant, cap))
        if (label, variant) == ("200k", "C") and attempts.count((label, variant, cap)) == 1:
            raise resume.WindowStopped("simulated window stop")
        publish(output, label, variant, seed)

    study = SimpleNamespace(
        IDENTITY="fixture", guard=lambda: None, init=lambda _: None, game=game,
        completed=lambda path, *_: path.read_bytes(),
    )
    with pytest.raises(resume.WindowStopped):
        resume.process_window(study, protocol, archive, output)
    assert attempts == [("100k", "E", 6), ("200k", "BC", 12), ("200k", "C", 12)]
    resume.process_window(study, protocol, archive, output)
    assert attempts[-2:] == [("200k", "C", 12), ("200k", "E", 12)]
    assert list(resume.pending_tasks(protocol, archive, output)) == []
