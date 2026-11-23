"""Run one off-peak window of the frozen public Lookahead study."""

import argparse
from datetime import date, datetime, time, timedelta
from functools import partial
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from zoneinfo import ZoneInfo

if __package__:
    from .inspect_lookahead_long_validation import read_record
else:
    from inspect_lookahead_long_validation import read_record


ROOT = Path(__file__).resolve().parents[1]
BEIJING = ZoneInfo("Asia/Shanghai")
ARCHIVE = Path("runs/lookahead_long_validation")
CONTINUATION = ROOT / "runs/lookahead_long_validation_resume"
PHASES = (
    ("100k", "main", "main_cap", ("A", "B", "C", "E", "BC")),
    ("200k", "extension", "extension_cap", ("BC", "C", "E")),
)
CLOSEOUT_SECONDS = 30


class WindowStopped(RuntimeError):
    """The current window can end with resumable state."""


def window_deadlines(day):
    return (
        datetime.combine(day, time(0, 30), BEIJING),
        datetime.combine(day, time(8, 30), BEIJING),
    )


def seconds_left(day, now=None):
    start, end = window_deadlines(day)
    now = datetime.now(BEIJING) if now is None else now
    deadline = end - timedelta(seconds=CLOSEOUT_SECONDS)
    if not start <= now < deadline:
        raise WindowStopped("Outside the Beijing 00:30–08:30 evaluation window")
    return (deadline - now).total_seconds()


def tasks(protocol):
    for label, group, cap, variants in PHASES:
        for seed in protocol["groups"][group]:
            for variant in variants:
                yield label, variant, seed, protocol[cap]


def result_path(run, label, variant, seed):
    return run / "evaluation" / label / variant / f"seed_{seed}.json"


def pending_tasks(protocol, archive, output):
    for label, variant, seed, cap in tasks(protocol):
        if not any(result_path(run, label, variant, seed).is_file() for run in (archive, output)):
            yield label, variant, seed, cap


def prepare_workspace(archive, output, protocol, raw_protocol):
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if protocol_path.read_bytes() != raw_protocol:
            raise RuntimeError("Continuation protocol does not match its historical archive")
    else:
        temporary = protocol_path.with_suffix(".tmp")
        temporary.write_bytes(raw_protocol)
        temporary.replace(protocol_path)

    # Publish the old result marker last, just as the original evaluator does.
    # Game execution writes only to the separate continuation workspace.
    for label, variant, seed, _ in tasks(protocol):
        previous = result_path(archive, label, variant, seed)
        if not previous.is_file():
            continue
        destination = result_path(output, label, variant, seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".npz", ".end.pkl", ".json"):
            source = previous.with_suffix(suffix)
            target = destination.with_suffix(suffix)
            if not source.is_file():
                raise RuntimeError(f"Incomplete historical artifact: {source}")
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            if target.exists() or target.is_symlink():
                raise RuntimeError(f"Historical artifact conflicts with continuation output: {target}")
            target.symlink_to(source)


def copy_checkpoint(previous, destination):
    previous = previous.with_suffix(".checkpoint.pkl")
    destination = destination.with_suffix(".checkpoint.pkl")
    if previous.is_file() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".importing")
        shutil.copyfile(previous, temporary)
        temporary.replace(destination)


def process_window(study, protocol, archive, output):
    phases = list(pending_tasks(protocol, archive, output))
    if not phases:
        return
    study.guard()
    study.init(study.IDENTITY)
    for label, variant, seed, cap in phases:
        study.guard()
        destination = result_path(output, label, variant, seed)
        if label == "200k":
            parent = result_path(output, "100k", variant, seed)
            if not parent.is_file():
                raise RuntimeError(f"No completed 100K parent for {variant}, seed {seed}")
        copy_checkpoint(result_path(archive, label, variant, seed), destination)
        study.game(variant, seed, cap, label)
        study.completed(destination, variant, seed, cap)


def enforce_runtime(study, output, day):
    seconds_left(day)
    if (output / "STOP").exists():
        raise WindowStopped("Continuation STOP marker is present")
    if study.disk_bytes() >= int(5.5 * 1024**3):
        raise WindowStopped("Archive and continuation reached the retained-data limit")
    if shutil.disk_usage(output).free < 8 * 1024**3:
        raise WindowStopped("Free disk is below the original 8 GiB limit")
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:") and int(line.split()[1]) < 2 * 1024**2:
            raise WindowStopped("Available memory is below the original 2 GiB limit")
    try:
        study.check_services()
    except study.BudgetStop as exc:
        raise WindowStopped(str(exc)) from exc


def load_frozen_study(checkout):
    # Frozen source verification resolves several source globs against cwd.
    sys.path.insert(0, str(checkout))
    os.chdir(checkout)
    from training import lookahead_long_validation as study

    study.freeze()
    return study


def run_worker(checkout, output, day, receipt):
    seconds_left(day)
    archive = checkout / ARCHIVE
    output.mkdir(parents=True, exist_ok=True)
    with (archive / "queue.lock").open("rb") as old_lock, (output / "queue.lock").open("a") as lock:
        fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw_protocol = (archive / "protocol.json").read_bytes()
        protocol = read_record(raw_protocol)
        started = datetime.now(BEIJING)
        outcome = "failed"
        reason = None

        def stop(signum, _frame):
            raise WindowStopped(f"Received {signal.Signals(signum).name}")

        signals = (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
        handlers = {signum: signal.signal(signum, stop) for signum in signals}
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds_left(day))
            study = load_frozen_study(checkout)
            prepare_workspace(archive, output, protocol, raw_protocol)
            study.RUN = output
            study.IDENTITY = hashlib.sha256(raw_protocol).hexdigest()
            study.guard = partial(enforce_runtime, study, output, day)
            if any(pending_tasks(protocol, archive, output)):
                study.guard()
                health = study.read_record(output / "health.json")
                if any(health["services"].get(port) != 200 for port in ("8080", "8081")):
                    raise WindowStopped("Historical 8080/8081 service checks have not passed")
                process_window(study, protocol, archive, output)
            outcome = "completed"
        except WindowStopped as exc:
            outcome, reason = "stopped", str(exc)
        except Exception as exc:
            reason = repr(exc)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
            remaining = list(pending_tasks(protocol, archive, output))
            document = dict(
                started=started.isoformat(), ended=datetime.now(BEIJING).isoformat(),
                outcome=outcome, reason=reason, remaining=len(remaining),
                next_game=remaining[0] if remaining else None,
            )
            temporary = receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(receipt)
            print(json.dumps(document, ensure_ascii=False), flush=True)


def launch(checkout, output, day):
    import resource_guard

    seconds = int(seconds_left(day) + CLOSEOUT_SECONDS)
    attempt = output / "windows" / f"{day.isoformat()}-{os.getpid()}-{datetime.now(BEIJING):%H%M%S%f}"
    return resource_guard.run(argparse.Namespace(
        profile=ROOT / "configs/training_resources.json", output=attempt, seconds=seconds,
        command=[
            sys.executable, str(Path(__file__).resolve()), str(checkout),
            "--output", str(output), "--window-date", day.isoformat(),
            "--worker-receipt", str(attempt / "window.json"),
        ],
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Frozen research worktree")
    parser.add_argument("--output", type=Path, default=CONTINUATION)
    parser.add_argument("--window-date", type=date.fromisoformat, help="Explicit Beijing date for execution")
    parser.add_argument("--worker-receipt", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    checkout, output = args.checkout.resolve(), args.output.resolve()
    archive = (checkout / ARCHIVE).resolve()
    if output.is_relative_to(archive) or archive.is_relative_to(output):
        parser.error("The continuation output must be separate from the historical archive")
    try:
        if args.worker_receipt is not None:
            if args.window_date is None:
                parser.error("The worker requires an explicit window date")
            run_worker(checkout, output, args.window_date, args.worker_receipt.resolve())
        elif args.window_date is not None:
            return launch(checkout, output, args.window_date)
        else:
            protocol = read_record((archive / "protocol.json").read_bytes())
            print(json.dumps(list(pending_tasks(protocol, archive, output)), indent=2))
    except WindowStopped as exc:
        parser.exit(2, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
