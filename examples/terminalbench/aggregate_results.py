#!/usr/bin/env python3
"""Check shard completion and aggregate three strict pass@1 trials."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tomllib
from pathlib import Path
from typing import Any


EXPECTED_TASKS = 89
EXPECTED_TRIALS = 3
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_MANIFEST = SCRIPT_DIR / "data" / "terminal-bench-2-1" / "tasks" / "dataset.toml"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def task_names() -> list[str]:
    data = tomllib.loads(DATASET_MANIFEST.read_text())
    return [str(item["name"]).split("/", 1)[-1] for item in data["tasks"]]


def readable_session_names(run_dir: Path, names: list[str]) -> set[str]:
    found: set[str] = set()
    longest_first = sorted(names, key=len, reverse=True)
    rollout_root = run_dir / "rollout_results"
    search_root = rollout_root if rollout_root.is_dir() else run_dir
    for path in search_root.rglob("ses_*.json"):
        try:
            session = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        text = f"{path.as_posix()} {session.get('task_id', '')}"
        for name in longest_first:
            if name in text:
                found.add(name)
                break
    return found


def shard_status(run_dir: Path) -> int:
    manifest = run_dir / "swebench_verified_selected.jsonl"
    if not manifest.is_file():
        print(f"incomplete: selected manifest not found: {manifest}")
        return 1
    selected = [str((row.get("metadata") or {})["instance_id"]) for row in load_jsonl(manifest)]
    attempted = readable_session_names(run_dir, selected)
    missing = [name for name in selected if name not in attempted]
    print(json.dumps({"run_dir": str(run_dir), "selected": len(selected), "attempted": len(attempted), "missing": missing}, indent=2))
    return 0 if selected and not missing else 1


def aggregate(result_root: Path) -> int:
    names = task_names()
    trials: list[dict[str, dict[str, Any]]] = []
    trial_summaries = []
    for trial_index in range(1, EXPECTED_TRIALS + 1):
        trial_dir = result_root / f"trial_{trial_index:02d}"
        by_task: dict[str, dict[str, Any]] = {}
        for path in sorted(trial_dir.glob("shard_*/strict_results.jsonl")):
            for row in load_jsonl(path):
                name = str(row["instance_id"])
                by_task.setdefault(name, row)
        attempted = set(by_task)
        missing = [name for name in names if name not in attempted]
        score = sum(float(by_task[name].get("score", 0.0)) for name in names if name in attempted)
        pass_rate = score / EXPECTED_TASKS
        trial_summaries.append({
            "trial": trial_index,
            "attempted_tasks": len(attempted),
            "resolved_tasks": int(score),
            "missing_tasks": missing,
            "pass_at_1": pass_rate,
        })
        trials.append(by_task)

    complete = all(not item["missing_tasks"] for item in trial_summaries)
    rates = [float(item["pass_at_1"]) for item in trial_summaries]
    mean = statistics.mean(rates)
    stderr = statistics.stdev(rates) / math.sqrt(len(rates)) if len(rates) > 1 else 0.0
    payload = {
        "benchmark": "terminal-bench/terminal-bench-2-1",
        "dataset_commit": "c5ee500c185224c97cd6caff7866a990a0057f41",
        "metric": "strict_pass_at_1",
        "complete": complete,
        "expected_tasks_per_trial": EXPECTED_TASKS,
        "expected_trials": EXPECTED_TRIALS,
        "trials": trial_summaries,
        "mean_pass_at_1": mean,
        "stderr_pass_at_1": stderr,
        "scoring_rule": "Each task's first readable session is final. Reward 1 with a clean completed session passes; failures, errors and timeouts score 0. No retries.",
    }
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "final_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    with (result_root / "final_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "trial_1", "trial_2", "trial_3", "passes"])
        for name in names:
            values = [int(float(trial.get(name, {}).get("score", 0.0))) for trial in trials]
            writer.writerow([name, *values, sum(values)])

    print(json.dumps(payload, indent=2))
    print(f"final_summary={result_root / 'final_summary.json'}")
    return 0 if complete else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard-status")
    shard.add_argument("run_dir", type=Path)
    combined = subparsers.add_parser("aggregate")
    combined.add_argument("result_root", type=Path)
    args = parser.parse_args()
    if args.command == "shard-status":
        return shard_status(args.run_dir)
    return aggregate(args.result_root)


if __name__ == "__main__":
    sys.exit(main())
