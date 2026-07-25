#!/usr/bin/env python3
"""Aggregate strict pass@1 SWE-bench shard outputs.

This script scores each instance once. It uses the first available session for an
instance by mtime/path across counted run directories. Missing or evaluator failures count as incorrect. Agent wrapper exit status is kept
for diagnostics but does not override a completed official evaluator result.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--expected-total", type=int, default=500)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise SystemExit(f"failed to parse {path}:{line_no}: {exc}") from exc
    return rows


def instance_id_from_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("instance_id"):
        return str(metadata["instance_id"])
    instance = metadata.get("instance") or {}
    if instance.get("instance_id"):
        return str(instance["instance_id"])
    raise SystemExit(f"manifest row missing instance id: keys={sorted(row)}")


def dataset_index_from_row(row: dict[str, Any], fallback: int) -> int:
    metadata = row.get("metadata") or {}
    try:
        return int(metadata.get("dataset_index"))
    except Exception:
        return fallback


def reward_from_session(session: dict[str, Any]) -> float | None:
    trajectory = session.get("trajectory") or {}
    metadata = trajectory.get("metadata") or {}
    evaluation = metadata.get("evaluation") or {}
    if evaluation.get("outcome_reward") is not None:
        try:
            return float(evaluation["outcome_reward"])
        except Exception:
            return None
    rewards: list[float] = []
    for trace in trajectory.get("traces") or []:
        if isinstance(trace, dict) and trace.get("reward") is not None:
            try:
                rewards.append(float(trace["reward"]))
            except Exception:
                pass
    return max(rewards) if rewards else None


def safe_load_session(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, str(exc)


def classify_first_session(path: Path) -> dict[str, Any]:
    session, parse_error = safe_load_session(path)
    base: dict[str, Any] = {
        "first_session_path": path.as_posix(),
        "parse_error": parse_error,
        "session_id": None,
        "task_id": None,
        "session_status": "PARSE_ERROR" if parse_error else None,
        "session_error": parse_error,
        "trajectory_status": None,
        "trajectory_error": None,
        "outcome_reward": None,
        "resolved": False,
        "clean_completed": False,
        "failure_kind": "PARSE_ERROR" if parse_error else "UNKNOWN",
    }
    if session is None:
        return base

    trajectory = session.get("trajectory") or {}
    metadata = trajectory.get("metadata") or {}
    evaluation = metadata.get("evaluation") or {}
    report = evaluation.get("report") or {}
    session_status = str(session.get("status") or "UNKNOWN").upper()
    trajectory_status = str(trajectory.get("status") or "").upper() or None
    session_error = session.get("error")
    trajectory_error = trajectory.get("error")
    error_text = " ".join(
        str(part)
        for part in [session_status, trajectory_status, session_error, trajectory_error]
        if part
    ).lower()
    reward = reward_from_session(session)
    clean_completed = (
        session_status == "COMPLETED"
        and not session_error
        and trajectory_status not in {"ERROR", "FAILED", "TIMEOUT"}
        and not trajectory_error
    )
    resolved = bool(reward == 1.0)
    if resolved:
        failure_kind = "RESOLVED"
    elif "timeout" in error_text or report.get("test_timeout"):
        failure_kind = "TIMEOUT"
    elif not clean_completed:
        failure_kind = session_status if session_status not in {"", "UNKNOWN"} else "ERROR"
    else:
        failure_kind = "UNRESOLVED"

    base.update(
        {
            "session_id": session.get("session_id"),
            "task_id": session.get("task_id"),
            "session_status": session_status,
            "session_error": session_error,
            "trajectory_status": trajectory_status,
            "trajectory_error": trajectory_error,
            "outcome_reward": reward,
            "resolved": resolved,
            "clean_completed": clean_completed,
            "failure_kind": failure_kind,
        }
    )
    return base


def discover_run_dirs(project_root: Path, run_group: str, explicit: list[Path]) -> list[Path]:
    if explicit:
        return sorted((p if p.is_absolute() else project_root / p) for p in explicit)
    results_root = project_root / "examples" / "swebench_verified" / "results"
    dirs = []
    for path in results_root.glob(f"{run_group}*"):
        if not path.is_dir():
            continue
        if path.name == run_group:
            continue
        if (path / "swebench_verified_selected.jsonl").is_file():
            dirs.append(path)
    return sorted(dirs)


def collect_manifests(run_dirs: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[int, list[str]]]:
    by_instance: dict[str, dict[str, Any]] = {}
    by_dataset: dict[int, list[str]] = {}
    for run_dir in run_dirs:
        manifest = run_dir / "swebench_verified_selected.jsonl"
        for ordinal, row in enumerate(load_jsonl(manifest), 1):
            iid = instance_id_from_row(row)
            dataset_index = dataset_index_from_row(row, ordinal)
            by_dataset.setdefault(dataset_index, []).append(iid)
            entry = by_instance.setdefault(
                iid,
                {
                    "instance_id": iid,
                    "dataset_index": dataset_index,
                    "manifest_source_run_ids": [],
                },
            )
            entry["dataset_index"] = min(int(entry["dataset_index"]), dataset_index)
            entry["manifest_source_run_ids"].append(run_dir.name)
    return by_instance, by_dataset


def collect_sessions(run_dirs: list[Path], instance_ids: list[str]) -> tuple[dict[str, list[Path]], list[Path]]:
    by_instance: dict[str, list[Path]] = {iid: [] for iid in instance_ids}
    unknown: list[Path] = []
    sorted_ids = sorted(instance_ids, key=len, reverse=True)
    for run_dir in run_dirs:
        rollout_dir = run_dir / "rollout_results"
        if not rollout_dir.exists():
            continue
        for path in rollout_dir.rglob("ses_*.json"):
            text = path.as_posix()
            matched = None
            for iid in sorted_ids:
                if iid in text:
                    matched = iid
                    break
            if matched is None:
                unknown.append(path)
            else:
                by_instance[matched].append(path)
    for paths in by_instance.values():
        paths.sort(key=lambda p: (p.stat().st_mtime_ns, p.as_posix()))
    return by_instance, unknown


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    run_dirs = discover_run_dirs(project_root, args.run_group, args.run_dir)
    if not run_dirs:
        raise SystemExit(f"no shard run dirs found for run group {args.run_group!r}")

    by_instance, by_dataset = collect_manifests(run_dirs)
    session_paths, unknown_sessions = collect_sessions(run_dirs, list(by_instance))

    results: list[dict[str, Any]] = []
    for iid, manifest_entry in sorted(by_instance.items(), key=lambda kv: (kv[1]["dataset_index"], kv[0])):
        paths = session_paths.get(iid, [])
        if paths:
            item = classify_first_session(paths[0])
        else:
            item = {
                "first_session_path": None,
                "parse_error": None,
                "session_id": None,
                "task_id": None,
                "session_status": "MISSING",
                "session_error": None,
                "trajectory_status": None,
                "trajectory_error": None,
                "outcome_reward": None,
                "resolved": False,
                "clean_completed": False,
                "failure_kind": "MISSING",
            }
        item.update(
            {
                "dataset_index": manifest_entry["dataset_index"],
                "instance_id": iid,
                "score": 1.0 if item["resolved"] else 0.0,
                "session_count": len(paths),
                "source_run_id": paths[0].relative_to(project_root / "examples" / "swebench_verified" / "results").parts[0] if paths else None,
                "manifest_source_run_ids": sorted(set(manifest_entry["manifest_source_run_ids"])),
                "extra_session_paths": [p.as_posix() for p in paths[1:]],
            }
        )
        results.append(item)

    expected_indices = set(range(1, args.expected_total + 1))
    observed_indices = {int(item["dataset_index"]) for item in results}
    missing_indices = sorted(expected_indices - observed_indices)
    duplicate_indices = sorted(idx for idx, ids in by_dataset.items() if len(set(ids)) > 1)
    manifest_conflicts = [
        {"dataset_index": idx, "instance_ids": sorted(set(ids))}
        for idx, ids in sorted(by_dataset.items())
        if len(set(ids)) > 1
    ]

    total = len(results)
    resolved = sum(1 for item in results if item["resolved"])
    attempted = sum(1 for item in results if item["session_count"] > 0)
    clean_completed = sum(1 for item in results if item["clean_completed"])
    timeouts = sum(1 for item in results if item["failure_kind"] == "TIMEOUT")
    missing = sum(1 for item in results if item["failure_kind"] == "MISSING")
    errors = sum(
        1
        for item in results
        if item["failure_kind"] in {"ERROR", "FAILED", "PARSE_ERROR"}
        or (item["session_status"] not in {"COMPLETED", "MISSING"} and item["failure_kind"] != "TIMEOUT")
    )
    duplicates = sum(1 for item in results if item["session_count"] > 1)
    validation_ok = (
        total == args.expected_total
        and len(observed_indices) == args.expected_total
        and not missing_indices
        and not duplicate_indices
        and not manifest_conflicts
    )
    summary = {
        "metric": "strict_pass_at_1",
        "run_group": args.run_group,
        "total_tasks": total,
        "expected_total_tasks": args.expected_total,
        "attempted_tasks": attempted,
        "clean_completed_tasks": clean_completed,
        "resolved_tasks": resolved,
        "pass_at_1": (resolved / total) if total else 0.0,
        "timeout_tasks": timeouts,
        "error_tasks": errors,
        "missing_tasks": missing,
        "duplicate_session_tasks": duplicates,
        "unique_instance_ids": len({item["instance_id"] for item in results}),
        "unique_dataset_indices": len(observed_indices),
        "missing_dataset_indices": missing_indices,
        "duplicate_dataset_indices": duplicate_indices,
        "manifest_conflicts": manifest_conflicts,
        "unknown_session_paths_count": len(unknown_sessions),
        "counted_run_count": len(run_dirs),
        "counted_run_ids": [p.name for p in run_dirs],
        "validation_ok": validation_ok,
        "scoring_rule": "global strict pass@1 from official evaluator outcome_reward; first session per instance; missing or evaluator failure counts as 0; agent exit status is diagnostic; extra sessions ignored and reported",
        "generated_at_unix": time.time(),
    }

    group_dir = project_root / "examples" / "swebench_verified" / "results" / args.run_group
    group_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "instances": results, "unknown_session_paths": [p.as_posix() for p in unknown_sessions]}
    (group_dir / "aggregate_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (group_dir / "aggregate_results.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    fieldnames = [
        "dataset_index",
        "instance_id",
        "score",
        "resolved",
        "failure_kind",
        "session_count",
        "session_status",
        "outcome_reward",
        "first_session_path",
        "source_run_id",
        "session_id",
        "task_id",
        "session_error",
        "trajectory_status",
        "trajectory_error",
    ]
    with (group_dir / "aggregate_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key) for key in fieldnames})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"aggregate_summary={group_dir / 'aggregate_summary.json'}")
    print(f"aggregate_results_jsonl={group_dir / 'aggregate_results.jsonl'}")
    print(f"aggregate_results_csv={group_dir / 'aggregate_results.csv'}")
    return 0 if validation_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
