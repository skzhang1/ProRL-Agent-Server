#!/usr/bin/env python3
"""Prepare SWE-bench Verified JSONL rows and Apptainer SIF symlinks.

This helper intentionally avoids the HuggingFace ``datasets`` dependency so it
can run inside the training container.  It fetches the 500 official test rows
from the HuggingFace dataset-server API, writes Slime-compatible JSONL rows, and
links each instance id to a matching SIF in the shared image store.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DATASET_CONFIG = "default"
DATASET_SPLIT = "test"
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_SHARED_SIF_DIR = Path(
    "/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/"
    "singularity_images_v3"
)
EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
DEFAULT_CACHE_PATH = EXAMPLE_DIR / "data" / "swebench_verified.json"
DEFAULT_OUTPUT_JSONL = EXAMPLE_DIR / "data" / "swebench_verified_eval.jsonl"
DEFAULT_SIF_DIR = EXAMPLE_DIR / "data" / "apptainer_image_links"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL containing the full selected range before resume "
            "filtering. Use this for combined summaries across retries."
        ),
    )
    parser.add_argument("--sif-dir", type=Path, default=DEFAULT_SIF_DIR)
    parser.add_argument("--shared-sif-dir", type=Path, default=DEFAULT_SHARED_SIF_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--refresh-dataset-cache", action="store_true")
    parser.add_argument(
        "--instance-range",
        default="",
        help=(
            "1-based inclusive dataset range to select, for example 1-100 or "
            "300-500. Comma-separated ranges are also accepted."
        ),
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=-1,
        help="Maximum tasks after range/id selection. -1 means no cap.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Restrict to one or more instance ids. Can be repeated.",
    )
    parser.add_argument(
        "--force-links",
        action="store_true",
        help="Replace existing symlinks in --sif-dir.",
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=None,
        help="Rollout result directory to inspect when --resume-completed is set.",
    )
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Skip instances that already have completed, non-error session results.",
    )
    parser.add_argument(
        "--completed-sessions-needed",
        type=int,
        default=1,
        help="Completed sessions required before an instance is skipped on resume.",
    )
    return parser.parse_args()


def _dataset_url(offset: int, length: int) -> str:
    params = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "offset": str(offset),
        "length": str(length),
    }
    return DATASET_ROWS_URL + "?" + urllib.parse.urlencode(params)


def _fetch_rows_from_dataset_server() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    total: int | None = None
    while total is None or offset < total:
        with urllib.request.urlopen(_dataset_url(offset, page_size), timeout=120) as response:
            payload = json.load(response)
        total = int(payload["num_rows_total"])
        rows.extend(item["row"] for item in payload["rows"])
        offset += page_size
    if len(rows) != total:
        raise RuntimeError(f"Expected {total} dataset rows, fetched {len(rows)}")
    return [_normalize_instance(row) for row in rows]


def load_instances(cache_path: Path, *, refresh: bool) -> list[dict[str, Any]]:
    if cache_path.is_file() and not refresh:
        return json.loads(cache_path.read_text())
    rows = _fetch_rows_from_dataset_server()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True, sort_keys=True))
    return rows


def _normalize_instance(instance: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(instance)
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = normalized.get(key)
        if isinstance(value, str):
            try:
                normalized[key] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                pass
    return normalized


def _parse_instance_range(value: str, total: int) -> set[int] | None:
    value = value.strip()
    if not value:
        return None

    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
        else:
            start = end = int(part)
        if start < 1 or end < start or end > total:
            raise SystemExit(
                f"Invalid --instance-range {value!r}: expected 1-based indices within 1-{total}"
            )
        selected.update(range(start, end + 1))
    if not selected:
        raise SystemExit(f"Invalid --instance-range {value!r}: no indices selected")
    return selected


def select_instances(
    args: argparse.Namespace,
    instances: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    selected = list(enumerate(instances, start=1))
    wanted_indices = _parse_instance_range(args.instance_range, len(instances))
    if wanted_indices is not None:
        selected = [(idx, row) for idx, row in selected if idx in wanted_indices]
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [(idx, row) for idx, row in selected if str(row.get("instance_id")) in wanted]
        missing = sorted(wanted - {str(row.get("instance_id")) for _, row in selected})
        if missing:
            raise SystemExit(f"Unknown instance_id(s): {', '.join(missing)}")
    if args.max_tasks > 0:
        selected = selected[: args.max_tasks]
    return selected


def _split_instance_id(instance_id: str) -> tuple[str, str, str]:
    owner, repo_with_issue = instance_id.split("__", 1)
    repo, issue_id = repo_with_issue.rsplit("-", 1)
    return owner, repo, issue_id


def candidate_sif_names(instance_id: str) -> list[str]:
    owner, repo, issue_id = _split_instance_id(instance_id)
    suffix = instance_id.replace("__", "_s_").lower()
    official = f"{owner.lower()}_1776_{repo.lower()}-{issue_id}"
    names = [
        f"xingyaoww_sweb.eval.x86_64.{suffix}.sif",
        f"docker.io_xingyaoww_sweb.eval.x86_64.{suffix}_latest.sif",
        f"docker.io_swebench_sweb.eval.x86_64.{official}_latest.sif",
        f"docker.io_swebench_sweb.eval.x86_64.{official}.sif",
        f"swebench_sweb.eval.x86_64.{official}_latest.sif",
        f"swebench_sweb.eval.x86_64.{official}.sif",
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def link_sif(instance_id: str, *, sif_dir: Path, shared_sif_dir: Path, force: bool) -> Path:
    target = sif_dir / f"{instance_id}.sif"
    if target.exists() or target.is_symlink():
        if not force:
            return target
        target.unlink()

    for name in candidate_sif_names(instance_id):
        source = shared_sif_dir / name
        if source.is_file():
            target.symlink_to(source)
            return target

    expected = "\n".join(f"  - {shared_sif_dir / name}" for name in candidate_sif_names(instance_id))
    raise FileNotFoundError(f"No SIF found for {instance_id}. Expected one of:\n{expected}")


def row_for_instance(dataset_index: int, instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "user", "content": str(instance["problem_statement"]).strip()}
        ],
        "label": "",
        "metadata": {
            "instance_id": str(instance["instance_id"]),
            "dataset_index": dataset_index,
            "instance": instance,
            "split": DATASET_SPLIT,
        },
    }


def _instance_for_task(task_id: str, instance_ids: list[str]) -> str | None:
    for instance_id in sorted(instance_ids, key=len, reverse=True):
        if instance_id in task_id:
            return instance_id
    return None


def _session_is_completed(session: dict[str, Any]) -> bool:
    if session.get("status") != "COMPLETED":
        return False
    if session.get("error"):
        return False
    trajectory = session.get("trajectory") or {}
    if trajectory.get("status") == "ERROR" or trajectory.get("error"):
        return False
    return True


def completed_instance_counts(rollout_dir: Path | None, instance_ids: list[str]) -> dict[str, int]:
    if rollout_dir is None or not rollout_dir.is_dir():
        return {}

    counts: dict[str, int] = {}
    for path in sorted(rollout_dir.glob("task_*/ses_*.json")):
        try:
            session = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not _session_is_completed(session):
            continue
        task_id = str(session.get("task_id") or path.parent.name.removeprefix("task_"))
        instance_id = _instance_for_task(task_id, instance_ids)
        if instance_id is not None:
            counts[instance_id] = counts.get(instance_id, 0) + 1
    return counts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n")
    else:
        path.write_text("")


def main() -> int:
    args = parse_args()
    instances = load_instances(args.cache_path, refresh=args.refresh_dataset_cache)
    selected = select_instances(args, instances)
    if not selected:
        raise SystemExit("No instances selected.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.sif_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    for _, instance in selected:
        link_sif(
            str(instance["instance_id"]),
            sif_dir=args.sif_dir,
            shared_sif_dir=args.shared_sif_dir,
            force=args.force_links,
        )
        linked += 1

    selected_rows = [row_for_instance(dataset_index, instance) for dataset_index, instance in selected]
    if args.manifest_jsonl is not None:
        write_jsonl(args.manifest_jsonl, selected_rows)

    completed_counts = completed_instance_counts(
        args.rollout_dir,
        [str(instance["instance_id"]) for _, instance in selected],
    )
    if args.resume_completed:
        needed = max(1, args.completed_sessions_needed)
        remaining = [
            (dataset_index, instance)
            for dataset_index, instance in selected
            if completed_counts.get(str(instance["instance_id"]), 0) < needed
        ]
    else:
        remaining = selected

    rows = [row_for_instance(dataset_index, instance) for dataset_index, instance in remaining]
    write_jsonl(args.output_jsonl, rows)

    print(f"Dataset: {DATASET_NAME}/{DATASET_SPLIT}")
    print(f"Selected rows: {len(selected_rows)}")
    print(f"Resume skipped rows: {len(selected_rows) - len(rows)}")
    print(f"Rows written: {len(rows)}")
    print(f"JSONL: {args.output_jsonl}")
    if args.manifest_jsonl is not None:
        print(f"Manifest JSONL: {args.manifest_jsonl}")
    print(f"SIF links: {linked}")
    print(f"SIF dir: {args.sif_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
