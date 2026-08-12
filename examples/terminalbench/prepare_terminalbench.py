#!/usr/bin/env python3
"""Prepare pinned Terminal-Bench 2.1 rows and per-task Apptainer images."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any


DATASET_REPO = "https://github.com/harbor-framework/terminal-bench-2-1.git"
# Last main-branch commit before the Tmax v1 paper was released. Task contents
# are identical to the current release, but pinning makes the evaluation exact.
DATASET_COMMIT = "c5ee500c185224c97cd6caff7866a990a0057f41"
EXPECTED_TASKS = 89
TASK_TIMEOUT_SECONDS = 1500.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "data" / "terminal-bench-2-1"
DEFAULT_SIF_ROOT = Path(
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/"
    "shaokunz/HarnessGen/terminal_bench_sif"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--manifest-jsonl", type=Path)
    parser.add_argument("--sif-dir", type=Path)
    parser.add_argument("--shared-sif-dir", type=Path, default=DEFAULT_SIF_ROOT)
    parser.add_argument("--cache-path", type=Path, help="Compatibility argument from the shared eval launcher.")
    parser.add_argument("--instance-range", default="")
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--rollout-dir", type=Path)
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--completed-sessions-needed", type=int, default=1)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ensure_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", DATASET_REPO, str(dataset_dir)])
    if not (dataset_dir / ".git").is_dir():
        raise SystemExit(f"Dataset path exists but is not a git checkout: {dataset_dir}")
    head = subprocess.check_output(
        ["git", "-C", str(dataset_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != DATASET_COMMIT:
        run(["git", "-C", str(dataset_dir), "checkout", "--detach", DATASET_COMMIT])
    count = len(list((dataset_dir / "tasks").glob("*/task.toml")))
    if count != EXPECTED_TASKS:
        raise SystemExit(f"Expected {EXPECTED_TASKS} tasks at {DATASET_COMMIT}, found {count}")


def ordered_task_dirs(dataset_dir: Path) -> list[Path]:
    manifest = tomllib.loads((dataset_dir / "tasks" / "dataset.toml").read_text())
    result = []
    for item in manifest["tasks"]:
        name = str(item["name"]).split("/", 1)[-1]
        task_dir = dataset_dir / "tasks" / name
        if not (task_dir / "task.toml").is_file():
            raise SystemExit(f"Manifest task is missing: {task_dir}")
        result.append(task_dir)
    if len(result) != EXPECTED_TASKS:
        raise SystemExit(f"Manifest contains {len(result)} tasks, expected {EXPECTED_TASKS}")
    return result


def parse_range(value: str, total: int) -> set[int] | None:
    if not value.strip():
        return None
    selected: set[int] = set()
    for part in value.split(","):
        bounds = part.strip().split("-", 1)
        start = int(bounds[0])
        end = int(bounds[-1])
        if start < 1 or end < start or end > total:
            raise SystemExit(f"Invalid --instance-range {value!r}; expected indices within 1-{total}")
        selected.update(range(start, end + 1))
    return selected


def select_tasks(args: argparse.Namespace, tasks: list[Path]) -> list[tuple[int, Path]]:
    selected = list(enumerate(tasks, start=1))
    indices = parse_range(args.instance_range, len(tasks))
    if indices is not None:
        selected = [(index, path) for index, path in selected if index in indices]
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [(index, path) for index, path in selected if path.name in wanted]
        missing = wanted - {path.name for _, path in selected}
        if missing:
            raise SystemExit(f"Unknown task name(s): {', '.join(sorted(missing))}")
    if args.max_tasks > 0:
        selected = selected[: args.max_tasks]
    return selected


def apptainer_binary() -> str:
    override = os.environ.get("POLAR_APPTAINER_BIN")
    if override:
        return override
    return shutil.which("apptainer") or "apptainer"


def ensure_sif(task_name: str, image_ref: str, sif_root: Path) -> Path:
    target = sif_root / f"{task_name}.sif"
    if target.is_file() and target.stat().st_size > 0:
        return target
    sif_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.environ.get("APPTAINER_CACHEDIR", sif_root / ".cache"))
    tmp_dir = Path(os.environ.get("APPTAINER_TMPDIR", sif_root / ".tmp"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    auth_file = tmp_dir / "anonymous-auth.json"
    auth_file.write_text("{\"auths\": {}}\n")
    env = {**os.environ, "APPTAINER_CACHEDIR": str(cache_dir), "APPTAINER_TMPDIR": str(tmp_dir)}
    with tempfile.NamedTemporaryFile(prefix=f".{task_name}.", suffix=".sif", dir=sif_root, delete=False) as handle:
        temporary = Path(handle.name)
    temporary.unlink()
    uri = image_ref if "://" in image_ref else f"docker://{image_ref}"
    try:
        print(f"+ {apptainer_binary()} pull --authfile {auth_file} --force {temporary} {uri}", flush=True)
        subprocess.run([apptainer_binary(), "pull", "--authfile", str(auth_file), "--force", str(temporary), uri], check=True, env=env)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def final_workdir(task_dir: Path) -> str:
    workdir = "/app"
    for line in (task_dir / "environment" / "Dockerfile").read_text().splitlines():
        match = re.match(r"\s*WORKDIR\s+(\S+)", line, flags=re.IGNORECASE)
        if match:
            workdir = match.group(1)
    if not workdir.startswith("/"):
        raise SystemExit(f"Unsupported relative WORKDIR for {task_dir.name}: {workdir}")
    return workdir


def attempted_tasks(rollout_dir: Path | None, names: list[str]) -> set[str]:
    if rollout_dir is None or not rollout_dir.is_dir():
        return set()
    attempted: set[str] = set()
    longest_first = sorted(names, key=len, reverse=True)
    for path in rollout_dir.rglob("ses_*.json"):
        try:
            session = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        text = f"{path.as_posix()} {session.get('task_id', '')}"
        for name in longest_first:
            if name in text:
                attempted.add(name)
                break
    return attempted


def task_row(index: int, task_dir: Path, sif_path: Path) -> dict[str, Any]:
    config = tomllib.loads((task_dir / "task.toml").read_text())
    environment = config["environment"]
    verifier_timeout = float(config.get("verifier", {}).get("timeout_sec", 120.0))
    return {
        "prompt": [{"role": "user", "content": (task_dir / "instruction.md").read_text().strip()}],
        "label": "",
        "metadata": {
            "instance_id": task_dir.name,
            "dataset_index": index,
            "task_dir": str(task_dir.resolve()),
            "tests_dir": str((task_dir / "tests").resolve()),
            "sif_path": str(sif_path.resolve()),
            "docker_image": str(environment["docker_image"]),
            "dataset_commit": DATASET_COMMIT,
            "timeout_seconds": TASK_TIMEOUT_SECONDS,
            "verifier_timeout": verifier_timeout,
            "memory_mb": int(environment["memory_mb"]),
            "allow_internet": bool(environment["allow_internet"]),
            "workdir": final_workdir(task_dir),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    ensure_dataset(args.dataset_dir.resolve())
    tasks = ordered_task_dirs(args.dataset_dir.resolve())
    print(f"Terminal-Bench 2.1: commit={DATASET_COMMIT}, tasks={len(tasks)}")
    if args.download_only:
        return 0
    if args.output_jsonl is None or args.manifest_jsonl is None or args.sif_dir is None:
        raise SystemExit("--output-jsonl, --manifest-jsonl and --sif-dir are required")

    selected = select_tasks(args, tasks)
    attempted = attempted_tasks(args.rollout_dir, [path.name for _, path in selected]) if args.resume_completed else set()
    manifest_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    args.sif_dir.mkdir(parents=True, exist_ok=True)
    for index, task_dir in selected:
        config = tomllib.loads((task_dir / "task.toml").read_text())
        shared_sif = ensure_sif(task_dir.name, str(config["environment"]["docker_image"]), args.shared_sif_dir)
        run_sif = args.sif_dir / shared_sif.name
        if run_sif.is_symlink() or run_sif.exists():
            if run_sif.resolve() != shared_sif.resolve():
                raise SystemExit(f"Unexpected existing SIF link: {run_sif}")
        else:
            run_sif.symlink_to(shared_sif)
        row = task_row(index, task_dir, run_sif)
        manifest_rows.append(row)
        if task_dir.name not in attempted:
            eval_rows.append(row)

    write_jsonl(args.manifest_jsonl, manifest_rows)
    write_jsonl(args.output_jsonl, eval_rows)
    print(f"selected={len(manifest_rows)} attempted={len(attempted)} remaining={len(eval_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
