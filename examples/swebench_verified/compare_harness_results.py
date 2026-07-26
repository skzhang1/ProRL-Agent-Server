#!/usr/bin/env python3
"""Compare checkpoint versus base results within one SWE-bench harness."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--harness", required=True)
    p.add_argument("--base-summary", type=Path, required=True)
    p.add_argument("--checkpoint-summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap-samples", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=73)
    return p.parse_args()


def load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload["summary"].get("validation_ok"):
        raise SystemExit(f"aggregate validation failed: {path}")
    rows = {row["instance_id"]: row for row in payload["instances"]}
    if len(rows) != len(payload["instances"]):
        raise SystemExit(f"duplicate instance IDs in {path}")
    return payload["summary"], rows


def percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def bootstrap_ci(differences, samples, seed):
    if not differences or samples <= 0:
        return None
    rng, n = random.Random(seed), len(differences)
    estimates = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def mcnemar_p(base_only, checkpoint_only):
    discordant = base_only + checkpoint_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(base_only, checkpoint_only) + 1))
    return min(1.0, 2.0 * tail / 2**discordant)


def compare(ids, base, checkpoint, samples, seed):
    before = [int(bool(base[i]["resolved"])) for i in ids]
    after = [int(bool(checkpoint[i]["resolved"])) for i in ids]
    differences = [a - b for b, a in zip(before, after)]
    total, base_n, checkpoint_n = len(ids), sum(before), sum(after)
    base_rate = base_n / total if total else 0.0
    checkpoint_rate = checkpoint_n / total if total else 0.0
    delta = checkpoint_rate - base_rate
    both = sum(b and a for b, a in zip(before, after))
    base_only = sum(b and not a for b, a in zip(before, after))
    checkpoint_only = sum(not b and a for b, a in zip(before, after))
    return {
        "total_tasks": total,
        "base_resolved_tasks": base_n,
        "checkpoint_resolved_tasks": checkpoint_n,
        "base_pass_at_1": base_rate,
        "checkpoint_pass_at_1": checkpoint_rate,
        "absolute_delta": delta,
        "absolute_delta_percentage_points": delta * 100,
        "relative_delta": delta / base_rate if base_rate else None,
        "relative_delta_percent": delta / base_rate * 100 if base_rate else None,
        "paired_transitions": {
            "both_resolved": both,
            "base_only_resolved": base_only,
            "checkpoint_only_resolved": checkpoint_only,
            "neither_resolved": total - both - base_only - checkpoint_only,
        },
        "paired_bootstrap_95pct_ci_absolute_delta": bootstrap_ci(differences, samples, seed),
        "exact_mcnemar_two_sided_p": mcnemar_p(base_only, checkpoint_only),
    }


def main():
    args = parse_args()
    base_summary, base = load(args.base_summary)
    checkpoint_summary, checkpoint = load(args.checkpoint_summary)
    if set(base) != set(checkpoint):
        raise SystemExit("base and checkpoint instance sets differ")
    ids = sorted(base, key=lambda i: (int(base[i]["dataset_index"]), i))
    overall = compare(ids, base, checkpoint, args.bootstrap_samples, args.seed)
    repositories = sorted({row.get("repository") or iid.split("__", 1)[0] for iid, row in base.items()})
    by_repository = {}
    for offset, repository in enumerate(repositories, 1):
        repo_ids = [i for i in ids if (base[i].get("repository") or i.split("__", 1)[0]) == repository]
        by_repository[repository] = compare(
            repo_ids, base, checkpoint, args.bootstrap_samples, args.seed + offset
        )
    payload = {
        "harness": args.harness,
        "comparison_scope": "checkpoint versus base within the same harness",
        "base_run_group": base_summary["run_group"],
        "checkpoint_run_group": checkpoint_summary["run_group"],
        "overall": overall,
        "by_repository": by_repository,
        "failure_kinds": {
            "base": dict(sorted(Counter(x["failure_kind"] for x in base.values()).items())),
            "checkpoint": dict(sorted(Counter(x["failure_kind"] for x in checkpoint.values()).items())),
        },
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.harness}_base_vs_checkpoint.json"
    csv_path = args.output_dir / f"{args.harness}_by_repository.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["repository", "total_tasks", "base_resolved_tasks", "checkpoint_resolved_tasks",
              "base_pass_at_1", "checkpoint_pass_at_1", "absolute_delta_percentage_points",
              "relative_delta_percent", "base_only_resolved", "checkpoint_only_resolved"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for repository, stats in by_repository.items():
            transitions = stats["paired_transitions"]
            writer.writerow({**{k: stats.get(k) for k in fields if k != "repository"},
                             "repository": repository,
                             "base_only_resolved": transitions["base_only_resolved"],
                             "checkpoint_only_resolved": transitions["checkpoint_only_resolved"]})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"comparison_json={json_path}")
    print(f"repository_csv={csv_path}")


if __name__ == "__main__":
    main()
