#!/usr/bin/env python3
"""Build the single normalized JSON artifact for a no-patch trajectory rollout."""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def nested(mapping, *path):
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first(mapping, paths):
    for path in paths:
        value = nested(mapping, *path)
        if value is not None:
            return value
    return None


def as_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def request_record(raw, source_file):
    response = raw.get("response") if isinstance(raw.get("response"), dict) else {}
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    choices = response.get("choices") if isinstance(response.get("choices"), list) else []
    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    meta = choice0.get("meta_info") if isinstance(choice0.get("meta_info"), dict) else {}
    request_id = response.get("id") or raw.get("request_id") or raw.get("completion_id") or source_file.stem
    prompt_tokens = as_number(usage.get("prompt_tokens"))
    decode_tokens = as_number(usage.get("completion_tokens"))
    cache_tokens = first(usage, [
        ("prompt_tokens_details", "cached_tokens"),
        ("cache_tokens",),
        ("cached_tokens",),
    ])
    if cache_tokens is None:
        cache_tokens = first(meta, [("cached_tokens",), ("cached_tokens_num",), ("prefix_cache_hit_tokens",)])
    prefill_latency = first(meta, [("prefill_latency",), ("prefill_time",), ("prefill_time_s",)])
    decode_latency = first(meta, [("decode_latency",), ("decode_time",), ("decode_time_s",)])
    start_time = first(raw, [("request_start_time",), ("start_time",)])
    end_time = first(raw, [("request_end_time",), ("end_time",), ("timestamp",), ("__written_at",)])
    duration = first(raw, [("duration_ms",), ("latency_ms",), ("elapsed_ms",)])
    return {
        "request_id": request_id,
        "session_id": raw.get("session_id"),
        "sequence_index": int(source_file.name.split("-", 1)[0]) if source_file.name.split("-", 1)[0].isdigit() else None,
        "start_time": start_time,
        "end_time": end_time,
        "duration_ms": as_number(duration),
        "prefill_latency_ms": as_number(prefill_latency),
        "decode_latency_ms": as_number(decode_latency),
        "prefill_token_count": prompt_tokens,
        "decode_token_count": decode_tokens,
        "cache_token_count": as_number(cache_tokens),
        "inference_node": None,
        "completion_id": raw.get("completion_id"),
    }


def valid_session(raw):
    trajectory = raw.get("trajectory")
    if not raw.get("session_id") or not isinstance(trajectory, dict):
        return False, "missing_session_or_trajectory"
    traces = trajectory.get("traces")
    if not isinstance(traces, list) or not traces:
        return False, "missing_traces"
    for trace in traces:
        if not isinstance(trace, dict):
            return False, "invalid_trace"
        if not isinstance(trace.get("prompt_messages"), list) or not trace["prompt_messages"]:
            return False, "missing_prompt_messages"
        if not isinstance(trace.get("response_messages"), list) or not trace["response_messages"]:
            return False, "missing_response_messages"
    return True, None


def load_metrics(path):
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            normalized = {}
            for key, value in row.items():
                if value == "":
                    normalized[key] = None
                elif key in {"timestamp_utc", "node", "node_ip", "scrape_status"}:
                    normalized[key] = value
                else:
                    normalized[key] = as_number(value)
            usage = normalized.get("token_usage")
            engines = normalized.get("engine_count")
            normalized["token_usage_mean_per_engine"] = usage / engines if usage is not None and engines else None
            rows.append(normalized)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=300)
    args = parser.parse_args()

    candidates = []
    rejected = []
    for path in sorted(args.rollout_dir.glob("task_*/ses_*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception as exc:
            rejected.append({"source_file": str(path), "reason": f"json_error:{type(exc).__name__}"})
            continue
        ok, reason = valid_session(raw)
        if not ok:
            rejected.append({"source_file": str(path), "reason": reason})
            continue
        meta = nested(raw, "trajectory", "metadata") or {}
        sort_key = (
            meta.get("rollout_step", 10**12),
            str(raw.get("task_id", "")),
            str(raw.get("session_id", "")),
        )
        candidates.append((sort_key, path))

    candidates.sort(key=lambda item: item[0])
    if len(candidates) < args.target:
        raise SystemExit(f"only {len(candidates)} structurally complete traces; need {args.target}")

    traces = []
    for _, path in candidates[: args.target]:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        session_id = raw["session_id"]
        completion_dir = path.parent / "sessions" / session_id / "completions"
        requests = []
        for completion_path in sorted(completion_dir.glob("*.json")):
            try:
                with completion_path.open(encoding="utf-8") as handle:
                    completion = json.load(handle)
                requests.append(request_record(completion, completion_path))
            except Exception as exc:
                requests.append({
                    "request_id": completion_path.stem,
                    "source_file": str(completion_path),
                    "parse_error": type(exc).__name__,
                })
        trace_meta = nested(raw, "trajectory", "metadata") or {}
        traces.append({
            "session_id": session_id,
            "task_id": raw.get("task_id"),
            "status": raw.get("status"),
            "termination_reason": nested(raw, "metadata", "termination_reason"),
            "rollout_step": trace_meta.get("rollout_step"),
            "policy_version": trace_meta.get("policy_version"),
            "group_id": trace_meta.get("group_id"),
            "gateway_node_id": raw.get("node_id"),
            "session_timing_ms": raw.get("timing"),
            "request_count": len(requests),
            "requests": requests,
            "trajectory": raw.get("trajectory"),
            "session_error": raw.get("error"),
            "session_metadata": raw.get("metadata"),
            "source_file": str(path),
        })

    metrics = load_metrics(args.metrics_csv)
    artifact = {
        "schema_version": "tmax-agent-traces-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trace_count": len(traces),
        "candidate_trace_count": len(candidates),
        "rejected_trace_count": len(rejected),
        "selection": "first 300 structurally complete sessions ordered by rollout_step, task_id, session_id",
        "complete_trace_definition": "session JSON parses, trajectory.traces is non-empty, and every trace has non-empty prompt_messages and response_messages",
        "rollout_mode": {
            "training_enabled": False,
            "interactive_nodes": 2,
            "gpus_per_node": 8,
            "runtime_patching": False,
        },
        "token_usage_definition": {
            "sampling_interval_seconds": 5,
            "prompt_tokens_total": "Native SGLang cumulative prompt-token counter summed across engines on the GPU node.",
            "cached_tokens_total": "Native SGLang cumulative prefix-cache-hit token counter summed across engines on the GPU node.",
            "uncached_prompt_tokens_total": "prompt_tokens_total minus cached_tokens_total.",
            "generation_tokens_total": "Native SGLang cumulative generated-token counter summed across engines on the GPU node.",
            "token_usage": "Native SGLang instantaneous KV token-pool usage gauge summed across engines; engine_count is included for interpretation.",
            "token_usage_mean_per_engine": "Node token_usage divided by engine_count, a 0..1 mean engine KV-pool occupancy ratio.",
            "rates": "Five-second counter deltas divided by actual elapsed wall time.",
        },
        "metric_provenance": {
            "request_level": "Polar completion/session JSON emitted by the existing gateway and builder.",
            "node_level": "Native SGLang /metrics endpoints discovered from the router /workers endpoint; no instrumentation or source patch.",
            "request_node_join": "gateway_node_id identifies the Polar gateway only; exact inference-engine/node routing is not exposed, so inference_node remains null.",
        },
        "known_unavailable_fields": {
            "prefill_latency_ms": "Polar normalization does not persist native SGLang per-request prefill timing.",
            "decode_latency_ms": "Polar normalization does not persist native SGLang per-request decode timing.",
            "request_start_time": "Completion records expose write/end time but not request admission time.",
            "request_duration_ms": "Cannot be derived exactly without request start time.",
            "cache_token_count": "Filled only if the unmodified backend response exposes it; otherwise null.",
            "inference_node": "Router does not persist request-to-engine assignment in Polar completion JSON.",
            "trajectory_token_ids": "Unmodified SGLang 0.5.13 router does not preserve token IDs in Polar traces; full prompt/response message trajectories and per-request usage token counts are preserved.",
        },
        "node_token_usage_samples": metrics,
        "traces": traces,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    os.replace(temp, args.output)
    print(json.dumps({
        "output": str(args.output),
        "trace_count": len(traces),
        "request_count": sum(t["request_count"] for t in traces),
        "node_metric_samples": len(metrics),
        "bytes": args.output.stat().st_size,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
