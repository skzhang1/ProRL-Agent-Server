#!/usr/bin/env python3
"""Scrape native SGLang Prometheus metrics and aggregate them by GPU node."""

import argparse
import csv
import json
import math
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

METRICS = (
    "prompt_tokens_total",
    "generation_tokens_total",
    "cached_tokens_total",
    "num_requests_total",
    "token_usage",
    "num_used_tokens",
    "kv_used_tokens",
    "kv_evictable_tokens",
    "kv_available_tokens",
    "num_running_reqs",
    "num_queue_reqs",
    "gen_throughput",
    "cache_hit_rate",
)
COUNTERS = {
    "prompt_tokens_total",
    "generation_tokens_total",
    "cached_tokens_total",
    "num_requests_total",
}
FIELDS = [
    "timestamp_utc", "unix_time", "node", "node_ip", "engine_count", "scrape_status",
    *METRICS,
    "uncached_prompt_tokens_total", "prompt_tokens_per_second",
    "generation_tokens_per_second", "cached_tokens_per_second",
]
SAMPLE_RE = re.compile(r"^([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)")
URL_RE = re.compile(r"^https?://[^\s]+$")


def fetch(url, timeout=2.0):
    req = Request(url, headers={"User-Agent": "trajectory-metrics-collector/1"})
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_worker_urls(value):
    found = set()
    def walk(item):
        if isinstance(item, str):
            candidate = item.rstrip("/,")
            if URL_RE.match(candidate):
                found.add(candidate)
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
    walk(value)
    return sorted(found)


def discover_workers(router_url):
    payload = json.loads(fetch(router_url.rstrip("/") + "/workers", timeout=3.0))
    return extract_worker_urls(payload)


def parse_prometheus(text):
    totals = {name: 0.0 for name in METRICS}
    seen = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        raw_name, raw_value = match.groups()
        name = raw_name.split(":")[-1]
        if name not in totals:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            totals[name] += value
            seen.add(name)
    return totals, seen


def resolve_host(url):
    host = urlparse(url).hostname or ""
    try:
        return socket.gethostbyname(host)
    except OSError:
        return host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--node-map", action="append", default=[], metavar="NODE=IP")
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    node_map = {}
    for item in args.node_map:
        node, sep, ip = item.partition("=")
        if not sep or not node or not ip:
            parser.error(f"invalid --node-map: {item}")
        node_map[ip] = node
    if not node_map:
        parser.error("at least one --node-map is required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not args.output.exists() or args.output.stat().st_size == 0
    previous = {}
    with args.output.open("a", newline="", encoding="utf-8", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        while True:
            started = time.time()
            timestamp = datetime.now(timezone.utc).isoformat()
            per_node = {
                node: {"engines": 0, "errors": 0, "seen": set(), **{m: 0.0 for m in METRICS}}
                for node in node_map.values()
            }
            router_error = None
            try:
                workers = discover_workers(args.router_url)
            except Exception as exc:
                workers = []
                router_error = f"router:{type(exc).__name__}"
            for worker in workers:
                ip = resolve_host(worker)
                node = node_map.get(ip)
                if node is None:
                    continue
                bucket = per_node[node]
                try:
                    values, seen = parse_prometheus(fetch(worker.rstrip("/") + "/metrics"))
                    bucket["engines"] += 1
                    bucket["seen"].update(seen)
                    for metric, value in values.items():
                        bucket[metric] += value
                except Exception:
                    bucket["errors"] += 1
            for ip, node in node_map.items():
                bucket = per_node[node]
                row = {
                    "timestamp_utc": timestamp,
                    "unix_time": f"{started:.6f}",
                    "node": node,
                    "node_ip": ip,
                    "engine_count": bucket["engines"],
                    "scrape_status": router_error or ("ok" if bucket["engines"] else "workers_not_ready"),
                }
                for metric in METRICS:
                    row[metric] = bucket[metric] if metric in bucket["seen"] else ""
                prompt = bucket["prompt_tokens_total"]
                cached = bucket["cached_tokens_total"]
                row["uncached_prompt_tokens_total"] = max(0.0, prompt - cached) if {"prompt_tokens_total", "cached_tokens_total"} <= bucket["seen"] else ""
                prior = previous.get(node)
                dt = started - prior["time"] if prior else 0.0
                for metric, rate_name in (
                    ("prompt_tokens_total", "prompt_tokens_per_second"),
                    ("generation_tokens_total", "generation_tokens_per_second"),
                    ("cached_tokens_total", "cached_tokens_per_second"),
                ):
                    if prior and dt > 0 and metric in bucket["seen"] and metric in prior["seen"]:
                        row[rate_name] = max(0.0, bucket[metric] - prior[metric]) / dt
                    else:
                        row[rate_name] = ""
                writer.writerow(row)
                previous[node] = {"time": started, "seen": set(bucket["seen"]), **{m: bucket[m] for m in COUNTERS}}
            elapsed = time.time() - started
            time.sleep(max(0.1, args.interval_seconds - elapsed))


if __name__ == "__main__":
    main()
