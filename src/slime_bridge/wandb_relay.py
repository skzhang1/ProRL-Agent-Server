"""Single-writer W&B relay for distributed Slime training.

W&B's shared mode lets every Ray process attach to one run, but that mode is
experimental and has lost otherwise valid rollout history on NRT.  This relay
keeps exactly one ordinary online W&B client in a zero-GPU Ray actor.  Every
producer waits for its log call to reach that actor, so a telemetry failure is
visible to the training job instead of silently dropping the HAPO history.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Mapping
from typing import Any


_RELAY_NAME_PREFIX = "slime-wandb-relay"
_cached_handle: Any | None = None
_cached_name: str | None = None
_handle_lock = threading.Lock()


def single_owner_enabled() -> bool:
    return os.environ.get("SLIME_WANDB_SINGLE_OWNER", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def relay_actor_name(args: Any | None = None) -> str:
    configured = os.environ.get("SLIME_WANDB_RELAY_NAME", "").strip()
    if configured:
        suffix = configured
    else:
        run_id = getattr(args, "wandb_run_id", None) if args is not None else None
        suffix = str(run_id or os.environ.get("SLURM_JOB_ID") or "default")
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "-", suffix).strip("-") or "default"
    return f"{_RELAY_NAME_PREFIX}-{suffix}"


class _WandbRelay:
    def __init__(self, args: Any) -> None:
        from slime.utils import wandb_utils

        self._finished = False
        wandb_utils.init_wandb_primary(args)

    def ready(self) -> dict[str, str | None]:
        import wandb

        run = wandb.run
        return {
            "id": None if run is None else run.id,
            "name": None if run is None else run.name,
            "url": None if run is None else run.url,
        }

    def log(self, metrics: dict[str, Any], step_key: str) -> None:
        if self._finished:
            raise RuntimeError("cannot log after the W&B relay has finished")
        import wandb
        from slime.utils import wandb_utils

        wandb_utils.define_logged_metric_axes(metrics, step_metric=step_key)
        wandb.log(metrics)

    def finish(self) -> None:
        if self._finished:
            return
        import wandb

        if wandb.run is not None:
            wandb.finish()
        self._finished = True


def start_relay(args: Any) -> tuple[Any, dict[str, str | None]]:
    """Create the sole W&B writer and wait until its cloud run is initialized."""
    import ray

    global _cached_handle, _cached_name
    name = relay_actor_name(args)
    with _handle_lock:
        if _cached_handle is None or _cached_name != name:
            remote_cls = ray.remote(num_cpus=0)(_WandbRelay)
            _cached_handle = remote_cls.options(name=name).remote(args)
            _cached_name = name
        handle = _cached_handle
    return handle, ray.get(handle.ready.remote())


def get_relay(args: Any) -> Any:
    import ray

    global _cached_handle, _cached_name
    name = relay_actor_name(args)
    with _handle_lock:
        if _cached_handle is None or _cached_name != name:
            _cached_handle = ray.get_actor(name)
            _cached_name = name
        return _cached_handle


def _to_transport_value(value: Any) -> Any:
    """Detach accelerator-backed metrics before Ray serializes them."""
    try:
        import torch

        if torch.is_tensor(value):
            tensor = value.detach()
            return tensor.item() if tensor.numel() == 1 else tensor.cpu().tolist()
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass

    if isinstance(value, Mapping):
        return {key: _to_transport_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_transport_value(item) for item in value)
    if isinstance(value, list):
        return [_to_transport_value(item) for item in value]
    return value


def log_via_relay(args: Any, metrics: dict[str, Any], step_key: str) -> None:
    """Synchronously hand a metric row to the sole writer actor."""
    import ray

    handle = get_relay(args)
    payload = _to_transport_value(dict(metrics))
    ray.get(handle.log.remote(payload, str(step_key)))


def finish_relay(handle: Any) -> None:
    import ray

    ray.get(handle.finish.remote())
