"""Small Slime eval hooks for persisted Polar SWE-bench sessions."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


def log_eval_rollout(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None = None,
) -> bool:
    """Log eval counts safely when agent traces have no trainable token arrays.

    Polar persists authoritative SWE-bench rewards in ``ses_*.json``. PI tool
    traces may intentionally omit Slime token arrays, leaving ``rewards`` empty;
    Slime's default logger divides by that empty length. Returning true tells
    Slime that this hook handled logging without changing any result data.
    """
    summary: dict[str, Any] = {"rollout_id": rollout_id}
    for name, values in data.items():
        rewards = list(values.get("rewards") or [])
        all_rewards = list(values.get("all_rewards") or [])
        summary[name] = {
            "completed_token_samples": len(rewards),
            "persisted_sessions": len(all_rewards),
            "mean_persisted_reward": (
                sum(float(value) for value in all_rewards) / len(all_rewards)
                if all_rewards
                else None
            ),
        }
    if extra_metrics:
        summary["extra_metrics"] = extra_metrics
    logger.info("SWE-bench eval summary: %s", summary)
    return True
