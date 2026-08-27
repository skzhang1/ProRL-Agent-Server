from __future__ import annotations

from types import SimpleNamespace

from slime_bridge.config import AdaptiveHarnessSampler
from slime_bridge.rollout import _polar_extra_metrics, _update_hapo_sampler


def _sample(
    session_id: str,
    reward: float,
    *,
    status: str = "COMPLETED",
    placeholder: bool = False,
    harness: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        reward={"score": reward},
        metadata={
            "polar": {
                "session_id": session_id,
                "session_status": status,
                "placeholder": placeholder,
                "harness": harness,
                "timing": {
                    "register_to_init_queue_ms": 1.0,
                    "init_ms": 2.0,
                    "run_ms": 3.0,
                    "postrun_ms": 4.0,
                },
            },
        },
    )


def test_polar_reward_mean_completed_uses_unique_non_placeholder_sessions() -> None:
    samples = [
        _sample("completed-1", 1.0),
        _sample("completed-1", 1.0),
        _sample("completed-2", 0.0),
        _sample("timeout-1", 0.0, status="TIMEOUT", placeholder=True),
        _sample("empty-completed", 0.0, placeholder=True),
    ]
    metrics = _polar_extra_metrics(
        samples,
        rewards=[1.0, 1.0, 0.0, 0.0, 0.0],
        reward_key="score",
    )

    assert metrics["polar/reward_mean"] == 0.4
    assert metrics["polar/reward_mean_completed"] == 0.5
    assert metrics["polar/rollout_success_rate"] == 0.5


def test_polar_metrics_count_unique_sessions_per_harness() -> None:
    samples = [
        _sample("pi-1", 1.0, harness="pi"),
        _sample("pi-1", 1.0, harness="pi"),
        _sample("codex-1", 0.0, harness="codex"),
    ]

    metrics = _polar_extra_metrics(samples, rewards=[1.0, 1.0, 0.0], reward_key="score")

    assert metrics["polar/harness/pi/sessions"] == 1.0
    assert metrics["polar/harness/codex/sessions"] == 1.0



class _FakeDataSource:
    def __init__(self) -> None:
        self.metadata = {}

    def update_metadata(self, metadata: dict) -> None:
        self.metadata.update(metadata)


def test_hapo_update_counts_unique_sessions_and_failures_once() -> None:
    sampler = AdaptiveHarnessSampler(
        (
            {"harness": "pi", "model_name": "openai/model"},
            {"harness": "codex", "model_name": "model"},
        ),
        seed=87,
        epsilon=0.2,
        learning_rate=0.1,
        correct_threshold=0.5,
    )
    data_source = _FakeDataSource()
    worker = SimpleNamespace(
        harness_sampler=sampler,
        config=SimpleNamespace(reward_key="score"),
        data_source=data_source,
    )
    samples = [
        _sample("pi-ok", 1.0, harness="pi"),
        _sample("pi-ok", 1.0, harness="pi"),  # second trace, same session
        _sample("codex-timeout", 1.0, status="TIMEOUT", harness="codex"),
    ]

    metrics = _update_hapo_sampler(worker, samples)

    assert sampler.sample_counts == [1, 1]
    assert sampler.correct_counts == [1, 0]
    assert metrics["polar/hapo/pi/batch_sampled"] == 1.0
    assert metrics["polar/hapo/codex/batch_correct"] == 0.0
    assert data_source.metadata["polar_hapo_sampler"] == sampler.state_dict()
