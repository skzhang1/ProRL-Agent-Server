from __future__ import annotations

from types import SimpleNamespace

import pytest

from slime_bridge.config import (
    AdaptiveHarnessSampler,
    render_instruction,
    render_task_payload,
    render_topology_template,
    resolve_polar_slime_config,
    resolve_sglang_router_base_url,
)


def _args(**overrides):
    base = {
        "polar_rollout_url": "http://rollout:8080/",
        "polar_task_template": {
            "agent": {"harness": "codex", "model_name": "{args.model_name}"},
            "runtime": {"image": "{sample.metadata.image}"},
            "metadata": {"instance": "{sample.metadata.instance_id}"},
        },
        "polar_task_id_template": "task-{rollout_id}-{sample.group_index}",
        "polar_instruction_template": "Instruction: {instruction}",
        "polar_reward_key": "score",
        "polar_max_async_level": 2,
        "rollout_batch_size": 3,
        "n_samples_per_prompt": 4,
        "update_weights_interval": 5,
        "polar_request_timeout": 60,
        "polar_callback_host": "127.0.0.1",
        "polar_scoring_mode": "group",
        "polar_min_complete_accept_fraction": 0.0,
        "polar_early_stop_grace_sessions": 2,
        "hf_checkpoint": "tokenizer-name",
        "polar_add_generation_prompt": True,
        "polar_eval_dataset_name": "eval",
        "model_name": "openai/gpt-test",
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_port": 30000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_resolve_polar_slime_config_computes_concurrency_and_normalizes_url() -> None:
    config = resolve_polar_slime_config(_args())

    assert config.rollout_server_url == "http://rollout:8080"
    assert config.max_concurrency == 6
    assert config.max_session_concurrency == 24
    assert config.max_off_policy_steps == 7
    assert config.request_timeout == 60.0
    assert config.min_complete_accept_fraction == 0.0


def test_resolve_polar_slime_config_requires_agent_template() -> None:
    with pytest.raises(ValueError, match="agent spec"):
        resolve_polar_slime_config(_args(polar_task_template={}))


def test_resolve_polar_slime_config_validates_harness_pool() -> None:
    config = resolve_polar_slime_config(
        _args(
            polar_harness_pool=[
                {"harness": "pi", "model_name": "openai/model"},
                {"harness": "codex", "model_name": "model"},
            ],
            polar_harness_seed=17,
        )
    )

    assert [spec["harness"] for spec in config.harness_pool] == ["pi", "codex"]
    assert config.harness_seed == 17


def test_resolve_polar_slime_config_rejects_duplicate_harnesses() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        resolve_polar_slime_config(
            _args(
                polar_harness_pool=[
                    {"harness": "pi"},
                    {"harness": "pi"},
                ]
            )
        )


def test_resolve_polar_slime_config_accepts_complete_fraction_threshold() -> None:
    config = resolve_polar_slime_config(
        _args(polar_min_complete_accept_fraction=0.8)
    )

    assert config.min_complete_accept_fraction == 0.8


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_resolve_polar_slime_config_rejects_invalid_complete_fraction(value) -> None:
    with pytest.raises(ValueError, match="polar_min_complete_accept_fraction"):
        resolve_polar_slime_config(_args(polar_min_complete_accept_fraction=value))


def test_render_task_payload_resolves_args_and_sample_placeholders() -> None:
    args = _args()
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={"image": "runtime:latest", "instance_id": "abc123"},
        group_index=9,
    )

    payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="Fix the bug",
        rollout_id=2,
        task_position=0,
        num_rollouts=4,
    )

    assert payload["task_id"] == "task-2-9"
    assert payload["instruction"] == "Fix the bug"
    assert payload["num_samples"] == 4
    assert "early_stop_min_usable_sessions" not in payload
    assert payload["agent"]["model_name"] == "openai/gpt-test"
    assert payload["runtime"]["image"] == "runtime:latest"
    assert payload["metadata"]["instance"] == "abc123"


def test_render_task_payload_samples_one_harness_per_prompt_deterministically() -> None:
    args = _args(
        polar_harness_pool=[
            {"harness": "pi", "model_name": "openai/model"},
            {"harness": "codex", "model_name": "model"},
            {"harness": "claude_code", "model_name": "model"},
            {"harness": "qwen_code", "model_name": "model"},
        ],
        polar_harness_seed=0,
    )
    config = resolve_polar_slime_config(args)

    def render(group_index: int) -> dict:
        return render_task_payload(
            args=args,
            config=config,
            sample=SimpleNamespace(
                prompt="prompt",
                metadata={"image": "runtime:latest", "instance_id": "abc123"},
                group_index=group_index,
            ),
            instruction="Fix the bug",
            rollout_id=2,
            task_position=0,
            num_rollouts=4,
        )

    first = render(9)
    assert render(9)["agent"] == first["agent"]
    assert first["metadata"]["harness"] == first["agent"]["harness"]
    assert {
        render(group_index)["agent"]["harness"]
        for group_index in range(8)
    } == {"pi", "codex", "claude_code", "qwen_code"}


def test_render_task_payload_without_pool_preserves_agent_and_metadata() -> None:
    args = _args()
    config = resolve_polar_slime_config(args)
    payload = render_task_payload(
        args=args,
        config=config,
        sample=SimpleNamespace(
            prompt="prompt",
            metadata={"image": "runtime:latest", "instance_id": "abc123"},
            group_index=9,
        ),
        instruction="Fix the bug",
        rollout_id=2,
        task_position=0,
        num_rollouts=4,
    )

    assert payload["agent"] == {
        "harness": "codex",
        "model_name": "openai/gpt-test",
    }
    assert payload["metadata"] == {"instance": "abc123"}


def test_render_task_payload_sets_early_stop_threshold_with_grace() -> None:
    args = _args(polar_min_complete_accept_fraction=0.5)
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={"image": "runtime:latest", "instance_id": "abc123"},
        group_index=0,
    )

    payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="Fix the bug",
        rollout_id=0,
        task_position=0,
        num_rollouts=32,
    )

    assert payload["early_stop_min_usable_sessions"] == 18


def test_render_instruction_uses_optional_template() -> None:
    args = _args()
    config = resolve_polar_slime_config(args)

    rendered = render_instruction(
        args=args,
        config=config,
        sample=SimpleNamespace(metadata={}),
        prompt_text="Fix the bug",
        rollout_id=1,
        task_position=0,
        num_rollouts=1,
    )

    assert rendered == "Instruction: Fix the bug"


def test_resolve_sglang_router_base_url_requires_both_ip_and_port() -> None:
    assert resolve_sglang_router_base_url(_args()) == "http://127.0.0.1:30000"
    assert resolve_sglang_router_base_url(_args(sglang_router_port=None)) is None


def test_render_topology_template_emits_inference_block(tmp_path) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        """
rollout: {host: 127.0.0.1, port: 8080, public_url: http://127.0.0.1:8080}
gateway:
  nodes:
    - id: n1
      host: 127.0.0.1
      port: 8100
      public_url: http://127.0.0.1:8100
      model_served: Qwen/Qwen3.5-4B
      inference: {engine: sglang, base_url: http://127.0.0.1:8000}
""".strip()
    )
    rendered = render_topology_template(str(topology_path), _args())
    node = rendered["gateway"]["nodes"][0]
    assert node["inference"] == {"engine": "sglang", "base_url": "http://127.0.0.1:30000"}
    assert "sglang" not in node



def test_hapo_sampler_updates_toward_the_harder_harness() -> None:
    pool = (
        {"harness": "pi", "model_name": "openai/model"},
        {"harness": "codex", "model_name": "model"},
    )
    sampler = AdaptiveHarnessSampler(
        pool,
        seed=17,
        epsilon=0.2,
        learning_rate=0.1,
        correct_threshold=0.5,
    )

    assert sampler.learned_probabilities() == pytest.approx([0.5, 0.5])
    metrics = sampler.update([("pi", True)] * 4 + [("codex", False)] * 4)

    learned = sampler.learned_probabilities()
    sampled = sampler.sampling_probabilities()
    assert learned[1] > learned[0]
    assert sampled[1] > sampled[0]
    assert min(sampled) >= 0.1 - 1e-12
    assert sum(sampled) == pytest.approx(1.0)
    assert metrics["polar/hapo/pi/accuracy"] == 1.0
    assert metrics["polar/hapo/codex/accuracy"] == 0.0
    assert metrics["polar/hapo/codex/relative_difficulty"] > 0.0


def test_hapo_relative_difficulty_uses_unweighted_harness_mean() -> None:
    pool = (
        {"harness": "pi", "model_name": "openai/model"},
        {"harness": "codex", "model_name": "model"},
        {"harness": "claude_code", "model_name": "model"},
    )
    sampler = AdaptiveHarnessSampler(
        pool,
        seed=17,
        epsilon=0.2,
        learning_rate=0.1,
        correct_threshold=0.5,
    )
    sampler.logits = [4.0, 0.0, -4.0]

    metrics = sampler.update(
        [("pi", True), ("codex", False), ("claude_code", False)]
    )

    assert metrics["polar/hapo/expected_difficulty"] == pytest.approx(2.0 / 3.0)
    assert metrics["polar/hapo/pi/relative_difficulty"] == pytest.approx(-2.0 / 3.0)
    assert metrics["polar/hapo/codex/relative_difficulty"] == pytest.approx(1.0 / 3.0)


def test_hapo_sampler_state_round_trips_and_sampling_is_reproducible() -> None:
    pool = (
        {"harness": "pi", "model_name": "openai/model"},
        {"harness": "codex", "model_name": "model"},
        {"harness": "claude_code", "model_name": "model"},
    )
    sampler = AdaptiveHarnessSampler(
        pool,
        seed=87,
        epsilon=0.3,
        learning_rate=0.1,
        correct_threshold=0.5,
    )
    sampler.update([("pi", True), ("codex", False), ("claude_code", False)])

    restored = AdaptiveHarnessSampler(
        pool,
        seed=87,
        epsilon=0.3,
        learning_rate=0.1,
        correct_threshold=0.5,
        state=sampler.state_dict(),
    )

    assert restored.state_dict() == sampler.state_dict()
    assert [restored.sample_agent(i) for i in range(20)] == [
        sampler.sample_agent(i) for i in range(20)
    ]


def test_resolve_polar_slime_config_validates_hapo_parameters() -> None:
    config = resolve_polar_slime_config(
        _args(
            polar_harness_pool=[{"harness": "pi"}, {"harness": "codex"}],
            polar_harness_sampling_strategy="hapo",
            polar_hapo_epsilon=0.3,
            polar_hapo_learning_rate=0.1,
            polar_hapo_correct_threshold=0.5,
        )
    )
    assert config.harness_sampling_strategy == "hapo"
    assert config.hapo_epsilon == 0.3

    with pytest.raises(ValueError, match="requires polar_harness_pool"):
        resolve_polar_slime_config(
            _args(polar_harness_sampling_strategy="hapo")
        )
    with pytest.raises(ValueError, match="polar_hapo_epsilon"):
        resolve_polar_slime_config(
            _args(
                polar_harness_pool=[{"harness": "pi"}],
                polar_harness_sampling_strategy="hapo",
                polar_hapo_epsilon=1.0,
            )
        )
