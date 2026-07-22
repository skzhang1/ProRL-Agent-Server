import asyncio
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock

from polar.gateway.node import GatewayNodeManager
from polar.runtime.models import ExecInput, ExecResult


def test_sigkill_is_untrainable_and_cancels_runtime(tmp_path):
    async def run():
        manager = object.__new__(GatewayNodeManager)
        manager._remaining_budget = lambda managed: 30.0
        runtime = SimpleNamespace(
            exec=AsyncMock(
                return_value=ExecResult(
                    return_code=-signal.SIGKILL,
                    stdout=None,
                    stderr=None,
                )
            ),
            cancel=AsyncMock(),
        )
        managed = SimpleNamespace(
            cancel_requested=False,
            max_steps_reached=None,
            session_dir=tmp_path,
            request=SimpleNamespace(session_id="session-1", task_id="task-1"),
        )

        result = await manager._run_exec_inputs(
            runtime,
            [ExecInput(command="pi --print")],
            {},
            managed,
        )

        assert result.status == "failed"
        assert result.return_code == -signal.SIGKILL
        assert result.metadata["failure_kind"] == "infrastructure_sigkill"
        assert result.metadata["trainable"] is False
        runtime.cancel.assert_awaited_once_with()

    asyncio.run(run())


def test_max_steps_is_a_controlled_timeout_not_infrastructure_sigkill(tmp_path):
    async def run():
        manager = object.__new__(GatewayNodeManager)
        manager._remaining_budget = lambda managed: 30.0
        runtime = SimpleNamespace(
            exec=AsyncMock(
                return_value=ExecResult(
                    return_code=-signal.SIGKILL,
                    stdout=None,
                    stderr=None,
                )
            ),
            cancel=AsyncMock(),
        )
        managed = SimpleNamespace(
            cancel_requested=False,
            max_steps_reached=160,
            session_dir=tmp_path,
            request=SimpleNamespace(session_id="session-1", task_id="task-1"),
        )

        result = await manager._run_exec_inputs(
            runtime,
            [ExecInput(command="pi --print")],
            {},
            managed,
        )

        assert result.status == "timeout"
        assert result.error == "session reached max_steps=160"
        assert result.metadata["termination_reason"] == "max_steps"
        assert result.metadata["max_steps"] == 160
        runtime.cancel.assert_not_awaited()

    asyncio.run(run())


def test_write_exec_log_recreates_missing_directory(tmp_path):
    log_dir = tmp_path / "deleted" / "logs" / "agent"

    GatewayNodeManager._write_exec_log(log_dir, "step.00", "out", "err")

    assert (log_dir / "step.00.stdout.log").read_text() == "out"
    assert (log_dir / "step.00.stderr.log").read_text() == "err"


def test_pi_harness_keeps_instruction_inline_without_instruction_file():
    from polar.agent.models import AgentSpec
    from polar.agent.presets.pi import PiHarness

    harness = PiHarness(
        AgentSpec(harness="pi", model_name="openai/test-model")
    )
    instruction = "fix 'quoted file.py'"

    steps = harness.run_steps(instruction)

    assert len(steps) == 1
    assert "pi-instruction.txt" not in steps[0].command
    assert "fix " in steps[0].command
    assert "quoted file.py" in steps[0].command
