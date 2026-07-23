import asyncio
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock

from polar.gateway.node import GatewayNodeManager
from polar.runtime.apptainer import ApptainerRuntime
from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecInput, ExecResult, RuntimeSpec


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


def test_session_timeout_cancels_entire_runtime(tmp_path):
    async def run():
        manager = object.__new__(GatewayNodeManager)
        manager._remaining_budget = lambda managed: 30.0
        runtime = SimpleNamespace(
            exec=AsyncMock(
                return_value=ExecResult(
                    return_code=-1,
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

        assert result.status == "timeout"
        runtime.cancel.assert_awaited_once_with()

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
    assert "X-Session-ID" in steps[0].command
    assert "OPENAI_API_KEY" in steps[0].command
    assert steps[0].env["TMPDIR"] == "/polar/session/tmp"


def test_runtime_stop_uses_sigterm_when_process_exits(monkeypatch):
    signals = []
    process = SimpleNamespace(pid=1234, returncode=None)

    def killpg(pid, sig):
        assert pid == process.pid
        signals.append(sig)
        process.returncode = -sig

    async def wait():
        return process.returncode

    process.wait = wait
    monkeypatch.setattr("polar.runtime.base.os.killpg", killpg)
    asyncio.run(BaseRuntime._kill_process_group(process))
    assert signals == [signal.SIGTERM]


def test_apptainer_isolates_pid_namespace_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("POLAR_APPTAINER_ISOLATE_PID", raising=False)
    runtime = ApptainerRuntime(
        RuntimeSpec(backend="apptainer", image="/tmp/task.sif"),
        "session-1",
        tmp_path,
    )
    runtime._run_local_command = AsyncMock(return_value=(0, None, None))

    asyncio.run(runtime.start())

    args = runtime._run_local_command.await_args.args
    assert args[:3] == (runtime._binary, "instance", "start")
    assert "--containall" in args
    assert "--pid" not in args


def test_apptainer_direct_exec_uses_supported_pid_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("POLAR_APPTAINER_DIRECT_EXEC", "1")
    monkeypatch.setenv("POLAR_APPTAINER_ISOLATE_PID", "1")
    runtime = ApptainerRuntime(
        RuntimeSpec(backend="apptainer", image="/tmp/task.sif"),
        "session-1",
        tmp_path,
    )
    runtime._run_local_command = AsyncMock(return_value=(0, None, None))

    asyncio.run(runtime.start())
    result = asyncio.run(runtime.exec("true"))

    args = runtime._run_local_command.await_args.args
    assert result.return_code == 0
    assert args[:2] == (runtime._binary, "exec")
    assert "--containall" in args

def test_apptainer_instance_prefers_inner_status_over_wrapper_sigkill(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("POLAR_APPTAINER_DIRECT_EXEC", raising=False)
    runtime = ApptainerRuntime(
        RuntimeSpec(backend="apptainer", image="/tmp/task.sif"),
        "session-1",
        tmp_path,
    )

    async def run_local_command(*args, **kwargs):
        (tmp_path / ".polar-exec-1.status").write_text("0\n")
        return -signal.SIGKILL, None, None

    runtime._run_local_command = AsyncMock(side_effect=run_local_command)
    result = asyncio.run(runtime.exec("true"))

    assert result.return_code == 0
    assert not (tmp_path / ".polar-exec-1.status").exists()
    command = runtime._run_local_command.await_args.args[-1]
    assert "setsid --wait bash -lc" in command
    assert "trap _polar_write_status EXIT" in command
    assert ".polar-exec-1.status" in command


def test_apptainer_instance_keeps_sigkill_without_inner_status(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("POLAR_APPTAINER_DIRECT_EXEC", raising=False)
    runtime = ApptainerRuntime(
        RuntimeSpec(backend="apptainer", image="/tmp/task.sif"),
        "session-1",
        tmp_path,
    )
    runtime._run_local_command = AsyncMock(
        return_value=(-signal.SIGKILL, None, None)
    )

    result = asyncio.run(runtime.exec("false"))

    assert result.return_code == 128 + signal.SIGKILL
