from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from polar.gateway.dispatcher import SessionDispatcher
from polar.gateway.server import _stop_if_max_steps_reached
from polar.gateway.session import SessionRegistry
from polar.rollout.models import TaskRequest


def test_task_request_validates_max_steps() -> None:
    agent = {"harness": "pi", "model_name": "openai/test-model"}

    assert TaskRequest(
        task_id="task-1",
        instruction="fix it",
        agent=agent,
        max_steps=160,
    ).max_steps == 160
    with pytest.raises(ValidationError):
        TaskRequest(
            task_id="task-1",
            instruction="fix it",
            agent=agent,
            max_steps=0,
        )


def test_dispatcher_stops_only_active_exec_at_max_steps() -> None:
    async def run() -> None:
        dispatcher = SessionDispatcher(
            max_init_workers=1,
            max_run_workers=1,
            max_postrun_workers=1,
        )
        runtime = SimpleNamespace(cancel_active_exec=AsyncMock())
        managed = SimpleNamespace(runtime=runtime, max_steps_reached=None)
        dispatcher._sessions["session-1"] = managed

        assert await dispatcher.stop_for_max_steps("session-1", 160)
        assert managed.max_steps_reached == 160
        runtime.cancel_active_exec.assert_awaited_once_with()
        assert not (await dispatcher.stop_for_max_steps("session-1", 160))
        runtime.cancel_active_exec.assert_awaited_once_with()

    asyncio.run(run())


def test_gateway_stops_after_the_configured_completion_count() -> None:
    async def run() -> None:
        node_manager = SimpleNamespace(stop_for_max_steps=AsyncMock(return_value=True))
        state = SimpleNamespace(
            storage=SimpleNamespace(
                get_session_metadata=lambda session_id: {"completion_count": 160}
            ),
            node_manager=node_manager,
        )
        session_info = SimpleNamespace(max_steps=160)

        await _stop_if_max_steps_reached(state, "session-1", session_info)

        node_manager.stop_for_max_steps.assert_awaited_once_with("session-1", 160)

    asyncio.run(run())


def test_session_registry_keeps_max_steps_out_of_user_metadata() -> None:
    registry = SessionRegistry()

    info = registry.register(
        "session-1",
        task_id="task-1",
        metadata={"group_id": 7},
        max_steps=160,
    )

    assert info.max_steps == 160
    assert info.metadata == {"group_id": 7}
