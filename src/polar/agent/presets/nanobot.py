"""nanobot harness — https://github.com/HKUDS/nanobot"""

from __future__ import annotations

import json
import shlex

from polar.agent.base import BaseHarness
from polar.agent.models import AgentRunResult
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR, RUNTIME_SESSION_DIR
from polar.runtime.models import ExecInput


class NanobotHarness(BaseHarness):
    """Run nanobot against the Polar OpenAI-compatible gateway."""

    _CONFIG_DIR = "$HOME/.nanobot"
    _CONFIG_PATH = f"{_CONFIG_DIR}/config.json"
    _WORKSPACE = f"{_CONFIG_DIR}/workspace"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._WORKSPACE}")
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._WORKSPACE}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* "
                f"{self._WORKSPACE}/skills/ 2>/dev/null || true"
            )

    async def postprocess(
        self, runtime: BaseRuntime, result: AgentRunResult
    ) -> None:
        if result.return_code == 0:
            return
        log = await runtime.exec(
            f"tail -c 8192 {RUNTIME_AGENT_LOG_DIR}/nanobot.txt 2>/dev/null || true"
        )
        detail = (log.stdout or log.stderr or "").strip()
        if detail:
            result.error = f"{result.error}\nnanobot log:\n{detail}"

    def run_steps(self, instruction: str) -> list[ExecInput]:
        if not self.model_name:
            raise ValueError("nanobot harness requires model_name")

        defaults: dict = {
            "workspace": self._WORKSPACE,
            "provider": "polar",
            "model": self.model_name,
            "maxTokens": int(self.settings.get("max_tokens", 8192)),
            "contextWindowTokens": int(
                self.settings.get("context_window", 200_000)
            ),
            "maxToolIterations": int(
                self.settings.get("max_tool_iterations", 200)
            ),
            "temperature": float(self.settings.get("temperature", 0.1)),
        }
        reasoning_effort = self.settings.get("reasoning_effort")
        if reasoning_effort is not None:
            defaults["reasoningEffort"] = str(reasoning_effort)

        config: dict = {
            "agents": {"defaults": defaults},
            "providers": {
                "polar": {
                    "apiKey": "${OPENAI_API_KEY}",
                    "apiBase": "${OPENAI_BASE_URL}",
                    "extraHeaders": {"X-Session-ID": "${OPENAI_API_KEY}"},
                }
            },
            "tools": {
                "exec": {
                    "timeout": int(self.settings.get("exec_timeout", 600))
                },
                "web": {"enable": bool(self.settings.get("web_enabled", False))},
            },
            "channels": {
                "sendProgress": False,
                "sendToolHints": False,
                "showReasoning": False,
            },
        }

        if self.mcp_servers:
            servers: dict[str, dict] = {}
            for server in self.mcp_servers:
                if server.transport == "stdio":
                    entry: dict = {"command": server.command}
                    if server.args:
                        entry["args"] = server.args
                else:
                    entry = {
                        "url": server.url,
                        "type": (
                            "streamableHttp"
                            if server.transport == "streamable-http"
                            else server.transport
                        ),
                    }
                servers[server.name] = entry
            config["tools"]["mcpServers"] = servers

        config_json = json.dumps(config)
        task = (
            f"Work on the repository at {RUNTIME_SESSION_DIR}/workspace. "
            "Make all requested code changes there. Inspect relevant files before "
            "editing and create the smallest correct patch early. Do not install or "
            "upgrade dependencies; if tests cannot run in the existing environment, "
            "leave the patch in place and report that limitation. Before finishing, "
            "check that git diff contains the intended changes.\n\n"
            f"{instruction}"
        )
        return [
            ExecInput(
                command=(
                    f"mkdir -p {self._CONFIG_DIR} {self._WORKSPACE} "
                    f"{RUNTIME_SESSION_DIR}/tmp && "
                    f"printf '%s' {shlex.quote(config_json)} "
                    f"> {self._CONFIG_PATH} && "
                    f"nanobot agent --no-markdown "
                    f"--config {self._CONFIG_PATH} "
                    f"--workspace {self._WORKSPACE} "
                    f"--session polar --message {shlex.quote(task)} "
                    f"> {RUNTIME_AGENT_LOG_DIR}/nanobot.txt 2>&1"
                ),
                env={
                    **self.env,
                    "TMPDIR": f"{RUNTIME_SESSION_DIR}/tmp",
                },
            )
        ]
