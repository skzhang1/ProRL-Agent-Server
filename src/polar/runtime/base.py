"""Runtime abstraction for container-backed rollout execution."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

from polar.runtime.models import ExecResult, RuntimeSpec

logger = logging.getLogger(__name__)

RUNTIME_SESSION_DIR: Final[str] = "/polar/session"
RUNTIME_ARTIFACTS_DIR: Final[str] = f"{RUNTIME_SESSION_DIR}/artifacts"
RUNTIME_LOGS_DIR: Final[str] = f"{RUNTIME_SESSION_DIR}/logs"
RUNTIME_AGENT_LOG_DIR: Final[str] = f"{RUNTIME_LOGS_DIR}/agent"
RUNTIME_EVAL_LOG_DIR: Final[str] = f"{RUNTIME_LOGS_DIR}/eval"
RUNTIME_EVAL_ARTIFACT_DIR: Final[str] = f"{RUNTIME_SESSION_DIR}/eval_artifacts"


class BaseRuntime(ABC):
    """Base class for long-lived per-session execution runtimes."""

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        self.spec = spec
        self.session_id = session_id
        self.session_dir = session_dir
        self.artifacts_dir = session_dir / "artifacts"
        self.runtime_session_dir = RUNTIME_SESSION_DIR
        self.runtime_artifacts_dir = RUNTIME_ARTIFACTS_DIR
        self.runtime_logs_dir = RUNTIME_LOGS_DIR
        self.runtime_agent_log_dir = RUNTIME_AGENT_LOG_DIR
        self._active_process: asyncio.subprocess.Process | None = None
        # Every local command gets its own POSIX session/process group. Keep
        # the group IDs until runtime teardown because a command can exit
        # successfully after starting a background process inside the
        # container. In Apptainer direct-exec mode there is no long-lived
        # instance for `apptainer instance stop` to clean those children up.
        self._process_groups: set[int] = set()
        self._destroyed = False

    @property
    @abstractmethod
    def runtime_id(self) -> str:
        """Identifier for the live runtime instance."""

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return False

    @property
    def supports_cpu_limits(self) -> bool:
        return False

    @property
    def supports_memory_limits(self) -> bool:
        return False

    @property
    def supports_storage_limits(self) -> bool:
        return False

    @abstractmethod
    async def start(self) -> None:
        """Create and start the runtime instance."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop and remove the runtime instance."""

    async def cancel(self) -> None:
        """Stop any in-flight command and tear the runtime down."""
        await self.cancel_active_exec()
        await self.stop()

    async def cancel_active_exec(self) -> None:
        """Stop the in-flight command without tearing down the runtime."""
        process = self._active_process
        if process is not None and process.returncode is None:
            await BaseRuntime._kill_process_group(process)

    @staticmethod
    async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
        """SIGKILL a command and every descendant still in its POSIX group."""
        # Kill first: scanning /proc here can delay cancellation under load.
        logger.warning("runtime killpg active pid=%s pgid=%s", process.pid, process.pid)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except (AttributeError, PermissionError):
            # Preserve the old best-effort behavior on non-POSIX platforms or
            # if a launcher unexpectedly changes process-group ownership.
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await process.wait()
        except ProcessLookupError:
            pass

    async def _kill_tracked_process_groups(self) -> None:
        """Remove background children left by completed local commands."""
        groups = self._process_groups
        self._process_groups = set()
        for pgid in groups:
            logger.warning(
                "runtime killpg tracked pgid=%s members=%s",
                pgid,
                BaseRuntime._describe_process_group(pgid),
            )
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                # Never fall back to killing an arbitrary reused numeric PID.
                pass

    @staticmethod
    def _describe_process_group(pgid: int) -> list[dict[str, object]]:
        """Best-effort evidence for every process targeted by ``killpg``."""
        members: list[dict[str, object]] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
                # comm may contain spaces and parentheses, so parse fields
                # only after its final closing parenthesis. pgrp is field 5.
                rest = stat.rsplit(")", 1)[1].split()
                process_group = int(rest[2])
                if process_group != pgid:
                    continue
                cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    errors="replace"
                )
                members.append({"pid": int(entry.name), "cmdline": cmdline[:1000]})
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                continue
        return members

    @staticmethod
    def _describe_process_tree(root_pid: int) -> list[dict[str, object]]:
        """Describe a live launcher and its descendants without scanning all /proc."""
        pending = [root_pid]
        seen: set[int] = set()
        members: list[dict[str, object]] = []
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            proc_dir = Path("/proc") / str(pid)
            try:
                children = (proc_dir / "task" / str(pid) / "children").read_text()
                pending.extend(int(child) for child in children.split())
                status = (proc_dir / "status").read_text().splitlines()
                fields = {
                    line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                    for line in status
                    if ":" in line
                }
                cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    errors="replace"
                )
                members.append(
                    {
                        "pid": pid,
                        "ppid": fields.get("PPid"),
                        "rss": fields.get("VmRSS"),
                        "vsz": fields.get("VmSize"),
                        "cmdline": cmdline[:500],
                    }
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                continue
        return sorted(members, key=lambda item: int(item["pid"]))

    @staticmethod
    async def _monitor_process_tree(root_pid: int) -> None:
        """Periodically retain resource evidence for unexpected signal exits."""
        elapsed = 0
        while True:
            await asyncio.sleep(10)
            elapsed += 10
            members = BaseRuntime._describe_process_tree(root_pid)
            logger.info(
                "runtime process sample root=%s elapsed_s=%s members=%s",
                root_pid,
                elapsed,
                members,
            )

    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        """Execute one command inside the runtime and return captured output."""

    @abstractmethod
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Copy a single file from the host into the runtime."""

    @abstractmethod
    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Copy a directory tree from the host into the runtime."""

    @abstractmethod
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Copy a single file from inside the runtime to the host."""

    @abstractmethod
    async def download_dir(self, remote_path: str, local_path: str) -> None:
        """Copy a directory tree from inside the runtime to the host."""

    def resolve_host_path(self, runtime_path: str) -> Path | None:
        """Map a runtime path back to a host path via the session bind mount."""
        normalized = Path(runtime_path)
        runtime_root = Path(RUNTIME_SESSION_DIR)
        try:
            relative = normalized.relative_to(runtime_root)
        except ValueError:
            return None
        return self.session_dir / relative

    def _copy_from_bind_mount(self, runtime_path: str, local_path: Path) -> bool:
        host_path = self.resolve_host_path(runtime_path)
        if host_path is None or not host_path.exists():
            return False
        if host_path.is_dir():
            if local_path.exists():
                shutil.rmtree(local_path)
            shutil.copytree(host_path, local_path)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(host_path, local_path)
        return True

    def _copy_to_bind_mount(self, local_path: str, runtime_path: str) -> bool:
        host_path = self.resolve_host_path(runtime_path)
        if host_path is None:
            return False
        source = Path(local_path)
        if not source.exists():
            raise FileNotFoundError(f"source path does not exist: {local_path}")
        if source.is_dir():
            if host_path.exists():
                shutil.rmtree(host_path)
            shutil.copytree(source, host_path)
        else:
            host_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, host_path)
        return True

    async def _run_local_command(
        self,
        *args: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        capture: bool = False,
    ) -> tuple[int, str | None, str | None]:
        """Run a local subprocess, optionally capturing stdout/stderr."""
        process_env = None if env is None else {**os.environ, **env}
        if capture:
            stdout_target = asyncio.subprocess.PIPE
            stderr_target = asyncio.subprocess.PIPE
        else:
            stdout_target = asyncio.subprocess.DEVNULL
            stderr_target = asyncio.subprocess.DEVNULL

        process = await asyncio.create_subprocess_exec(
            *args,
            env=process_env,
            stdout=stdout_target,
            stderr=stderr_target,
            start_new_session=True,
        )
        monitor_task = asyncio.create_task(BaseRuntime._monitor_process_tree(process.pid))
        self._process_groups.add(process.pid)
        self._active_process = process
        try:
            if timeout is None:
                stdout_bytes, stderr_bytes = await process.communicate()
            else:
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    await BaseRuntime._kill_process_group(process)
                    return -1, None, None
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            self._active_process = None
            # Avoid retaining ordinary, already-empty process groups and thus
            # avoid any chance of a numeric PGID being reused before teardown.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                self._process_groups.discard(process.pid)
            except PermissionError:
                pass

        rc = process.returncode or 0
        if rc < 0:
            logger.warning(
                "runtime command exited by signal root=%s returncode=%s final_members=%s",
                process.pid,
                rc,
                BaseRuntime._describe_process_tree(process.pid),
            )
        stdout_str = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr_str = stderr_bytes.decode(errors="replace") if stderr_bytes else None
        return rc, stdout_str, stderr_str
