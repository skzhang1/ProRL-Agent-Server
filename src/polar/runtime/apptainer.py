"""Apptainer-backed rollout runtime."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
import shlex
import shutil
from pathlib import Path

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec

logger = logging.getLogger(__name__)


class ApptainerRuntime(BaseRuntime):
    """Apptainer instance used across rollout stages."""

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        super().__init__(spec, session_id, session_dir)
        # Use a hash suffix to guarantee uniqueness even when session IDs
        # share a long prefix (e.g. "sk-polar-...-eval" vs "sk-polar-...").
        short_hash = hashlib.sha256(session_id.encode()).hexdigest()[:8]
        safe_name = session_id.replace("/", "-")[:30]
        self._instance_name = f"polar-{safe_name}-{short_hash}"
        self._runtime_marker = self._instance_name
        self._binary = self._resolve_binary()
        self._direct_exec = os.environ.get("POLAR_APPTAINER_DIRECT_EXEC", "0") == "1"

    @property
    def runtime_id(self) -> str:
        return self._instance_name

    @property
    def supports_gpus(self) -> bool:
        return True

    @property
    def can_disable_internet(self) -> bool:
        return True

    @property
    def supports_memory_limits(self) -> bool:
        # Apptainer does not expose an unprivileged cgroup limit here. Apply a
        # POSIX address-space rlimit in every container command instead; it is
        # inherited by forked, exec'd, and daemonized descendants.
        return True

    async def start(self) -> None:
        if self._destroyed:
            raise RuntimeError("apptainer runtime was already destroyed")
        # Use a host-backed overlay directory instead of --writable-tmpfs
        # (default tmpfs overlay is only 64 MB, too small for most workloads).
        self._overlay_dir = self.session_dir / "overlay"
        self._overlay_dir.mkdir(parents=True, exist_ok=True)
        if self._direct_exec:
            return
        args = [self._binary, "instance", "start",
                "--overlay", str(self._overlay_dir)]
        if self.spec.gpus > 0:
            args.append("--nv")
        network_name: str | None
        if not self.spec.allow_internet:
            network_name = "none"
        else:
            network_name = self.spec.network
        if network_name and network_name != "host":
            args.extend(["--net", "--network", network_name])
        args.extend(["--bind", f"{self.session_dir}:{self.runtime_session_dir}"])
        # Match DockerRuntime's kwargs.volumes contract. Apptainer accepts the
        # same src[:dst[:opts]] bind syntax for the read-only CLI mount used by
        # SWE-Gym.
        for volume in self.spec.kwargs.get("volumes", []):
            args.extend(["--bind", str(volume)])
        args.extend([self.spec.image, self._instance_name])
        rc, _, _ = await self._run_local_command(*args)
        if rc != 0:
            raise RuntimeError(
                f"{self._binary} instance start failed with exit code {rc}"
            )

    _STOP_TIMEOUT = 30.0

    async def stop(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if self._direct_exec:
            await self._kill_marked_processes()
            await self._kill_tracked_process_groups()
            return
        rc, _, stderr = await self._run_local_command(
            self._binary, "instance", "stop", self._instance_name,
            timeout=self._STOP_TIMEOUT, capture=True,
        )
        if rc != 0:
            logger.warning(
                "%s instance stop failed for %s (rc=%s): %s",
                self._binary, self._instance_name, rc, stderr,
            )

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        effective_env = {**self.spec.env, **(env or {})}
        # A daemon can escape its launcher's process group with setsid(). Keep
        # a unique inherited marker so teardown can still identify it safely.
        effective_env["POLAR_RUNTIME_ID"] = self._runtime_marker
        effective_workdir = cwd or self.spec.workdir or self.runtime_session_dir
        wrapped_command = command
        if effective_workdir:
            wrapped_command = f"cd {shlex.quote(effective_workdir)} && {command}"
        shell_exports = []
        for key in ("HOME", "PATH"):
            if key in effective_env:
                shell_exports.append(f"export {key}={shlex.quote(str(effective_env[key]))};")
        if shell_exports:
            wrapped_command = " ".join(shell_exports + [wrapped_command])
        if self.spec.memory_mb is not None:
            limit_kib = self.spec.memory_mb * 1024
            wrapped_command = f"ulimit -v {limit_kib}; {wrapped_command}"
        args = [self._binary, "exec"]
        if self._direct_exec:
            args.extend(["--overlay", str(self._overlay_dir)])
            if self.spec.gpus > 0:
                args.append("--nv")
            network_name = "none" if not self.spec.allow_internet else self.spec.network
            if network_name and network_name != "host":
                args.extend(["--net", "--network", network_name])
            args.extend(["--bind", f"{self.session_dir}:{self.runtime_session_dir}"])
            for volume in self.spec.kwargs.get("volumes", []):
                args.extend(["--bind", str(volume)])
            args.append(self.spec.image)
        else:
            args.append(f"instance://{self._instance_name}")
        if effective_env:
            args.append("env")
            args.extend(f"{key}={value}" for key, value in effective_env.items())
        args.extend(["bash", "-lc", wrapped_command])
        rc, stdout, stderr = await self._run_local_command(
            *args, timeout=timeout_sec, capture=True
        )
        return ExecResult(stdout=stdout, stderr=stderr, return_code=rc)

    async def _kill_marked_processes(self) -> None:
        """Kill daemonized descendants that escaped their launch process group."""
        marker = f"POLAR_RUNTIME_ID={self._runtime_marker}".encode()
        self_pid = os.getpid()
        # Repeat because a process can fork while the first /proc pass is in
        # progress. SIGKILL plus three event-loop yields closes that race for
        # ordinary runtime teardown without touching unrelated processes.
        for _ in range(3):
            found = False
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                if pid == self_pid:
                    continue
                try:
                    environ = (entry / "environ").read_bytes()
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                if marker not in environ.split(b"\0"):
                    continue
                found = True
                try:
                    cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                        errors="replace"
                    )
                    try:
                        pgid = os.getpgid(pid)
                    except ProcessLookupError:
                        pgid = None
                    logger.warning(
                        "runtime marker SIGKILL runtime=%s pid=%s pgid=%s cmdline=%s",
                        self._runtime_marker,
                        pid,
                        pgid,
                        cmdline[:1000],
                    )
                    os.kill(pid, signal.SIGKILL)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if not found:
                return
            await asyncio.sleep(0)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        if self._copy_to_bind_mount(local_path, remote_path):
            return
        parent = str(Path(remote_path).parent)
        filename = Path(local_path).name
        source_dir = str(Path(local_path).parent)
        result = await self.exec(f"mkdir -p {shlex.quote(parent)}")
        if result.return_code != 0:
            raise RuntimeError(f"failed to create directory {parent} in runtime")
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"tar -cf - -C {shlex.quote(source_dir)} {shlex.quote(filename)} | "
            f"{self._binary} exec instance://{self._instance_name} "
            f"tar -xf - -C {shlex.quote(parent)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(f"apptainer upload_file failed with exit code {rc}")

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        if self._copy_to_bind_mount(local_path, remote_path):
            return
        result = await self.exec(f"mkdir -p {shlex.quote(remote_path)}")
        if result.return_code != 0:
            raise RuntimeError(
                f"failed to create directory {remote_path} in runtime"
            )
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"tar -cf - -C {shlex.quote(local_path)} . | "
            f"{self._binary} exec instance://{self._instance_name} "
            f"tar -xf - -C {shlex.quote(remote_path)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(f"apptainer upload_dir failed with exit code {rc}")

    async def download_file(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        parent = str(Path(remote_path).parent)
        filename = Path(remote_path).name
        local_dir = str(Path(local_path).parent)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"{self._binary} exec instance://{self._instance_name} "
            f"tar -cf - -C {shlex.quote(parent)} {shlex.quote(filename)} | "
            f"tar -xf - -C {shlex.quote(local_dir)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(
                f"apptainer download_file failed with exit code {rc}"
            )

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        Path(local_path).mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"{self._binary} exec instance://{self._instance_name} "
            f"tar -cf - -C {shlex.quote(remote_path)} . | "
            f"tar -xf - -C {shlex.quote(local_path)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(
                f"apptainer download_dir failed with exit code {rc}"
            )

    @staticmethod
    def _resolve_binary() -> str:
        override = os.environ.get("POLAR_APPTAINER_BIN")
        if override:
            return override
        for candidate in ("/usr/bin/apptainer", "/bin/apptainer"):
            if Path(candidate).is_file():
                return candidate
        resolved = shutil.which("apptainer")
        if resolved:
            return resolved
        return "apptainer"
