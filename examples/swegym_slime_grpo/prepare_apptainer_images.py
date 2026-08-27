#!/usr/bin/env python3
"""Prepare local Apptainer SIF images for the full SWE-Gym train dataset."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from sample_tasks import fetch_all_instances, registry_image_for_instance_id


DEFAULT_IMAGE_DIR = PROJECT_ROOT / "tmp" / "swegym_apptainer_images"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "tmp" / "apptainer_cache"
DEFAULT_TMP_DIR = PROJECT_ROOT / "tmp" / "apptainer_tmp"
DEFAULT_AGENT_CLI_DIR = PROJECT_ROOT / "tmp" / "swegym_agent_cli" / "opt_node"
NODE_VERSION = "22.11.0"
OPENCLAW_NODE_VERSION = "24.15.0"
DEFAULT_HARNESSES = ("codex", "claude_code", "qwen_code", "opencode", "pi")
NPM_PACKAGE_BY_HARNESS = {
    "codex": "@openai/codex@latest",
    "claude_code": "@anthropic-ai/claude-code@latest",
    "qwen_code": "@qwen-code/qwen-code@latest",
    "opencode": "opencode-ai@latest",
    "pi": "@mariozechner/pi-coding-agent@latest",
    "openclaw": "openclaw@2026.5.27",
}
PYTHON_PACKAGE_BY_HARNESS = {
    "mini_swe_agent": "mini-swe-agent==2.4.2",
    "nanobot": "nanobot-ai==0.2.2",
}
BIN_BY_HARNESS = {
    "codex": "codex",
    "claude_code": "claude",
    "qwen_code": "qwen",
    "opencode": "opencode",
    "pi": "pi",
    "openclaw": "openclaw",
    "mini_swe_agent": "mini-swe-agent",
    "nanobot": "nanobot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Only prepare the SIF image for this instance_id.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory for prepared .sif images.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Apptainer cache directory.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=DEFAULT_TMP_DIR,
        help="Apptainer temporary build directory.",
    )
    parser.add_argument(
        "--agent-cli-dir",
        type=Path,
        default=DEFAULT_AGENT_CLI_DIR,
        help="Host directory mounted as /opt/node in task containers.",
    )
    parser.add_argument(
        "--skip-cli",
        action="store_true",
        help="Only prepare SIF images; do not prepare the shared CLI directory.",
    )
    parser.add_argument(
        "--force-cli",
        action="store_true",
        help="Rebuild and re-extract the shared Node/agent CLI directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-pull SIF images even when the output file exists.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Number of concurrent apptainer pull jobs.",
    )
    parser.add_argument(
        "--refresh-dataset-cache",
        action="store_true",
        help="Refresh the cached dataset rows before preparing images.",
    )
    return parser.parse_args()


def select_instances(args: argparse.Namespace) -> list[dict[str, object]]:
    instances = fetch_all_instances(refresh=args.refresh_dataset_cache)
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [instance for instance in instances if str(instance.get("instance_id")) in wanted]
        missing = sorted(wanted - {str(instance.get("instance_id")) for instance in selected})
        if missing:
            raise SystemExit(f"Unknown instance_id(s): {', '.join(missing)}")
        return selected
    return instances


def sif_path_for_instance(instance_id: str, image_dir: Path) -> Path:
    if "/" in instance_id or "\0" in instance_id:
        raise ValueError(f"instance_id is not safe for a filename: {instance_id!r}")
    return image_dir / f"{instance_id}.sif"


def apptainer_registry_uri(image_ref: str) -> str:
    if image_ref.startswith(("docker://", "oras://", "library://", "shub://")):
        return image_ref
    return f"docker://{image_ref}"


def image_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def ensure_sif(
    instance: dict[str, object],
    *,
    image_dir: Path,
    force: bool,
    env: dict[str, str],
) -> tuple[str, str]:
    instance_id = str(instance["instance_id"])
    target = sif_path_for_instance(instance_id, image_dir)
    if image_ready(target) and not force:
        return ("skipped", str(target))

    image_ref = registry_image_for_instance_id(instance_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    if tmp_path.exists():
        tmp_path.unlink()

    command = [
        _apptainer_binary(),
        "pull",
        "--force",
        str(tmp_path),
        apptainer_registry_uri(image_ref),
    ]
    print("+", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, env=env)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return ("pulled", str(target))


def prepare_sifs(
    instances: list[dict[str, object]],
    *,
    image_dir: Path,
    cache_dir: Path,
    tmp_dir: Path,
    force: bool,
    jobs: int,
) -> None:
    env = {
        **os.environ,
        "APPTAINER_CACHEDIR": str(cache_dir.resolve()),
        "APPTAINER_TMPDIR": str(tmp_dir.resolve()),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    if jobs <= 1:
        for instance in instances:
            status, path = ensure_sif(
                instance, image_dir=image_dir, force=force, env=env
            )
            print(f"{status}: {path}", flush=True)
        return

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                ensure_sif, instance, image_dir=image_dir, force=force, env=env
            )
            for instance in instances
        ]
        for future in as_completed(futures):
            status, path = future.result()
            print(f"{status}: {path}", flush=True)


def _apptainer_binary() -> str:
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


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, check=True, env=env)


def _agent_cli_missing_bins(agent_cli_dir: Path, harnesses: tuple[str, ...]) -> list[str]:
    required = ("node", "npm", "npx", *(BIN_BY_HARNESS[name] for name in harnesses))
    return [
        name for name in required
        if not (agent_cli_dir / "bin" / name).is_file()
    ]


def ensure_agent_cli_dir(
    agent_cli_dir: Path,
    *,
    force: bool,
    harnesses: tuple[str, ...] = DEFAULT_HARNESSES,
) -> None:
    unknown = sorted(set(harnesses) - set(BIN_BY_HARNESS))
    if unknown:
        raise ValueError(f"Unsupported harness CLI bundle entries: {unknown}")
    if len(harnesses) != len(set(harnesses)):
        raise ValueError(f"Duplicate harness CLI bundle entries: {harnesses}")
    missing_bins = _agent_cli_missing_bins(agent_cli_dir, harnesses)
    if not missing_bins and not force:
        print(f"Shared agent CLI directory already exists: {agent_cli_dir}")
        return

    agent_cli_dir = agent_cli_dir.resolve()
    agent_cli_dir.parent.mkdir(parents=True, exist_ok=True)
    if agent_cli_dir.exists():
        shutil.rmtree(agent_cli_dir)
    agent_cli_dir.mkdir(parents=True)

    node_version = OPENCLAW_NODE_VERSION if "openclaw" in harnesses else NODE_VERSION
    node_dist_url = (
        f"https://nodejs.org/dist/v{node_version}/"
        f"node-v{node_version}-linux-x64.tar.xz"
    )
    run_command([
        "bash",
        "-c",
        (
            f"curl -fsSL {shlex.quote(node_dist_url)} | "
            f"tar -xJ --strip-components=1 -C {shlex.quote(str(agent_cli_dir))}"
        ),
    ])

    npm_bin = agent_cli_dir / "bin" / "npm"
    env = {
        **os.environ,
        "PATH": f"{agent_cli_dir / 'bin'}:{os.environ.get('PATH', '')}",
    }
    npm_packages = [
        NPM_PACKAGE_BY_HARNESS[name]
        for name in harnesses
        if name in NPM_PACKAGE_BY_HARNESS
    ]
    if npm_packages:
        run_command(
            [
                str(npm_bin),
                "install",
                "-g",
                "--no-audit",
                "--no-fund",
                f"--prefix={agent_cli_dir}",
                *npm_packages,
            ],
            env=env,
        )

    python_packages = [
        PYTHON_PACKAGE_BY_HARNESS[name]
        for name in harnesses
        if name in PYTHON_PACKAGE_BY_HARNESS
    ]
    if python_packages:
        uv_bin = agent_cli_dir / "bin" / "uv"
        uv_install_env = {
            **env,
            "UV_INSTALL_DIR": str(agent_cli_dir / "bin"),
            "UV_NO_MODIFY_PATH": "1",
        }
        run_command(
            [
                "bash",
                "-c",
                "curl -LsSf https://astral.sh/uv/0.8.13/install.sh | sh",
            ],
            env=uv_install_env,
        )
        python_install_dir = agent_cli_dir / "python"
        uv_env = {
            **env,
            "UV_PYTHON_INSTALL_DIR": str(python_install_dir),
        }
        run_command([str(uv_bin), "python", "install", "3.12.11"], env=uv_env)
        python_path = Path(
            subprocess.check_output(
                [str(uv_bin), "python", "find", "3.12.11", "--python-preference", "only-managed"],
                env=uv_env,
                text=True,
            ).strip()
        )
        venv_dir = agent_cli_dir / "python-env"
        run_command(
            [str(uv_bin), "venv", "--python", str(python_path), str(venv_dir)],
            env=uv_env,
        )
        python_path = venv_dir / "bin" / "python"
        run_command(
            [str(uv_bin), "pip", "install", "--python", str(python_path), *python_packages],
            env=uv_env,
        )

        # uv records absolute build-host paths in both the venv interpreter
        # symlink and console-script shebangs.  The bundle is mounted at
        # /opt/node in task containers, so make the interpreter relocatable
        # and invoke console scripts explicitly through it below.
        managed_python_relative = os.path.relpath(python_path.resolve(), python_path.parent)
        python_path.unlink()
        python_path.symlink_to(managed_python_relative)
        pyvenv_cfg = venv_dir / "pyvenv.cfg"
        pyvenv_cfg.write_text(
            pyvenv_cfg.read_text(encoding="utf-8").replace(
                str(agent_cli_dir), "/opt/node"
            ),
            encoding="utf-8",
        )
        for harness in harnesses:
            if harness not in PYTHON_PACKAGE_BY_HARNESS:
                continue
            binary = BIN_BY_HARNESS[harness]
            installed_binary = python_path.parent / binary
            if not installed_binary.is_file():
                raise RuntimeError(f"Python tool install did not create {installed_binary}")
            relative_binary = installed_binary.relative_to(agent_cli_dir)
            relative_python = python_path.relative_to(agent_cli_dir)
            wrapper = agent_cli_dir / "bin" / binary
            wrapper.write_text(
                "#!/usr/bin/env bash\n"
                f'exec "/opt/node/{relative_python}" "/opt/node/{relative_binary}" "$@"\n',
                encoding="utf-8",
            )
            wrapper.chmod(0o755)

    missing_bins = _agent_cli_missing_bins(agent_cli_dir, harnesses)
    if missing_bins:
        raise RuntimeError(
            "agent CLI setup did not create expected executable(s): "
            + ", ".join(missing_bins)
        )
    print(f"Prepared shared agent CLI directory: {agent_cli_dir}")


def main() -> int:
    args = parse_args()
    if shutil.which(_apptainer_binary()) is None and not Path(_apptainer_binary()).is_file():
        raise SystemExit("apptainer not found")

    if not args.skip_cli:
        ensure_agent_cli_dir(args.agent_cli_dir, force=args.force_cli)

    instances = select_instances(args)
    if not instances:
        raise SystemExit("No instances selected.")

    print(
        f"Preparing {len(instances)} Apptainer SIF image(s) with {max(args.jobs, 1)} pull job(s).",
        flush=True,
    )
    print(f"Image dir: {args.image_dir.resolve()}", flush=True)
    print(f"Cache dir: {args.cache_dir.resolve()}", flush=True)
    print(f"Tmp dir: {args.tmp_dir.resolve()}", flush=True)
    prepare_sifs(
        instances,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        tmp_dir=args.tmp_dir,
        force=args.force,
        jobs=max(args.jobs, 1),
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
