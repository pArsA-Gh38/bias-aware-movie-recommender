"""Run the real local frontend/backend with persistent private demo state."""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def secret(directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.is_symlink():
        raise ValueError("local state must not be a symlink")
    path = directory / "auth-secret"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_hex(32))
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise ValueError("unsafe local authentication secret file")
    value = path.read_text().strip()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid local authentication secret")
    return value


def run(command: list[str], directory: Path, env: dict) -> None:
    subprocess.run(command, cwd=directory, env=env, check=True)


def ready(url: str, process: subprocess.Popen) -> None:
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError("service exited; inspect ignored local logs")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise RuntimeError("service readiness timed out; inspect ignored local logs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("synthetic", "movielens"), default="synthetic"
    )
    args = parser.parse_args()
    for tool in ("uv", "node", "npm"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is required; see docs/SETUP.md")
    backend_port = int(os.environ.get("BACKEND_PORT", "8000"))
    frontend_port = int(os.environ.get("FRONTEND_PORT", "3000"))
    if backend_port == frontend_port:
        raise ValueError("frontend and backend ports must differ")
    for port in (backend_port, frontend_port):
        if not 1 <= port <= 65535:
            raise ValueError("ports must be between 1 and 65535")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    state = ROOT / ".local" / args.dataset
    env = os.environ.copy()
    env.update(
        {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    env["MOVIE_RECOMMENDER_AUTH_SECRET_KEY"] = env.get(
        "MOVIE_RECOMMENDER_AUTH_SECRET_KEY"
    ) or secret(state)
    if len(env["MOVIE_RECOMMENDER_AUTH_SECRET_KEY"]) < 32:
        raise ValueError("authentication secret requires at least 32 characters")
    env.setdefault(
        "MOVIE_RECOMMENDER_DATABASE_URL",
        "sqlite+pysqlite:///" + str(state / "app.sqlite"),
    )
    env["MOVIE_RECOMMENDER_ARTIFACT_PATH"] = str(state / "artifact")
    env["MOVIE_RECOMMENDER_EXP003_PATH"] = str(ROOT)
    env["BACKEND_API_URL"] = f"http://127.0.0.1:{backend_port}"
    env["NEXT_TELEMETRY_DISABLED"] = "1"
    env["DEMO_DATASET_LABEL"] = (
        "Synthetic demo · fictional movies"
        if args.dataset == "synthetic"
        else "MovieLens · model built locally"
    )
    for project in ("ml", "backend"):
        run(["uv", "sync", "--locked"], ROOT / project, env)
    if not (ROOT / "frontend/node_modules").is_dir():
        run(["npm", "ci", "--no-fund", "--no-audit"], ROOT / "frontend", env)
    run(
        ["uv", "run", "python", "-m", "scripts.prepare_demo", "--source", args.dataset],
        ROOT / "ml",
        env,
    )
    run(["uv", "run", "python", "-m", "app.db.seed_local"], ROOT / "backend", env)
    processes = []
    logs = []
    try:
        commands = [
            (
                [
                    "uv",
                    "run",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(backend_port),
                ],
                "backend",
                f"http://127.0.0.1:{backend_port}/health/ready",
            ),
            (
                [
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--hostname",
                    "127.0.0.1",
                    "--port",
                    str(frontend_port),
                ],
                "frontend",
                f"http://127.0.0.1:{frontend_port}/login",
            ),
        ]
        for command, project, url in commands:
            log = (state / f"{project}.log").open("a")
            logs.append(log)
            process = subprocess.Popen(
                command,
                cwd=ROOT / project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            ready(url, process)
        print(
            f"Demo ready: http://127.0.0.1:{frontend_port} (register and rate five films)",
            flush=True,
        )
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        raise RuntimeError("a service exited; inspect ignored local logs")
    except KeyboardInterrupt:
        return 0
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
