"""Start/stop llama-server and the gateway as subprocesses; read macOS memory footprint."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

from .manifest import LLAMA_BIN


def llama_server_cmd(model: Path, alias: str, parallel: int, ctx_per_slot: int, port: int) -> list[str]:
    return [
        str(LLAMA_BIN / "llama-server"),
        "-m", str(model),
        "--alias", alias,
        "-np", str(parallel),
        "-c", str(ctx_per_slot * parallel),  # total KV; split evenly across slots (kv_unified=false)
        "-ngl", "99",  # all layers on the GPU (Metal)
        "--host", "127.0.0.1",
        "--port", str(port),
        "--metrics",
    ]  # fmt: skip


def _wait_healthy(url: str, proc: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited with {proc.returncode} before becoming healthy: {url}")
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"not healthy after {timeout_s}s: {url}")


@contextmanager
def _process(
    cmd: list[str], health_url: str, log: Path, env: dict[str, str] | None = None
) -> Iterator[subprocess.Popen[bytes]]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    try:
        _wait_healthy(health_url, proc, timeout_s=180)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@contextmanager
def llama_server(
    model: Path, alias: str, parallel: int, ctx_per_slot: int, port: int, log: Path
) -> Iterator[subprocess.Popen[bytes]]:
    cmd = llama_server_cmd(model, alias, parallel, ctx_per_slot, port)
    with _process(cmd, f"http://127.0.0.1:{port}/health", log) as p:
        yield p


@contextmanager
def gateway(
    upstream_port: int, port: int, max_concurrent: int, max_queue: int, log: Path
) -> Iterator[subprocess.Popen[bytes]]:
    env = {
        **os.environ,
        "LOCALSERVE_UPSTREAM": f"http://127.0.0.1:{upstream_port}",
        "LOCALSERVE_PORT": str(port),
        "LOCALSERVE_MAX_CONCURRENT": str(max_concurrent),
        "LOCALSERVE_MAX_QUEUE": str(max_queue),
    }
    cmd = [sys.executable, "-m", "localserve.gateway.app"]
    with _process(cmd, f"http://127.0.0.1:{port}/healthz", log, env) as p:
        yield p


_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}


def parse_footprint(text: str) -> dict[str, int]:
    """Parse `footprint -p PID` output -> {"phys_footprint_bytes", "phys_footprint_peak_bytes"}.

    phys_footprint is what macOS charges a process for memory, *including* Metal GPU buffers; RSS
    misses those on Apple Silicon (measured: RSS 1.5 GB vs footprint 2.6 GB for the same server).
    """
    out = {}
    for key in ("phys_footprint", "phys_footprint_peak"):
        m = re.search(rf"^\s*{key}:\s*([\d.]+)\s*([KMG]?B)\s*$", text, re.M)
        if m:
            out[f"{key}_bytes"] = int(float(m.group(1)) * _UNITS[m.group(2)])
    return out


def footprint(pid: int) -> dict[str, int]:
    try:
        r = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return parse_footprint(r.stdout)
