"""Run provenance: hardware, OS, git commit, package versions, file hashes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[2]
LLAMA_BIN = ROOT / ".vendor/llama.cpp/build/bin"
_HASH_CACHE = ROOT / ".cache/sha256.json"
_PACKAGES = ("fastapi", "uvicorn", "httpx", "numpy", "psutil", "prometheus-client", "matplotlib", "pyyaml")


def sha256_file(path: Path) -> str:
    """sha256 of a file, cached by (path, size, mtime) so multi-GB models are hashed once."""
    st = path.stat()
    key = f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    cache: dict[str, str] = json.loads(_HASH_CACHE.read_text()) if _HASH_CACHE.exists() else {}
    if key not in cache:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while block := f.read(1 << 22):
                h.update(block)
        cache[key] = h.hexdigest()
        _HASH_CACHE.parent.mkdir(exist_ok=True)
        _HASH_CACHE.write_text(json.dumps(cache, indent=1))
    return cache[key]


def _run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hardware() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "chip": _run("sysctl", "-n", "machdep.cpu.brand_string"),
        "model": _run("sysctl", "-n", "hw.model"),
        "mem_bytes": int(_run("sysctl", "-n", "hw.memsize") or 0),
        "cpu_perf_cores": _run("sysctl", "-n", "hw.perflevel0.physicalcpu"),
        "cpu_eff_cores": _run("sysctl", "-n", "hw.perflevel1.physicalcpu"),
        "macos": platform.mac_ver()[0],
        "mem_available_bytes_at_start": vm.available,
        "power_source": _run("pmset", "-g", "batt").splitlines()[0] if platform.system() == "Darwin" else "",
    }


def git_state() -> dict[str, Any]:
    return {
        "commit": _run("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "dirty": _run("git", "-C", str(ROOT), "status", "--porcelain") not in ("", "unknown"),
    }


def llama_cpp_version() -> str:
    return _run(str(LLAMA_BIN / "llama-server"), "--version").replace("\n", " | ")


def base_manifest() -> dict[str, Any]:
    return {
        "git": git_state(),
        "hardware": hardware(),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in _PACKAGES},
        "llama_cpp": llama_cpp_version(),
    }
