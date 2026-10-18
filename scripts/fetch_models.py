"""Download each source GGUF from configs/models.yaml, verify sha256, quantize to each listed level.

Usage: uv run python scripts/fetch_models.py [--model qwen3-1.7b]
Writes models/<model>-<QUANT>.gguf and models/models.lock.json (sha256 + size of every file).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from localserve.manifest import LLAMA_BIN, ROOT, sha256_file

MODELS_DIR = ROOT / "models"


def model_path(name: str, quant: str) -> Path:
    return MODELS_DIR / f"{name}-{quant}.gguf"


def fetch_source(spec: dict[str, Any]) -> Path:
    dst: Path = MODELS_DIR / spec["file"]
    if not dst.exists() or dst.stat().st_size != spec["size_bytes"]:
        print(f"downloading {spec['url']}", flush=True)
        subprocess.run(["curl", "-sSL", "-C", "-", "--retry", "5", "-o", str(dst), spec["url"]], check=True)
    digest = sha256_file(dst)
    if digest != spec["sha256"]:
        sys.exit(f"sha256 mismatch for {dst}: got {digest}, expected {spec['sha256']}")
    print(f"verified {dst.name} sha256={digest}")
    return dst


def quantize(src: Path, dst: Path, quant: str) -> None:
    if dst.exists():
        return
    tmp = dst.with_suffix(".tmp")
    subprocess.run([str(LLAMA_BIN / "llama-quantize"), str(src), str(tmp), quant], check=True)
    tmp.rename(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="only this model key")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]
    MODELS_DIR.mkdir(exist_ok=True)
    lock_path = MODELS_DIR / "models.lock.json"
    lock: dict[str, Any] = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    for name, spec in cfg.items():
        if args.model and name != args.model:
            continue
        src = fetch_source(spec)
        lock[f"{name}-F16"] = {"path": src.name, "sha256": spec["sha256"], "bytes": src.stat().st_size}
        for q in spec["quants"]:
            dst = model_path(name, q)
            quantize(src, dst, q)
            size = dst.stat().st_size
            entry = {"path": dst.name, "sha256": sha256_file(dst), "bytes": size}
            lock[f"{name}-{q}"] = entry
            print(f"{dst.name}: {size / 1e9:.3f} GB sha256={entry['sha256']}")
    lock_path.write_text(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
