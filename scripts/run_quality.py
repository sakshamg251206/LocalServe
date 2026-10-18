"""Quality + raw-speed runs (no server): llama-bench per quant, and KL divergence vs the reference.

  uv run python scripts/run_quality.py --run-id minimum

Writes results/<run-id>/raw/quality/{bench_<Q>.json, kld_<Q>_vs_<REF>.{log,json}, dataset.json}.
Resumable: existing outputs are skipped. Base-logit files (GBs) live in data/ and are deleted after use.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml

from localserve.manifest import LLAMA_BIN, ROOT, sha256_file
from localserve.quality.kld import base_cmd, kld_cmd, parse_kld_output

DATA = ROOT / "data"
WIKITEXT = {
    "name": "wikitext-2-raw-v1 (test split)",
    "url": "https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip",
    "zip_sha256": "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11",
    "member": "wikitext-2-raw/wiki.test.raw",
    "licence": "CC BY-SA 3.0 (Merity et al., 2016)",
}
# Reference logits cost ~300 KB per scored token (152k-entry vocabulary), so the full test split
# (~45 GB of logits) does not fit this machine's disk. We score the first CHUNKS chunks of CTX
# tokens; llama-perplexity scores the second half of each chunk.
CTX, CHUNKS = 512, 100


def wikitext_test() -> tuple[Path, dict[str, Any]]:
    z = DATA / "wikitext-2-raw-v1.zip"
    if not z.exists():
        DATA.mkdir(exist_ok=True)
        subprocess.run(["curl", "-sSL", "-o", str(z), WIKITEXT["url"]], check=True)
    if sha256_file(z) != WIKITEXT["zip_sha256"]:
        raise SystemExit(f"sha256 mismatch for {z}")
    out = DATA / "wiki.test.raw"
    if not out.exists():
        with zipfile.ZipFile(z) as zf:  # read one named member; never extract arbitrary paths
            out.write_bytes(zf.read(WIKITEXT["member"]))
    text = out.read_text(encoding="utf-8")
    meta = {**WIKITEXT, "test_sha256": sha256_file(out), "test_lines": text.count("\n"),
            "test_bytes": out.stat().st_size, "ctx": CTX, "chunks": CHUNKS}  # fmt: skip
    return out, meta


def run_logged(cmd: list[str], log: Path) -> str:
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    text = f"$ {' '.join(cmd)}\n# exit={r.returncode} seconds={time.time() - t0:.1f}\n{r.stdout}\n{r.stderr}"
    log.write_text(text)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {r.returncode}); see {log}")
    return r.stdout + "\n" + r.stderr


def bench(model: Path, out: Path) -> None:
    """Raw single-stream speed: prompt processing (pp512) and generation (tg128), 5 repetitions."""
    if out.exists():
        return
    cmd = [str(LLAMA_BIN / "llama-bench"), "-m", str(model), "-p", "512", "-n", "128", "-r", "5",
           "-ngl", "99", "-o", "json"]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out.write_text(json.dumps({"cmd": cmd, "results": json.loads(r.stdout)}, indent=2))
    print(f"  wrote {out.name}", flush=True)


def kld(model: Path, base_logits: Path, out_json: Path) -> None:
    if out_json.exists():
        return
    text = run_logged(kld_cmd(model, base_logits, CTX, CHUNKS), out_json.with_suffix(".log"))
    stats = parse_kld_output(text)
    if "kld_mean" not in stats:
        raise RuntimeError(f"no KLD statistics parsed; see {out_json.with_suffix('.log')}")
    out_json.write_text(json.dumps(stats, indent=2))
    print(f"  wrote {out_json.name}: mean KLD {stats['kld_mean']}", flush=True)


def kld_against(ref_name: str, ref_model: Path, others: dict[str, Path], text: Path, qdir: Path) -> None:
    todo = {q: m for q, m in others.items() if not (qdir / f"kld_{q}_vs_{ref_name}.json").exists()}
    if not todo:
        return
    base_logits = DATA / f"base_logits_{ref_name}.kld"
    print(f"[quality] reference logits from {ref_name}", flush=True)
    run_logged(base_cmd(ref_model, text, base_logits, CTX, CHUNKS), qdir / f"base_{ref_name}.log")
    try:
        for q, m in todo.items():
            print(f"[quality] KLD {q} vs {ref_name}", flush=True)
            kld(m, base_logits, qdir / f"kld_{q}_vs_{ref_name}.json")
    finally:
        base_logits.unlink(missing_ok=True)  # multi-GB; regenerated on demand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    spec = yaml.safe_load((ROOT / "configs/sweep.yaml").read_text())
    mcfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"][spec["model"]]
    qdir = ROOT / "results" / args.run_id / "raw" / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    text, meta = wikitext_test()
    (qdir / "dataset.json").write_text(json.dumps(meta, indent=2))

    models = {q: ROOT / "models" / f"{spec['model']}-{q}.gguf" for q in spec["quants"]}
    for q, m in models.items():
        print(f"[quality] llama-bench {q}", flush=True)
        bench(m, qdir / f"bench_{q}.json")

    ref = mcfg["reference_quant"]
    # Sanity check of the reference itself: Q8_0 vs the F16 source.
    kld_against("F16", ROOT / "models" / mcfg["file"], {ref: models[ref]}, text, qdir)
    kld_against(ref, models[ref], {q: m for q, m in models.items() if q != ref}, text, qdir)


if __name__ == "__main__":
    main()
