"""Run benchmark phases. Raw results go to results/<run-id>/raw/<phase>/. Resumable: finished points
(files that exist) are skipped, and the interleaved order is planned up front from a fixed seed so
a resumed run follows the same plan.

  uv run python scripts/run_sweep.py --run-id minimum --phase capacity
  uv run python scripts/make_report.py --run-id minimum          # computes capacity
  uv run python scripts/run_sweep.py --run-id minimum --phase p70
  uv run python scripts/run_sweep.py --run-id minimum --phase overhead
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import psutil
import yaml

from localserve.loadgen.client import RequestResult, send_chat
from localserve.loadgen.run import LoadSpec, chat_prompts, run_load
from localserve.manifest import ROOT, base_manifest
from localserve.server import footprint, gateway, llama_server, llama_server_cmd

RESULTS = ROOT / "results"


def load_configs() -> dict[str, Any]:
    c = {name: yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text()) for name in ("sweep", "slo")}
    c["models"] = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]
    c["llama_cpp"] = yaml.safe_load((ROOT / "configs/llama_cpp.yaml").read_text())
    return c


def model_file(model: str, quant: str) -> Path:
    return ROOT / "models" / f"{model}-{quant}.gguf"


def alias(model: str, quant: str) -> str:
    return f"{model}-{quant.lower()}"


def point_path(raw: Path, phase: str, quant: str, np_: int, rate: float, rep: int, target: str) -> Path:
    return raw / phase / f"{quant}_np{np_}_r{rate:g}_rep{rep}_{target}.jsonl"


def write_jsonl(path: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)  # atomic: a crash mid-write never leaves a file that looks finished


def log_failure(raw: Path, phase: str, what: dict[str, Any], err: BaseException) -> None:
    raw.joinpath(phase).mkdir(parents=True, exist_ok=True)
    with raw.joinpath(phase, "failures.jsonl").open("a") as f:
        f.write(json.dumps({**what, "error": repr(err), "time": time.time()}) + "\n")
    print(f"FAILED {what}: {err!r}", flush=True)


def therm() -> str:
    r = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True)
    return " | ".join(line.strip() for line in r.stdout.splitlines() if line.strip())


def measure(path: Path, spec: LoadSpec, pid: int, meta: dict[str, Any]) -> None:
    swap0 = psutil.swap_memory().sout
    t0 = time.time()
    results, resources = asyncio.run(run_load(spec, pid))
    meta = {
        **meta,
        "spec": asdict(spec),
        "t_start": t0,
        "t_end": time.time(),
        "swap_out_bytes": psutil.swap_memory().sout - swap0,  # >0: the OS paged out during the run
        "therm": therm(),
        **resources,
        **footprint(pid),
    }
    write_jsonl(path, meta, [asdict(r) for r in results])
    ok = sum(r.ok for r in results)
    print(f"  wrote {path.name}: {ok}/{len(results)} ok", flush=True)


@contextmanager
def session(cfg: dict[str, Any], quant: str, np_: int, logs: Path) -> Iterator[int]:
    """llama-server + gateway for one (quant, parallel) config; yields the llama-server pid."""
    s = cfg["sweep"]
    ports = s["ports"]
    tag = f"{quant}_np{np_}"
    with (
        llama_server(
            model_file(s["model"], quant),
            alias(s["model"], quant),
            np_,
            s["ctx_per_slot"],
            ports["llama"],
            logs / f"llama_{tag}.log",
        ) as srv,
        gateway(
            ports["llama"], ports["gateway"], np_, s["gateway"]["max_queue"], logs / f"gateway_{tag}.log"
        ),
    ):
        yield srv.pid


def spec_for(cfg: dict[str, Any], quant: str, rate: float, seed: int, base_url: str) -> LoadSpec:
    s = cfg["sweep"]
    return LoadSpec(
        base_url=base_url,
        model=alias(s["model"], quant),
        rate=rate,
        duration_s=s["duration_s"],
        warmup_s=s["warmup_s"],
        max_tokens=s["max_tokens"],
        prompt_words=s["prompt_words"],
        request_timeout_s=s["request_timeout_s"],
        seed=seed,
    )


def run_grid(
    cfg: dict[str, Any], run_dir: Path, phase: str, rates_for: dict[tuple[str, int], list[float]]
) -> None:
    """For each repeat: visit configs in shuffled order; within a config, rates in shuffled order."""
    s = cfg["sweep"]
    raw, logs = run_dir / "raw", run_dir / "logs"
    gw_url = f"http://127.0.0.1:{s['ports']['gateway']}"
    rng = np.random.default_rng(s["order_seed"])
    configs = sorted(rates_for)
    plan = []
    for rep in range(s["repeats"]):
        for ci in rng.permutation(len(configs)):
            q, n = configs[ci]
            rates = rates_for[(q, n)]
            plan.append((rep, q, n, [rates[j] for j in rng.permutation(len(rates))]))
    for rep, q, n, rates in plan:
        todo = [r for r in rates if not point_path(raw, phase, q, n, r, rep, "gateway").exists()]
        if not todo:
            continue
        print(f"[{phase}] rep {rep} {q} np={n} rates={todo}", flush=True)
        try:
            with session(cfg, q, n, logs) as pid:
                for r in todo:
                    seed = s["seed_base"] + 100 * rep + round(r * 1000)
                    meta = {
                        "phase": phase,
                        "quant": q,
                        "parallel": n,
                        "rate": r,
                        "rep": rep,
                        "target": "gateway",
                    }
                    meta["server_cmd"] = llama_server_cmd(
                        model_file(s["model"], q),
                        alias(s["model"], q),
                        n,
                        s["ctx_per_slot"],
                        s["ports"]["llama"],
                    )
                    try:
                        measure(
                            point_path(raw, phase, q, n, r, rep, "gateway"),
                            spec_for(cfg, q, r, seed, gw_url),
                            pid,
                            meta,
                        )
                    except Exception as e:  # record and continue; never drop a condition silently
                        log_failure(raw, phase, {"quant": q, "parallel": n, "rate": r, "rep": rep}, e)
                    time.sleep(s["cooldown_s"])
        except Exception as e:
            log_failure(raw, phase, {"quant": q, "parallel": n, "rep": rep, "stage": "server_start"}, e)


def phase_capacity(cfg: dict[str, Any], run_dir: Path) -> None:
    s = cfg["sweep"]
    rates = {(q, n): list(s["rates"][n]) for q in s["quants"] for n in s["parallel"]}
    run_grid(cfg, run_dir, "capacity", rates)


def phase_p70(cfg: dict[str, Any], run_dir: Path) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    rates: dict[tuple[str, int], list[float]] = {}
    for row in summary["capacity"]:
        if row["capacity_rps"] is None:
            print(f"skip p70 for {row['quant']} np={row['parallel']}: capacity below grid", flush=True)
            continue
        rate = round(cfg["sweep"]["p70_fraction"] * row["capacity_rps"], 4)
        rates[(row["quant"], row["parallel"])] = [rate]
    run_grid(cfg, run_dir, "p70", rates)


async def _serial_ab(direct: str, gw: str, cfg: dict[str, Any], quant: str) -> list[dict[str, Any]]:
    o = cfg["sweep"]["overhead"]
    prompts = chat_prompts(2 * o["serial_pairs"], cfg["sweep"]["prompt_words"], seed=99)
    rows = []
    async with httpx.AsyncClient(base_url=direct) as cd, httpx.AsyncClient(base_url=gw) as cg:
        spec = spec_for(cfg, quant, 1.0, 0, direct)
        for i in range(o["serial_pairs"]):
            order = [("direct", cd), ("gateway", cg)] if i % 2 == 0 else [("gateway", cg), ("direct", cd)]
            for k, (target, client) in enumerate(order):
                body = {**spec.body(prompts[2 * i + k]), "max_tokens": o["serial_max_tokens"]}
                res: RequestResult = await send_chat(client, "/v1/chat/completions", body, 60.0, idx=i)
                rows.append({"pair": i, "target": target, **asdict(res)})
                await asyncio.sleep(0.2)
    return rows


def phase_overhead(cfg: dict[str, Any], run_dir: Path) -> None:
    s, o = cfg["sweep"], cfg["sweep"]["overhead"]
    q, n = o["quant"], o["parallel"]
    raw, logs = run_dir / "raw", run_dir / "logs"
    direct = f"http://127.0.0.1:{s['ports']['llama']}"
    gw = f"http://127.0.0.1:{s['ports']['gateway']}"
    with session(cfg, q, n, logs) as pid:
        serial = raw / "overhead" / "serial_ab.jsonl"
        if not serial.exists():
            print("[overhead] serial A/B", flush=True)
            asyncio.run(_serial_ab(direct, gw, cfg, q))  # warm both paths once (discarded)
            rows = asyncio.run(_serial_ab(direct, gw, cfg, q))
            write_jsonl(serial, {"phase": "overhead", "kind": "serial_ab", "quant": q, "parallel": n}, rows)
        for rep in range(s["repeats"]):
            targets = [("direct", direct), ("gateway", gw)]
            if rep % 2:
                targets.reverse()
            for r in o["rates"]:
                for target, url in targets:
                    path = point_path(raw, "overhead", q, n, r, rep, target)
                    if path.exists():
                        continue
                    seed = s["seed_base"] + 100 * rep + round(r * 1000)  # identical schedule per target
                    meta = {
                        "phase": "overhead",
                        "quant": q,
                        "parallel": n,
                        "rate": r,
                        "rep": rep,
                        "target": target,
                    }
                    print(f"[overhead] rep {rep} rate {r} {target}", flush=True)
                    try:
                        measure(path, spec_for(cfg, q, r, seed, url), pid, meta)
                    except Exception as e:
                        log_failure(raw, "overhead", meta, e)
                    time.sleep(s["cooldown_s"])


def update_manifest(run_dir: Path, cfg: dict[str, Any], phase: str, event: str) -> None:
    path = run_dir / "manifest.json"
    man: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    lock_path = ROOT / "models/models.lock.json"
    man.update(
        {
            "run_id": run_dir.name,
            "configs": {k: cfg[k] for k in ("sweep", "slo", "llama_cpp", "models")},
            "model_files": json.loads(lock_path.read_text()) if lock_path.exists() else {},
        }
    )
    man.setdefault("phases", {}).setdefault(phase, {})[event] = {"time": time.time(), **base_manifest()}
    path.write_text(json.dumps(man, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--phase", choices=["capacity", "p70", "overhead"], required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing check; not a result")
    args = ap.parse_args()
    cfg = load_configs()
    if args.smoke:
        s = cfg["sweep"]
        s.update(quants=s["quants"][-1:], repeats=1, duration_s=8, warmup_s=2, cooldown_s=1)
        s["rates"] = {n: s["rates"][n][:1] for n in s["parallel"]}
        s["overhead"].update(serial_pairs=3, rates=s["overhead"]["rates"][:1])
    run_dir = RESULTS / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    update_manifest(run_dir, cfg, args.phase, "started")
    {"capacity": phase_capacity, "p70": phase_p70, "overhead": phase_overhead}[args.phase](cfg, run_dir)
    update_manifest(run_dir, cfg, args.phase, "finished")


if __name__ == "__main__":
    main()
