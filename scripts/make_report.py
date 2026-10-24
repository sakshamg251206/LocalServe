"""Build results/<run-id>/summary.json, figures, and the generated blocks inside README.md and
docs/REPORT.md (between <!-- BEGIN:name --> / <!-- END:name --> markers). Every number in the docs
comes from here; nothing is typed by hand.

  uv run python scripts/make_report.py --run-id minimum
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from localserve.manifest import ROOT
from localserve.report.aggregate import (
    bench_row,
    capacity_rows,
    overhead_openloop,
    overhead_serial,
    pool,
    read_phase,
)

# Categorical slots 1-3 of the validated default palette (dataviz skill); fixed order per quant.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]  # secondary encoding: identity never by colour alone
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SWAP_FLAG_BYTES = 64 * 2**20


def clean(x: Any) -> Any:
    """JSON-safe: inf -> "inf" (a p95 that includes failed requests), nan -> null."""
    if isinstance(x, float):
        return "inf" if math.isinf(x) else (None if math.isnan(x) else x)
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    return x


def f(x: Any, nd: int = 2, scale: float = 1.0, unit: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if x == "inf" or (isinstance(x, float) and math.isinf(x)):
        return "∞ (failures > 5%)"
    return f"{x * scale:.{nd}f}{unit}"


def quality(raw: Path, ref: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    q: dict[str, dict[str, Any]] = {}
    qdir = raw / "quality"
    for p in sorted(qdir.glob("bench_*.json")):
        q.setdefault(p.stem.removeprefix("bench_"), {}).update(bench_row(json.loads(p.read_text())))
    for p in sorted(qdir.glob("kld_*_vs_*.json")):
        quant, base = p.stem.removeprefix("kld_").split("_vs_")
        stats = json.loads(p.read_text())
        key = "kld" if base == ref else f"kld_vs_{base}"
        q.setdefault(quant, {})[key] = stats
    ds = qdir / "dataset.json"
    return q, (json.loads(ds.read_text()) if ds.exists() else {})


def failures(raw: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(raw.glob("*/failures.jsonl")):
        out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return out


def build_summary(run_dir: Path) -> dict[str, Any]:
    raw = run_dir / "raw"
    manifest_path = run_dir / "manifest.json"
    recorded = json.loads(manifest_path.read_text()).get("configs", {}) if manifest_path.exists() else {}
    configs = {
        name: recorded.get(name) or yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text())
        for name in ("sweep", "slo", "models")
    }
    sweep = configs["sweep"]
    sweep["rates"] = {int(k): v for k, v in sweep["rates"].items()}
    slo = configs["slo"]["chat"]["ttft_p95_s"]
    models = configs["models"]
    if "models" in models:
        models = models["models"]
    ref = models[sweep["model"]]["reference_quant"]
    cap_points = read_phase(raw, "capacity")
    curve, observed_caps = capacity_rows(cap_points, slo)
    caps = []
    for q in sweep["quants"]:
        for n in sweep["parallel"]:
            expected = {(float(r), rep) for r in sweep["rates"][n] for rep in range(sweep["repeats"])}
            observed = {
                (p.meta["rate"], p.meta["rep"])
                for p in cap_points
                if p.meta["quant"] == q and p.meta["parallel"] == n
            }
            missing = sorted(expected - observed)
            cap = next(
                (c for c in observed_caps if c["quant"] == q and c["parallel"] == n),
                {"quant": q, "parallel": n, "min_runs": 0},
            )
            if missing:
                cap = {**cap, "capacity_rps": None, "bound": "incomplete"}
            caps.append({**cap, "missing_points": [{"rate": r, "rep": rep} for r, rep in missing]})
    p70 = []
    for p in read_phase(raw, "p70"):
        p70.append(p)
    p70_rows = []
    for key in sorted({(p.meta["quant"], p.meta["parallel"], p.meta["rate"]) for p in p70}):
        pts = [p for p in p70 if (p.meta["quant"], p.meta["parallel"], p.meta["rate"]) == key]
        p70_rows.append({"quant": key[0], "parallel": key[1], "rate": key[2], **pool(pts)})
    ov_pts = read_phase(raw, "overhead")
    serial_path = raw / "overhead" / "serial_ab.jsonl"
    serial = None
    if serial_path.exists():
        rows = [json.loads(x) for x in serial_path.read_text().splitlines()[1:]]
        serial = overhead_serial(rows)
    q, dataset = quality(raw, ref)
    all_points = cap_points + p70 + ov_pts
    swapped = [
        {k: p.meta[k] for k in ("phase", "quant", "parallel", "rate", "rep", "target")}
        | {"swap_out_mb": p.meta.get("swap_out_bytes", 0) / 2**20}
        for p in all_points
        if p.meta.get("swap_out_bytes", 0) > SWAP_FLAG_BYTES
    ]
    return {
        "run_id": run_dir.name,
        "model": sweep["model"],
        "reference_quant": ref,
        "slo_chat_ttft_p95_s": slo,
        "quants": sweep["quants"],
        "parallel": sweep["parallel"],
        "capacity_curve": curve,
        "capacity": caps,
        "p70": p70_rows,
        "overhead": {"serial": serial, "openloop": overhead_openloop([p for p in ov_pts])},
        "quality": q,
        "dataset": dataset,
        "n_points": {"capacity": len(cap_points), "p70": len(p70), "overhead": len(ov_pts)},
        "runs_with_swap": swapped,
        "swap_flag_mb": SWAP_FLAG_BYTES / 2**20,
        "failures": failures(raw),
    }


def figure1(s: dict[str, Any], out: Path) -> None:
    """Latency-throughput curves: pooled p95 TTFT vs arrival rate, one panel per parallel setting."""
    curve = [c for c in s["capacity_curve"]]
    if not curve:
        return
    pars = s["parallel"]
    cap_y = 60.0  # draw p95 = inf at the top edge with an open marker
    fig, axes = plt.subplots(1, len(pars), figsize=(5.2 * len(pars), 4.0), sharey=True, facecolor=SURFACE)
    for ax, n in zip(axes, pars, strict=True):
        ax.set_facecolor(SURFACE)
        for i, q in enumerate(s["quants"]):
            rows = sorted(
                (c for c in curve if c["quant"] == q and c["parallel"] == n), key=lambda c: c["rate"]
            )
            if not rows:
                continue
            xs = [c["rate"] for c in rows]
            ys = [min(c["ttft_p95_slo_s"], cap_y) for c in rows]
            lo = [min(c["ttft_p95_slo_run_min_s"], cap_y) for c in rows]
            hi = [min(c["ttft_p95_slo_run_max_s"], cap_y) for c in rows]
            yerr = [
                [max(y - a, 0) for y, a in zip(ys, lo, strict=True)],
                [max(b - y, 0) for y, b in zip(ys, hi, strict=True)],
            ]
            ax.errorbar(xs, ys, yerr=yerr, color=COLORS[i], marker=MARKERS[i], ms=7, lw=2, capsize=3,
                        elinewidth=1, label=q, mec=SURFACE, mew=1.5)  # fmt: skip
            for x, c in zip(xs, rows, strict=True):
                if math.isinf(c["ttft_p95_slo_s"]):
                    ax.plot(x, cap_y, marker=MARKERS[i], ms=9, mfc="none", mec=COLORS[i], mew=1.5)
        ax.axhline(s["slo_chat_ttft_p95_s"], color=INK2, lw=1, ls="--")
        ax.text(ax.get_xlim()[0], s["slo_chat_ttft_p95_s"] * 1.12, f" SLO {s['slo_chat_ttft_p95_s']:g} s",
                color=INK2, fontsize=9)  # fmt: skip
        ax.set_yscale("log")
        ax.set_title(f"--parallel {n}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("arrival rate (req/s, Poisson)", color=INK2)
        ax.grid(True, which="major", color="#e6e5e0", lw=0.8)
        ax.tick_params(colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c9c8c2")
    axes[0].set_ylabel("p95 TTFT, s (log; failures = ∞, open marker)", color=INK2)
    axes[-1].legend(frameon=False, title="quant", labelcolor=INK)
    fig.suptitle(f"{s['model']}: p95 time-to-first-token vs load (bars = run min-max)", color=INK, x=0.01,
                 ha="left", fontsize=12)  # fmt: skip
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _kld(qrow: dict[str, Any]) -> str:
    k = qrow.get("kld")
    if not k:
        return "0 (reference)" if qrow.get("_is_ref") else "–"
    return f"{k['kld_mean']:.4f} ± {k.get('kld_mean_err', math.nan):.4f}"


def table1(s: dict[str, Any]) -> str:
    ref = s["reference_quant"]
    lines = [
        "| Quant | File size (GB) | Prompt tok/s (pp512) | Gen tok/s (tg128) | Mean KLD vs " + ref
        + " | Same top token vs " + ref + " (%) | PPL (wikitext-2 subset) |",
        "|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for q in s["quants"]:
        r = {**s["quality"].get(q, {}), "_is_ref": q == ref}
        k = r.get("kld") or {}
        ppl = k.get("ppl_q") if k else (r.get("kld_vs_F16", {}) or {}).get("ppl_q")
        top = (
            f"{k['same_top_p_pct']:.2f} ± {k.get('same_top_p_pct_err', math.nan):.2f}"
            if k
            else ("100 (reference)" if q == ref else "–")
        )
        lines.append(
            f"| {q} | {f(r.get('model_size_bytes'), 3, 1e-9)} | {f(r.get('pp_tok_s'), 0)} ± {f(r.get('pp_tok_s_sd'), 0)} "
            f"| {f(r.get('tg_tok_s'), 1)} ± {f(r.get('tg_tok_s_sd'), 1)} | {_kld(r)} | {top} | {f(ppl, 3)} |"
        )
    sanity = (s["quality"].get(ref, {}) or {}).get("kld_vs_F16")
    if sanity:
        lines.append("")
        lines.append(
            f"Reference check — {ref} vs the F16 source: mean KLD {sanity['kld_mean']:.5f} ± "
            f"{sanity.get('kld_mean_err', math.nan):.5f}, same top token {sanity['same_top_p_pct']:.2f}%."
        )
    return "\n".join(lines)


def table2(s: dict[str, Any]) -> str:
    p70 = {(r["quant"], r["parallel"]): r for r in s["p70"]}
    curve = s["capacity_curve"]
    lines = [
        f"| Quant | --parallel | Mean KLD vs {s['reference_quant']} | Capacity (req/s @ p95 TTFT ≤ {s['slo_chat_ttft_p95_s']:g} s) "
        "| Rate tested at 70% | p95 TTFT @70% (s) | p95 ITL @70% (ms) | Output tok/s @70% | Server footprint peak (GB) | Runs/point |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for c in s["capacity"]:
        q, n = c["quant"], c["parallel"]
        r = p70.get((q, n), {})
        bound = {
            "exact": "",
            "lower_bound": " (≥, grid max)",
            "below_grid": "< lowest rate",
            "incomplete": "incomplete",
        }[c["bound"]]
        capv = f"{c['capacity_rps']:g}{bound}" if c["capacity_rps"] is not None else bound
        foot = max(
            (x["footprint_peak_bytes_max"] for x in curve if x["quant"] == q and x["parallel"] == n),
            default=None,
        )
        qrow = {**s["quality"].get(q, {}), "_is_ref": q == s["reference_quant"]}
        lines.append(
            f"| {q} | {n} | {_kld(qrow)} | {capv} | {f(r.get('rate'), 3)} | {f(r.get('ttft_p95_slo_s'), 2)} "
            f"| {f(r.get('itl_p95_s'), 1, 1000)} | {f(r.get('output_tok_s_mean'), 1)} | {f(foot, 2, 1e-9)} | {c['min_runs']} |"
        )
    return "\n".join(lines)


def overhead_block(s: dict[str, Any]) -> str:
    o = s["overhead"]
    out = []
    if o["serial"]:
        se = o["serial"]
        t, e = se["ttft_s"], se["e2e_s"]
        out.append(
            f"Idle server, one request at a time, alternating direct/gateway ({se['n_pairs']} pairs): median TTFT "
            f"{f(t['direct_median'], 1, 1000)} ms direct vs {f(t['gateway_median'], 1, 1000)} ms via gateway; "
            f"paired median difference **{f(t['diff_median'], 2, 1000)} ms** (95% bootstrap CI "
            f"{f(t['diff_ci95'][0], 2, 1000)} to {f(t['diff_ci95'][1], 2, 1000)} ms). End-to-end: "
            f"{f(e['diff_median'], 2, 1000)} ms (CI {f(e['diff_ci95'][0], 2, 1000)} to {f(e['diff_ci95'][1], 2, 1000)} ms)."
        )
    if o["openloop"]:
        out += ["", "| Rate (req/s) | Path | Runs | Requests | p50 TTFT (ms) | p95 TTFT (s) | p95 ITL (ms) | Output tok/s |",
                "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in o["openloop"]:
            out.append(
                f"| {r['rate']:g} | {r['target']} | {r['n_runs']} | {r['n_requests']} | {f(r['ttft_p50_s'], 0, 1000)} "
                f"| {f(r['ttft_p95_slo_s'], 2)} | {f(r['itl_p95_s'], 1, 1000)} | {f(r['output_tok_s_mean'], 1)} |"
            )
    return "\n".join(out) if out else "_Not run yet._"


def findings(s: dict[str, Any]) -> str:
    out = []
    pt = [c["prompt_tokens_mean"] for c in s["capacity_curve"] if not math.isnan(c["prompt_tokens_mean"])]
    if pt:
        out.append(f"- Mean prompt length (server-counted, incl. chat template): {sum(pt) / len(pt):.0f} tokens; "
                   f"output fixed at 128 tokens.")  # fmt: skip
    for n in s["parallel"]:
        caps = [c for c in s["capacity"] if c["parallel"] == n and c["capacity_rps"] is not None]
        if caps:
            best = max(caps, key=lambda c: c["capacity_rps"])
            out.append(f"- `--parallel {n}`: highest measured capacity {best['capacity_rps']:g} req/s ({best['quant']}"
                       f"{', lower bound' if best['bound'] == 'lower_bound' else ''}).")  # fmt: skip
    for q in s["quants"]:
        c = {c["parallel"]: c for c in s["capacity"] if c["quant"] == q}
        if len(s["parallel"]) >= 2:
            a, b = c.get(s["parallel"][0]), c.get(s["parallel"][-1])
            if a and b and a["capacity_rps"] and b["capacity_rps"]:
                out.append(f"- {q}: capacity ×{b['capacity_rps'] / a['capacity_rps']:.2f} going from `--parallel "
                           f"{s['parallel'][0]}` to `--parallel {s['parallel'][-1]}` (grid resolution limits precision).")  # fmt: skip
    if s["overhead"]["serial"]:
        t = s["overhead"]["serial"]["ttft_s"]
        out.append(f"- Gateway TTFT overhead (idle, paired): {f(t['diff_median'], 2, 1000)} ms median "
                   f"(95% CI {f(t['diff_ci95'][0], 2, 1000)}–{f(t['diff_ci95'][1], 2, 1000)} ms); target was < 5 ms.")  # fmt: skip
    return "\n".join(out) if out else "_No results yet._"


def data_caveats(s: dict[str, Any]) -> str:
    out = [f"- Measured points: {s['n_points']}. Failures recorded: {len(s['failures'])}."]
    missing = sum(len(c.get("missing_points", [])) for c in s["capacity"])
    if missing:
        out.append(f"- Capacity sweep incomplete: {missing} required points missing; capacity withheld.")
    for fl in s["failures"]:
        out.append(f"  - {json.dumps(fl)}")
    out.append(f"- Runs where the OS swapped out > {s['swap_flag_mb']:.0f} MB during measurement: "
               f"{len(s['runs_with_swap'])} of {sum(s['n_points'].values())}.")  # fmt: skip
    small = [c for c in s["capacity_curve"] if c["n_requests"] < 20]
    if small:
        out.append(f"- {len(small)} curve points have < 20 requests pooled over all runs (low rates × 60 s windows); "
                   "their p95 is close to the sample maximum and noisy.")  # fmt: skip
    ds = s.get("dataset") or {}
    if ds:
        out.append(f"- KLD text: {ds['name']}, first {ds['chunks']} chunks × {ds['ctx']} tokens (subset; disk limit on "
                   f"reference logits), sha256 {ds['test_sha256'][:12]}…, licence {ds['licence']}.")  # fmt: skip
    return "\n".join(out)


def inject(doc: Path, blocks: dict[str, str]) -> None:
    if not doc.exists():
        return
    text = doc.read_text()
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->)\n.*?(<!-- END:{name} -->)", re.S)

        def fill(m: re.Match[str], body: str = body) -> str:
            return m.group(1) + "\n" + body + "\n" + m.group(2)

        text = pat.sub(fill, text)
    doc.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_dir = ROOT / "results" / args.run_id
    s = build_summary(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(clean(s), indent=2, allow_nan=False))
    figure1(s, run_dir / "figures" / "fig1_latency_throughput.png")
    blocks = {"table1": table1(s), "table2": table2(s), "overhead": overhead_block(s),
              "findings": findings(s), "data_caveats": data_caveats(s)}  # fmt: skip
    (run_dir / "tables.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in blocks.items()) + "\n")
    if args.run_id == "minimum":
        for doc in (ROOT / "README.md", ROOT / "docs" / "REPORT.md"):
            inject(doc, blocks)
    print((run_dir / "tables.md").read_text())


if __name__ == "__main__":
    main()
