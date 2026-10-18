"""KL divergence of a quantized model vs a reference, via `llama-perplexity --kl-divergence`.

For each scored token position t, the reference model gives a next-token distribution P_t and the
quantized model gives Q_t. KLD_t = sum_v P_t(v) * log(P_t(v) / Q_t(v)) >= 0; it is 0 only when the
distributions match. Mean KLD averages over positions. Unlike perplexity, which looks only at the
probability of the *actual* next token, KLD compares the whole distribution, so it catches changes
in which wrong tokens are likely (what sampling and constrained/JSON output depend on).
"Same top p" = % of positions where both models' most likely token is the same.
"""

from __future__ import annotations

import re
from pathlib import Path

from localserve.manifest import LLAMA_BIN

# e.g. "Mean    KLD:   0.012345 ±   0.000123" or "Same top p: 95.123 ± 0.123 %"
_STAT = re.compile(
    r"^(?P<key>[A-Za-z0-9 .%()/Δ,\-]+?)\s*:\s*(?P<val>-?[\d.]+)(?:\s*±\s*(?P<err>[\d.]+))?\s*%?\s*$"
)

WANTED = {
    "Mean PPL(Q)": "ppl_q",
    "Mean PPL(base)": "ppl_base",
    "Mean ln(PPL(Q)/PPL(base))": "ln_ppl_ratio",
    "Mean KLD": "kld_mean",
    "Maximum KLD": "kld_max",
    "99.0% KLD": "kld_p99",
    "Median KLD": "kld_median",
    "RMS Δp": "rms_dp_pct",
    "Same top p": "same_top_p_pct",
}


def parse_kld_output(text: str) -> dict[str, float]:
    """Pick the statistics we report out of llama-perplexity's KLD summary; `<name>_err` = ± value."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _STAT.match(line.strip())
        if not m:
            continue
        key = " ".join(m.group("key").split())  # collapse alignment spaces
        name = WANTED.get(key)
        if name is None:
            continue
        out[name] = float(m.group("val"))
        if m.group("err") is not None:
            out[name + "_err"] = float(m.group("err"))
    return out


def base_cmd(model: Path, text: Path, logits_out: Path, ctx: int, chunks: int) -> list[str]:
    """Run the reference model and save its full logits for every scored position."""
    return [str(LLAMA_BIN / "llama-perplexity"), "-m", str(model), "-f", str(text),
            "--kl-divergence-base", str(logits_out), "-c", str(ctx), "--chunks", str(chunks),
            "-ngl", "99"]  # fmt: skip


def kld_cmd(model: Path, logits_base: Path, ctx: int, chunks: int) -> list[str]:
    """Compare a model against saved reference logits (the text is read from the logits file)."""
    return [str(LLAMA_BIN / "llama-perplexity"), "-m", str(model),
            "--kl-divergence-base", str(logits_base), "--kl-divergence", "-c", str(ctx),
            "--chunks", str(chunks), "-ngl", "99"]  # fmt: skip
