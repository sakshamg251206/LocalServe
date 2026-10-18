import pytest

from localserve.quality.kld import parse_kld_output

# Layout of llama-perplexity's KLD summary (values made up for the parser test).
FIXTURE = """
====== Perplexity statistics ======
Mean PPL(Q)                   :  12.345678 ±   0.123456
Mean PPL(base)                :  12.000000 ±   0.120000
Cor(ln(PPL(Q)), ln(PPL(base))):  99.12%
Mean ln(PPL(Q)/PPL(base))     :   0.028403 ±   0.001234
Mean PPL(Q)/PPL(base)         :   1.028810 ±   0.001270

====== KL divergence statistics ======
Mean    KLD:   0.034567 ±   0.000456
Maximum KLD:   5.678900
99.9%   KLD:   1.234500
99.0%   KLD:   0.456700
Median  KLD:   0.012300
 1.0%   KLD:   0.000012
Minimum KLD:  -0.000001

====== Token probability statistics ======
Mean    Δp: -0.456 ± 0.034 %
RMS Δp    :  4.567 ± 0.089 %
Same top p: 93.456 ± 0.123 %
"""


def test_parse_kld_output() -> None:
    s = parse_kld_output(FIXTURE)
    assert s["kld_mean"] == pytest.approx(0.034567)
    assert s["kld_mean_err"] == pytest.approx(0.000456)
    assert s["same_top_p_pct"] == pytest.approx(93.456)
    assert s["same_top_p_pct_err"] == pytest.approx(0.123)
    assert s["ppl_q"] == pytest.approx(12.345678)
    assert s["ppl_base"] == pytest.approx(12.0)
    assert s["kld_p99"] == pytest.approx(0.4567)
    assert s["kld_median"] == pytest.approx(0.0123)
    assert s["kld_max"] == pytest.approx(5.6789)
    assert s["rms_dp_pct"] == pytest.approx(4.567)
    assert "kld_max_err" not in s


def test_parse_ignores_unrelated_text() -> None:
    assert parse_kld_output("llama_model_loader: foo\nMean speed: 3 t/s\n") == {}
