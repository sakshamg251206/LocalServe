## table1

| Quant | File size (GB) | Prompt tok/s (pp512) | Gen tok/s (tg128) | Mean KLD vs Q8_0 | Same top token vs Q8_0 (%) | PPL (wikitext-2 subset) |
|---|---|---|---|---|---|---|
| Q8_0 | 2.159 | 738 ± 4 | 46.6 ± 0.2 | 0 (reference) | 100 (reference) | 17.061 |
| Q5_K_M | 1.466 | 599 ± 2 | 54.2 ± 0.2 | 0.0297 ± 0.0005 | 92.58 ± 0.16 | 17.323 |
| Q4_K_M | 1.276 | 663 ± 1 | 67.5 ± 1.3 | 0.1020 ± 0.0016 | 87.13 ± 0.21 | 18.674 |

Reference check — Q8_0 vs the F16 source: mean KLD 0.00147 ± 0.00002, same top token 98.37%.

## table2

| Quant | --parallel | Mean KLD vs Q8_0 | Capacity (req/s @ p95 TTFT ≤ 1 s) | Rate tested at 70% | p95 TTFT @70% (s) | p95 ITL @70% (ms) | Output tok/s @70% | Server footprint peak (GB) | Runs/point |
|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | 1 | 0 (reference) | < lowest rate | – | – | – | – | 3.25 | 3 |
| Q8_0 | 4 | 0 (reference) | 0.2 | 0.140 | 0.36 | 47.3 | 25.6 | 6.83 | 3 |
| Q5_K_M | 1 | 0.0297 ± 0.0005 | < lowest rate | – | – | – | – | 3.23 | 3 |
| Q5_K_M | 4 | 0.0297 ± 0.0005 | 0.1 | 0.070 | 0.40 | 19.5 | 4.3 | 6.83 | 3 |
| Q4_K_M | 1 | 0.1020 ± 0.0016 | < lowest rate | – | – | – | – | 3.69 | 3 |
| Q4_K_M | 4 | 0.1020 ± 0.0016 | 0.2 | 0.140 | 0.36 | 42.3 | 25.6 | 6.95 | 3 |

## overhead

Idle server, one request at a time, alternating direct/gateway (100 pairs): median TTFT 173.0 ms direct vs 177.8 ms via gateway; paired median difference **3.87 ms** (95% bootstrap CI -6.89 to 9.15 ms). End-to-end: 2.31 ms (CI -3.96 to 12.86 ms).

| Rate (req/s) | Path | Runs | Requests | p50 TTFT (ms) | p95 TTFT (s) | p95 ITL (ms) | Output tok/s |
|---|---|---|---|---|---|---|---|
| 0.2 | direct | 3 | 34 | 180 | 0.77 | 76.6 | 24.2 |
| 0.2 | gateway | 3 | 34 | 110 | 0.48 | 44.6 | 24.2 |
| 0.4 | direct | 3 | 65 | 346 | 5.22 | 89.7 | 46.2 |
| 0.4 | gateway | 3 | 65 | 365 | 9.94 | 105.3 | 46.2 |

## findings

- Mean prompt length (server-counted, incl. chat template): 177 tokens; output fixed at 128 tokens.
- `--parallel 4`: highest measured capacity 0.2 req/s (Q8_0).
- Gateway TTFT overhead (idle, paired): 3.87 ms median (95% CI -6.89–9.15 ms); target was < 5 ms.

## data_caveats

- Measured points: {'capacity': 108, 'p70': 9, 'overhead': 12}. Failures recorded: 0.
- Runs where the OS swapped out > 64 MB during measurement: 1 of 129.
- 3 curve points have < 20 requests pooled over all runs (low rates × 60 s windows); their p95 is close to the sample maximum and noisy.
- KLD text: wikitext-2-raw-v1 (test split), first 100 chunks × 512 tokens (subset; disk limit on reference logits), sha256 173c87a53759…, licence CC BY-SA 3.0 (Merity et al., 2016).
