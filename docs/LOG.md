# LOG

What was run, what failed, what changed — in order. Benchmark numbers live in
`results/<run-id>/summary.json` and the generated blocks of `REPORT.md`, not here.

## 2026-10-18

### Step 0 — hardware check → re-scope
- Machine: MacBook Air (Mac14,2), Apple M2 (4P+4E CPU, 8-core GPU), **8 GB unified memory**, fanless,
  macOS 27.0.1. Exact values are recorded in every `manifest.json`.
- The plan required ≥ 16 GB, so work stopped and a re-scope was proposed and approved:
  - 2k context per slot (not 4k); the "long ≈4k tokens" prompt class is dropped (limitation).
  - STRONG scope, when approved: Qwen3-4B served only at Q5_K_M/Q4_K_M/Q3_K_M, with Q8_0 used only offline
    as the quality reference; parallel {1,2,4}; no F16 serving.
  - Thermal: interleaved order + cool-down between points; timestamps and `pmset -g therm` per run.
- Memory at the start of the session: heavy swap already in use and about 1 GB available, because of other apps
  (browser, editor, assistant sessions). These apps can't be closed from here, so every run records
  `swap_out_bytes` (bytes the OS paged out during that run) and the report counts flagged runs.
- llama.cpp pinned at `5aeac37a` (b11719), built with Metal (`configs/llama_cpp.yaml`,
  `scripts/build_llama.sh`). Metal reported `recommendedMaxWorkingSetSize` in the build log
  (the GPU working-set limit on this machine).

### Step 1 — gateway (fake upstream first)
- FastAPI gateway: SSE streaming, bounded admission (429 + `Retry-After`), first-token and total
  timeouts, `/metrics` (Prometheus), one JSON log line per request. 14 tests against a fake async upstream.
- **Bug found by my own test** (loadgen 429/timeout test): a client that hung up while *queued or awaiting
  its first token* kept its slot until upstream produced a token. Fixed by watching for `http.disconnect`
  before headers are sent; regression test added.
- **Bug found by automated security review**: if sending response headers failed, the relay generator
  never started, its `finally` never ran, and the concurrency slot leaked permanently. Fixed with an
  idempotent cleanup that also runs from the response's `__call__`; regression test drives the ASGI app
  with a failing `send`.

### Step 2 — load generator
- Open-loop Poisson arrivals; nearest-rank percentiles (numpy interpolation returns NaN when failed requests
  are counted as infinite TTFT, so "linear" can't be used). Requests are selected by *scheduled* time.
- Against the real server: `llama-server` sends a role-only first chunk (`content: null`) before any token,
  so TTFT is measured to the first *token-bearing* event (gateway and client both).
- Against the real server: one SSE event is not always one token (the server occasionally merges a held-back
  token into the next event). Output token counts now come from `usage.completion_tokens`; TPOT added;
  ITL is defined as the gap between token-bearing events.

### Step 3 — models
- Source: `ggml-org/Qwen3-1.7B-GGUF` F16 at a pinned revision, sha256 verified. Licence Apache-2.0 (model card).
- All quants are made locally with `llama-quantize` (no imatrix) so the quant level is the only difference.
  Check: the local Q8_0 is bit-identical (same sha256) to ggml-org's published Q8_0.
- Memory accounting: psutil RSS misses Metal buffers on Apple Silicon, so `phys_footprint` from
  `/usr/bin/footprint` is recorded as well. Under memory pressure the two disagree in both directions
  (compression/swap), so neither is a clean "peak memory"; both are reported.

### Pilot (not a result)
- Two 30 s runs on Q4_K_M with 4 slots, used only to place the rate grid in `configs/sweep.yaml`:
  the higher pilot rate was far past saturation and the lower one already showed queueing bursts.
  Pilot data was not saved and is not used anywhere.

### Step 4 — MINIMUM sweep
- Capacity sweep started 21:42 IST: 3 quants × parallel {1,4} × 6 rates × 3 repeats = 108 points,
  60 s measured + 15 s warm-up each, interleaved order, 15 s cool-down.
- Each point takes ~95 s wall time (75 s load + drain + 15 s cool-down), so the capacity phase is ~3 h.
  (I first logged "3–4 min per point" from a misread clock; checked against `t_start`/`t_end` and corrected.)
- Development (editing, occasional unit-test runs at `nice 19`) continued on the same machine during the
  sweep; this is a small extra source of noise.
- KLD reference logits cost ~300 KB per scored token (152k vocabulary); the full wikitext-2 test split would
  need tens of GB, more than the free disk. The KLD uses the first 100 chunks × 512 tokens (stated in the report).
