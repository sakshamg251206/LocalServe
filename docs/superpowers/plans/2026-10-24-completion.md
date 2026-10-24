# LocalServe completion plan

Goal: complete the README MINIMUM experiment and its reproducible serving pipeline.
Specification: README.md, docs/REPORT.md, configs/sweep.yaml.
Architecture: preserve the existing FastAPI gateway, HTTPX open-loop client, llama.cpp subprocesses,
and raw JSONL aggregation. No additional runtime dependencies.

1. Repair the relocated uv environment; run `make check` and real-model integration tests.
2. Add regression tests before fixes for truncated upstream streams, total deadlines, and incomplete
   capacity grids. Missing repeats or rates must never be presented as a completed capacity result.
3. Make reporting use the recorded experiment configuration, keep smoke reports separate from the
   published report, and ensure overhead serial data is not read as open-loop data.
4. Run offline checks and real-model smoke phases, including quality parser and report generation.
5. Run the full measurements serially on Metal, preserving old raw data and recording provenance;
   regenerate tables and figure only from actual results. Record any environmental blockers honestly.

Review focus: missing rate/repeat; missing SSE terminator; total timeout before first token;
serial overhead JSONL schema; changed configuration on resume.
Validation: `make check`, `make integration`, isolated smoke run and report, full measurement report.

Completion (2026-10-25): steps 1–5 finished. The ordered pipeline passed check/integration and
produced all configured measurements. See docs/LOG.md and results/minimum/summary.json.
