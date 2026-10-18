# Operating the gateway

## Run

```bash
# 1. model server (one model at a time on 8 GB)
.vendor/llama.cpp/build/bin/llama-server -m models/qwen3-1.7b-Q4_K_M.gguf --alias qwen3-1.7b-q4_k_m \
    -np 4 -c 8192 -ngl 99 --host 127.0.0.1 --port 8080 --metrics
# 2. gateway
LOCALSERVE_MAX_CONCURRENT=4 uv run localserve-gateway      # http://127.0.0.1:8000/v1
```

| Env var | Default | Meaning |
|---|---|---|
| `LOCALSERVE_UPSTREAM` | `http://127.0.0.1:8080` | llama-server base URL |
| `LOCALSERVE_MAX_CONCURRENT` | 4 | requests sent upstream at once; set equal to `--parallel` |
| `LOCALSERVE_MAX_QUEUE` | 64 | requests allowed to wait; beyond `concurrent + queue` → 429 |
| `LOCALSERVE_FIRST_TOKEN_TIMEOUT` | 30 | seconds from arrival to first token, else 504 |
| `LOCALSERVE_REQUEST_TIMEOUT` | 300 | total seconds; mid-stream expiry ends the stream with an error event |
| `LOCALSERVE_HOST` / `LOCALSERVE_PORT` | 127.0.0.1 / 8000 | bind address |

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible, streaming or not. The body is forwarded unchanged,
  so `tools`, `chat_template_kwargs` (e.g. `{"enable_thinking": false}` for Qwen3) and llama.cpp
  extensions (`ignore_eos`, `cache_prompt`) pass through.
- `GET /v1/models` — proxied from llama-server; shows which model is actually loaded.
- `GET /metrics` — Prometheus: `localserve_requests_total{outcome}`, `localserve_ttft_seconds`,
  `localserve_e2e_seconds`, `localserve_queue_wait_seconds`, `localserve_output_chunks_total`,
  `localserve_inflight`, `localserve_queued`.
- `GET /healthz`.

## Behaviour under load and failure

| Situation | Response |
|---|---|
| more than `concurrent + queue` requests | `429` + `Retry-After: 1`, immediately |
| no first token within the first-token timeout | `504`; upstream request cancelled |
| client disconnects while queued / waiting for first token | upstream cancelled, slot freed, logged as `cancelled` |
| client disconnects mid-stream | upstream connection closed (llama-server stops that slot), slot freed |
| total timeout mid-stream | final SSE `{"error": {"type": "timeout"}}`, no `[DONE]` |
| upstream returns non-200 | status and body passed through |
| upstream unreachable | `502` |

Each request produces one JSON log line on stdout: `request_id, model, outcome, status, queue_wait_s,
ttft_s, e2e_s, chunks`.

## Security

Binds to localhost by default. There's no authentication, and llama-server itself warns that it has no
API key and allows all CORS origins. Don't expose either port beyond localhost without putting auth in front.
