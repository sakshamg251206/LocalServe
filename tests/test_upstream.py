import httpx
import pytest

from localserve.gateway.upstream import HttpUpstream


async def test_truncated_upstream_is_not_success() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"a"}}]}\n\n')

    upstream = HttpUpstream("http://fake", 1)
    await upstream.aclose()
    upstream._client = httpx.AsyncClient(base_url="http://fake", transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(httpx.RemoteProtocolError, match="DONE"):
            _ = [event async for event in upstream.stream_chat({})]
    finally:
        await upstream.aclose()


@pytest.mark.parametrize("tokens", [64, 128])
async def test_fixed_length_client_checks_usage_even_on_token_event(tokens: int) -> None:
    import json

    from localserve.loadgen.client import send_chat

    event = {
        "choices": [{"delta": {"content": "a"}}],
        "usage": {"completion_tokens": tokens, "prompt_tokens": 150},
    }

    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

    async with httpx.AsyncClient(base_url="http://fake", transport=httpx.MockTransport(respond)) as client:
        result = await send_chat(client, "/v1/chat/completions", {"ignore_eos": True, "max_tokens": 128}, 1)
    assert result.completion_tokens == tokens
    assert result.ok == (tokens == 128)
