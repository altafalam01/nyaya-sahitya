"""LLM provider tests: Gemini and Ollama request/response handling with mock transports. No network."""
import json

import httpx
import pytest

import generation.llm as llm_mod
from generation.llm import AnthropicLLM, GeminiLLM, OllamaLLM, make_llm


def mock(status=200, body=None, seen=None):
    def handler(req: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(req)
        return httpx.Response(status, json=body if body is not None else {})
    return httpx.MockTransport(handler)


def test_gemini_request_and_parse():
    seen = []
    body = {"candidates": [{"content": {"parts": [{"text": "Hello "}, {"text": "world"}]}}]}
    out = GeminiLLM(model="m1", api_key="k", transport=mock(200, body, seen)).complete("SYS", "USER")
    assert out == "Hello world"
    req = seen[0]
    sent = json.loads(req.content)
    assert req.url.path.endswith("/models/m1:generateContent") and req.headers["x-goog-api-key"] == "k"
    assert "k" not in str(req.url.query)  # key goes in a header, never the URL
    assert sent["system_instruction"]["parts"][0]["text"] == "SYS"
    assert sent["contents"][0]["parts"][0]["text"] == "USER" and sent["generationConfig"]["temperature"] == 0


def test_gemini_missing_key_and_errors():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiLLM(api_key="", transport=mock()).complete("s", "u")
    with pytest.raises(RuntimeError, match="429"):
        GeminiLLM(api_key="k", transport=mock(429, {"error": "quota"})).complete("s", "u")


def test_gemini_blocked_response_is_empty_not_crash():
    assert GeminiLLM(api_key="k", transport=mock(200, {"candidates": []})).complete("s", "u") == ""
    assert GeminiLLM(api_key="k", transport=mock(200, {})).complete("s", "u") == ""


def test_ollama_request_and_parse():
    seen = []
    out = OllamaLLM(model="q", url="http://h:1/", transport=mock(200, {"message": {"content": "hi"}}, seen)).complete("S", "U")
    sent = json.loads(seen[0].content)
    assert out == "hi" and str(seen[0].url) == "http://h:1/api/chat"
    assert sent["model"] == "q" and sent["stream"] is False
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_ollama_unreachable_is_runtime_error():
    def boom(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(RuntimeError, match="unreachable"):
        OllamaLLM(transport=httpx.MockTransport(boom)).complete("s", "u")


def test_make_llm_selects_provider_and_rejects_unknown():
    assert isinstance(make_llm("gemini"), GeminiLLM)
    assert isinstance(make_llm("OLLAMA"), OllamaLLM)
    assert isinstance(make_llm("anthropic"), AnthropicLLM)
    with pytest.raises(ValueError):
        make_llm("nope")


def test_llm_configured_reflects_provider(monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr(llm_mod, "settings", replace(llm_mod.settings, llm_provider="gemini", gemini_api_key=""))
    assert llm_mod.llm_configured() is False
    monkeypatch.setattr(llm_mod, "settings", replace(llm_mod.settings, gemini_api_key="x"))
    assert llm_mod.llm_configured() is True
    monkeypatch.setattr(llm_mod, "settings", replace(llm_mod.settings, llm_provider="ollama"))
    assert llm_mod.llm_configured() is True
