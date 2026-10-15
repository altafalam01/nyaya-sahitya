"""LLM access behind a tiny interface so tests can inject a fake. Providers: anthropic, gemini (free tier), ollama (local)."""
from typing import Optional, Protocol

import httpx

from config import settings


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, max_tokens: Optional[int] = None):
        self.model = model or settings.answer_model
        self.max_tokens = max_tokens or settings.answer_max_tokens
        self._key = api_key or settings.anthropic_api_key
        self._client = None

    def complete(self, system: str, user: str) -> str:
        if self._client is None:
            if not self._key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (put it in .env)")
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._key)
        msg = self._client.messages.create(model=self.model, max_tokens=self.max_tokens, temperature=0,
                                           system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class GeminiLLM:
    """Google Gemini REST API (has a free tier; get a key at https://aistudio.google.com/apikey)."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, max_tokens: Optional[int] = None,
                 transport: Optional[httpx.BaseTransport] = None):
        self.model = model or settings.gemini_model
        self.max_tokens = max_tokens or settings.answer_max_tokens
        self._key = api_key or settings.gemini_api_key
        self._transport = transport

    def complete(self, system: str, user: str) -> str:
        if not self._key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
        body = {"system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": self.max_tokens}}
        try:
            with httpx.Client(timeout=90, transport=self._transport) as c:
                r = c.post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                           headers={"x-goog-api-key": self._key}, json=body)
        except httpx.HTTPError as e:
            raise RuntimeError(f"Gemini unreachable: {e}") from e
        if r.status_code != 200:
            raise RuntimeError(f"Gemini error {r.status_code}")
        parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)  # empty (e.g. blocked) -> caller treats as "not found"


class OllamaLLM:
    """Local model through Ollama (free, offline). Install from https://ollama.com, then `ollama pull <model>`."""

    def __init__(self, model: Optional[str] = None, url: Optional[str] = None,
                 transport: Optional[httpx.BaseTransport] = None):
        self.model = model or settings.ollama_model
        self.url = (url or settings.ollama_url).rstrip("/")
        self._transport = transport

    def complete(self, system: str, user: str) -> str:
        body = {"model": self.model, "stream": False, "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            with httpx.Client(timeout=300, transport=self._transport) as c:
                r = c.post(f"{self.url}/api/chat", json=body)
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama unreachable at {self.url}: {e}") from e
        if r.status_code != 200:
            raise RuntimeError(f"Ollama error {r.status_code}")
        return r.json().get("message", {}).get("content", "")


def make_llm(provider: Optional[str] = None) -> LLM:
    provider = (provider or settings.llm_provider).lower()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "gemini":
        return GeminiLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"LLM_PROVIDER must be anthropic, gemini or ollama (got {provider!r})")


def llm_configured() -> bool:
    return {"anthropic": bool(settings.anthropic_api_key), "gemini": bool(settings.gemini_api_key),
            "ollama": True}.get(settings.llm_provider.lower(), False)
