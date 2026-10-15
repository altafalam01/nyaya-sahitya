"""LLM access behind a tiny interface so tests can inject a fake."""
from typing import Optional, Protocol

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
