"""Thin typed client for the backend /ask endpoint. All network and error handling lives here, not in the UI."""
from dataclasses import dataclass, field
from typing import Optional

import httpx


class ApiError(Exception):
    """Carries a stable `kind` the UI turns into a translated message."""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind  # "unreachable" | "unconfigured" | "invalid" | "server"


@dataclass(frozen=True)
class Source:
    cite: str
    locator: str
    legal_status: str
    text: str


@dataclass(frozen=True)
class Reply:
    answer: str
    lang: str
    grounded: bool
    citations: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    disclaimer: str = ""


def ask(base_url: str, question: str, *, lang: Optional[str] = None, k: Optional[int] = None,
        timeout: float = 90.0, transport: Optional[httpx.BaseTransport] = None) -> Reply:
    payload = {"question": question, **{n: v for n, v in (("lang", lang), ("k", k)) if v is not None}}
    try:
        with httpx.Client(base_url=base_url, timeout=timeout, transport=transport) as c:
            r = c.post("/ask", json=payload)
    except httpx.HTTPError as e:
        raise ApiError("unreachable", str(e)) from e
    if r.status_code == 503:
        raise ApiError("unconfigured")
    if r.status_code == 422:
        raise ApiError("invalid")
    if r.status_code != 200:
        raise ApiError("server", str(r.status_code))
    d = r.json()
    return Reply(d["answer"], d["lang"], d["grounded"], d["citations"],
                 [Source(s["cite"], s["locator"], s["legal_status"], s["text"]) for s in d["sources"]], d["disclaimer"])
