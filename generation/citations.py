"""Citation checking: every kept sentence must cite, and every citation must match a retrieved source."""
import re
from dataclasses import dataclass, field
from typing import Iterable

_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")
_MASK = "\x00"


def norm(cite: str) -> str:
    return " ".join(cite.lower().split())


@dataclass
class Checked:
    text: str                                            # answer with unsupported sentences removed
    citations: list[str] = field(default_factory=list)   # distinct valid cites used, in order
    dropped: list[str] = field(default_factory=list)     # removed sentences (uncited or badly cited)


def _split_sentences(text: str) -> list[str]:
    # hide full stops inside [..] (e.g. "State v. X, Para 5") so they do not split a sentence
    masked = _BRACKET.sub(lambda m: "[" + m.group(1).replace(".", _MASK) + "]", text)
    return [s.replace(_MASK, ".").strip() for s in _SPLIT.split(masked) if s.strip()]


def check_citations(answer: str, valid_cites: Iterable[str]) -> Checked:
    valid = {norm(c): c for c in valid_cites}
    kept: list[str] = []
    used: list[str] = []
    dropped: list[str] = []
    for sent in _split_sentences(answer):
        cites = [c.strip() for c in _BRACKET.findall(sent)]
        if not cites or any(norm(c) not in valid for c in cites):
            dropped.append(sent)
            continue
        kept.append(sent)
        for c in cites:
            if valid[norm(c)] not in used:
                used.append(valid[norm(c)])
    return Checked(" ".join(kept), used, dropped)
