"""Query pre-processor: finds section/article references and case names, expands IPC<->BNS references."""
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional

from .bm25 import tokenize  # noqa: F401  (digit folding is shared)
import unicodedata

ACTS = {  # surface form -> act key (matched case-insensitively on the folded query)
    "ipc": "IPC", "indian penal code": "IPC", "आईपीसी": "IPC", "भारतीय दंड संहिता": "IPC", "भा.द.सं": "IPC",
    "bns": "BNS", "bharatiya nyaya sanhita": "BNS", "बीएनएस": "BNS", "भारतीय न्याय संहिता": "BNS",
    "bnss": "BNSS", "bharatiya nagarik suraksha sanhita": "BNSS", "बीएनएसएस": "BNSS",
    "bsa": "BSA", "bharatiya sakshya adhiniyam": "BSA", "बीएसए": "BSA",
    "crpc": "CrPC", "cr.p.c": "CrPC", "code of criminal procedure": "CrPC",
    "constitution": "CONST", "संविधान": "CONST",
}
_ACT_ALT = "|".join(sorted((re.escape(a) for a in ACTS), key=len, reverse=True))
_NUM = r"(\d{1,3}[A-Za-z]{0,2})"
_SEC = r"(?:sections?|secs?\.?|s\.|धारा)"
SEC_THEN_ACT = re.compile(rf"(?<!\w){_SEC}\s*{_NUM}(?:\s*(?:\(\w{{1,3}}\))*)\s*(?:of\s+(?:the\s+)?|,\s*)?(?:({_ACT_ALT})(?![A-Za-z]))?", re.I)
ACT_THEN_SEC = re.compile(rf"(?<!\w)({_ACT_ALT})\s*(?:{_SEC}\s*)?{_NUM}(?!\d)", re.I)
ARTICLE = re.compile(rf"(?<!\w)(?:articles?|art\.?|अनुच्छेद)\s*{_NUM}", re.I)
_W = r"[A-Z][\w.&'’-]*"
_NAME = rf"{_W}(?:(?:\s+(?:of|the|and|&)){{0,2}}\s+{_W}){{0,5}}"  # "Union of India", "State of Kerala"
CASE = re.compile(rf"\b({_NAME}\s+v(?:s)?\.?\s+{_NAME})")


@dataclass(frozen=True)
class Ref:
    kind: str            # "section" or "article"
    number: str          # "302", "498A"
    act: Optional[str]   # "IPC", "BNS", ... or None when the query does not say


@dataclass
class ProcessedQuery:
    original: str
    search_text: str                       # original plus any expansion; what the retrievers see
    refs: list[Ref] = field(default_factory=list)
    mapped: list[Ref] = field(default_factory=list)   # counterparts found through the section map
    cases: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def fold(text: str) -> str:
    """NFKC and Devanagari digits -> ASCII, so 'धारा ४२०' reads as 'धारा 420'."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(str(unicodedata.digit(c)) if c.isdigit() and unicodedata.digit(c, None) is not None else c for c in text)


def find_refs(query: str) -> list[Ref]:
    q = fold(query)
    refs: list[Ref] = []

    def add(kind: str, number: str, act: Optional[str]) -> None:
        r = Ref(kind, number.upper(), act)
        if r not in refs:
            refs.append(r)

    for m in ACT_THEN_SEC.finditer(q):
        add("section", m.group(2), ACTS[m.group(1).lower()])
    for m in SEC_THEN_ACT.finditer(q):
        act = ACTS[m.group(2).lower()] if m.group(2) else None
        if act is None and any(r.number == m.group(1).upper() for r in refs):
            continue
        add("article" if act == "CONST" else "section", m.group(1), act)
    for m in ARTICLE.finditer(q):
        add("article", m.group(1), "CONST")
    # a bare "Section 420" duplicated by an act-qualified one keeps only the qualified one
    return [r for r in refs if r.act or not any(o.number == r.number and o.act for o in refs)]


def find_cases(query: str) -> list[str]:
    return [m.group(1).strip() for m in CASE.finditer(query)]


def preprocess(query: str, section_map: Mapping, headings: Optional[Mapping[str, Mapping[str, str]]] = None) -> ProcessedQuery:
    """section_map: output of mapping.load_map(); headings: {"BNS": {num: heading}, "IPC": {...}} for expansion text."""
    headings = headings or {}
    pq = ProcessedQuery(original=query, search_text=query, refs=find_refs(query), cases=find_cases(query))
    extra: list[str] = []
    for r in pq.refs:
        if r.kind != "section":
            continue
        if r.act == "IPC":
            targets, other = section_map.get("ipc_to_bns", {}).get(r.number, []), "BNS"
            if targets:
                pq.notes.append(f"IPC Section {r.number} is repealed; the corresponding BNS section(s): {', '.join(targets)}")
        elif r.act == "BNS":
            targets, other = section_map.get("bns_to_ipc", {}).get(r.number, []), "IPC"
            if targets:
                pq.notes.append(f"BNS Section {r.number} replaces IPC Section(s): {', '.join(targets)}")
        else:
            continue
        for t in targets[:4]:
            pq.mapped.append(Ref("section", t, other))
            extra.append(f"{other} Section {t} {headings.get(other, {}).get(t, '')}".strip())
    if extra:
        pq.search_text = f"{query} ({'; '.join(extra)})"
    return pq
