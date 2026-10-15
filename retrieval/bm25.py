"""Small Okapi BM25 over data/chunks, persisted next to the Chroma index and rebuilt only when the chunks change."""
import hashlib
import json
import math
import pickle
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from config import settings

STOPWORDS = frozenset("a an and are as at be by for from has in is it of on or that the to was were which with".split())
TOKEN_RE = re.compile(r"[\wऀ-ॣ०-ॿ]+", re.UNICODE)  # \w alone splits Devanagari at vowel signs; the danda is excluded
INDEX_VERSION = 2  # bump whenever tokenize() or searchable() changes, so a stale pickle is rebuilt


def tokenize(text: str) -> list[str]:
    """Lowercased NFKC word pieces; numbers kept ('302', '498a'); Devanagari digits folded to ASCII."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(str(unicodedata.digit(c)) if c.isdigit() and unicodedata.digit(c, None) is not None else c for c in text)
    return [t for t in TOKEN_RE.findall(text) if t not in STOPWORDS]


def searchable(record: dict) -> str:
    """Heading and locator are indexed with the text, so 'Section 318 cheating' finds the right chunk."""
    return f"{record.get('locator', '')} {record.get('heading', '')} {record.get('doc_title', '')} {record['text']}"


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.meta: list[dict] = []
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.avgdl = 0.0

    @classmethod
    def build(cls, records: list[dict], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        ix = cls(k1, b)
        for n, r in enumerate(records):
            toks = tokenize(searchable(r))
            ix.ids.append(r["chunk_id"])
            ix.meta.append({k: r.get(k, "") for k in ("doc_id", "chunk_type", "doc_type", "act", "legal_status", "court",
                                                      "date", "unit_label", "unit_number", "locator", "cite", "doc_title")})
            ix.lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                ix.postings.setdefault(term, []).append((n, tf))
        ix.avgdl = sum(ix.lengths) / len(ix.lengths) if ix.lengths else 0.0
        return ix

    def search(self, query: str, k: int = 50, accept: Optional[Callable[[dict], bool]] = None) -> list[tuple[str, float]]:
        """[(chunk_id, score)] best first; `accept` filters on chunk metadata before ranking."""
        n_docs = len(self.ids)
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for n, tf in plist:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[n] / (self.avgdl or 1))
                scores[n] = scores.get(n, 0.0) + idf * tf * (self.k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        out: list[tuple[str, float]] = []
        for n, sc in ranked:
            if accept is None or accept(self.meta[n]):
                out.append((self.ids[n], sc))
                if len(out) == k:
                    break
        return out


def fingerprint(records: list[dict]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(f"{r['chunk_id']}:{r.get('content_hash', '')}:{len(r['text'])}\n".encode())
    return h.hexdigest()


def load_records(chunks_dir: Path) -> list[dict]:
    recs: list[dict] = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        recs.extend(json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
    return recs


def load_or_build(chunks_dir: Optional[Path] = None, cache: Optional[Path] = None) -> BM25Index:
    chunks_dir = chunks_dir or settings.chunks_data_dir
    cache = cache or settings.index_dir / "bm25.pkl"
    recs = load_records(chunks_dir)
    fp = f"{INDEX_VERSION}:{fingerprint(recs)}"
    if cache.exists():
        try:
            with cache.open("rb") as f:
                saved = pickle.load(f)
            if saved.get("fingerprint") == fp:
                return saved["index"]
        except Exception:  # unreadable or from an older class layout: rebuild
            pass
    ix = BM25Index.build(recs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump({"fingerprint": fp, "index": ix}, f)
    return ix
