"""Dense search over the index. CLI: python -m retrieval.search "cheating" --k 5 [--court C --year Y ...]"""
import argparse
from dataclasses import dataclass
from typing import Optional

from config import setup_logging

from .embedder import Embedder
from .store import Store, build_where


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    cite: str
    locator: str
    doc_id: str
    legal_status: str
    court: str
    year: int
    chunk_type: str
    text: str


def search(query: str, k: int = 5, *, embedder: Optional[Embedder] = None, store: Optional[Store] = None,
           **filters) -> list[Hit]:
    """filters: act, court, year, year_from, year_to, doc_type, legal_status, doc_id, include_index."""
    embedder = embedder or Embedder()
    store = store or Store(embedder.collection_name)
    rows = store.query(embedder.embed_query(query), k, build_where(**filters))
    return [Hit(r["chunk_id"], r["score"], m["cite"], m["locator"], m["doc_id"], m["legal_status"], m["court"],
                m["year"], m["chunk_type"], r["text"]) for r in rows for m in [r["meta"]]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--act")
    ap.add_argument("--court")
    ap.add_argument("--year", type=int)
    ap.add_argument("--year-from", type=int)
    ap.add_argument("--year-to", type=int)
    ap.add_argument("--doc-type")
    ap.add_argument("--legal-status")
    ap.add_argument("--include-index", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()
    hits = search(a.query, a.k, act=a.act, court=a.court, year=a.year, year_from=a.year_from, year_to=a.year_to,
                  doc_type=a.doc_type, legal_status=a.legal_status, include_index=a.include_index)
    if not hits:
        print("no results")
    for n, h in enumerate(hits, 1):
        flag = f" [{h.legal_status.upper()}]" if h.legal_status and h.legal_status != "in_force" else ""
        print(f"{n}. {h.score:.3f} {h.cite}{flag}\n   {' '.join(h.text.split())[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
