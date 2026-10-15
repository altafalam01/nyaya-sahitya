"""Hybrid search: dense + BM25 -> reciprocal rank fusion -> cross-encoder rerank, with reference-aware boosting.

    hybrid_search("Section 302 IPC", k=5)                      # full pipeline
    hybrid_search("cheating", mode="dense")                    # pure vector (the Step 5 behaviour)
"""
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from config import settings

from .bm25 import BM25Index, load_or_build
from .embedder import Embedder
from .fusion import rrf
from .mapping import BNS_DOC, IPC_DOC, load_map, section_headings
from .query import ProcessedQuery, Ref, preprocess
from .rerank import Reranker
from .search import Hit
from .store import Store, build_where, year_of

MODES = ("dense", "bm25", "hybrid")
ACT_PREFIX = {"IPC": "IPC", "BNS": "BNS", "BNSS": "BNSS", "BSA": "BSA", "CrPC": "CRPC"}


@lru_cache(maxsize=1)
def _headings(chunks_dir: str) -> dict[str, dict[str, str]]:
    p = Path(chunks_dir)
    return {"BNS": section_headings(p, BNS_DOC), "IPC": section_headings(p, IPC_DOC)}


def accept_fn(act=None, court=None, year=None, year_from=None, year_to=None, doc_type=None, legal_status=None,
              doc_id=None, include_index=False) -> Callable[[dict], bool]:
    """Same filter semantics as store.build_where, applied to BM25 metadata."""
    def ok(m: dict) -> bool:
        if act and m.get("act") != act or court and m.get("court") != court or doc_type and m.get("doc_type") != doc_type:
            return False
        if legal_status and m.get("legal_status") != legal_status or doc_id and m.get("doc_id") != doc_id:
            return False
        y = year_of(m)
        if year is not None and y != year or year_from is not None and y < year_from or year_to is not None and y > year_to:
            return False
        return include_index or m.get("chunk_type") not in ("index", "front_matter")
    return ok


def exact_matches(ix: BM25Index, ref: Ref) -> list[str]:
    """Chunk ids of the real section/article a reference points at (acts without a named act match any)."""
    want_label = "article" if ref.kind == "article" else "section"
    prefix = ACT_PREFIX.get(ref.act or "", "")
    out = []
    for cid, m in zip(ix.ids, ix.meta):
        if m.get("chunk_type") != "section" or str(m.get("unit_number", "")).upper() != ref.number:
            continue
        if want_label not in str(m.get("unit_label", "")).lower():
            continue
        if ref.act == "CONST" and m.get("doc_type") != "constitution":
            continue
        if prefix and not str(m.get("act", "")).upper().startswith(prefix):
            continue
        out.append(cid)
    return out


def _to_hit(cid: str, row: dict, score: float, detail: dict) -> Hit:
    m = row["meta"]
    return Hit(cid, score, m.get("cite", ""), m.get("locator", ""), m.get("doc_id", ""), m.get("legal_status", ""),
               m.get("court", ""), m.get("year", 0), m.get("chunk_type", ""), row["text"], detail)


def hybrid_search(query: str, k: int = 5, *, mode: str = "hybrid", rerank: bool = True,
                  embedder: Optional[Embedder] = None, store: Optional[Store] = None, bm25: Optional[BM25Index] = None,
                  reranker: Optional[Reranker] = None, section_map: Optional[dict] = None, **filters) -> list[Hit]:
    """filters: act, court, year, year_from, year_to, doc_type, legal_status, doc_id, include_index."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    embedder = embedder or Embedder()
    store = store or Store(embedder.collection_name)
    pq: ProcessedQuery = preprocess(query, section_map if section_map is not None else load_map(),
                                    _headings(str(settings.chunks_data_dir))) if mode != "dense" else \
        ProcessedQuery(query, query)
    ranks: dict[str, dict] = {}

    dense_ids: list[str] = []
    if mode in ("dense", "hybrid"):
        rows = store.query(embedder.embed_query(pq.search_text), settings.dense_candidates if mode == "hybrid" else k,
                           build_where(**filters))
        dense_ids = [r["chunk_id"] for r in rows]
        for n, cid in enumerate(dense_ids, 1):
            ranks.setdefault(cid, {})["dense"] = n
    bm25_ids: list[str] = []
    ix = bm25
    if mode in ("bm25", "hybrid"):
        ix = ix or load_or_build()
        hits = ix.search(pq.search_text, settings.bm25_candidates if mode == "hybrid" else k, accept_fn(**filters))
        bm25_ids = [cid for cid, _ in hits]
        for n, cid in enumerate(bm25_ids, 1):
            ranks.setdefault(cid, {})["bm25"] = n

    if mode == "dense":
        final = dense_ids[:k]
        scores = {r["chunk_id"]: r["score"] for r in rows}
    elif mode == "bm25":
        final, scores = bm25_ids[:k], dict(hits)
    else:
        accept = accept_fn(**filters)
        by_id = dict(zip(ix.ids, ix.meta))
        pinned: list[str] = []  # the sections the user literally named, then their mapped counterparts
        for ref in pq.refs + pq.mapped:
            if ref.kind == "section" and ref.act is None:
                continue  # "Section 420" with no act: leave it to the retrievers, do not guess the act
            for cid in exact_matches(ix, ref):
                if accept(by_id[cid]) and cid not in pinned:
                    pinned.append(cid)
                    ranks.setdefault(cid, {})["reference"] = len(pinned)
        fused = rrf([dense_ids, bm25_ids, pinned], weights=[1.0, 1.0, 2.0] if pinned else [1.0, 1.0, 0.0])
        pool = [cid for cid, _ in fused[:settings.rerank_top]]
        scores = dict(fused)
        order = pool
        if rerank and pool:
            rows_by_id = store.get(pool)
            pool = [c for c in pool if c in rows_by_id]
            reranker = reranker or Reranker()
            rr = reranker.rerank(query, [(c, rows_by_id[c]["text"]) for c in pool])
            order = [c for c, _ in rr]
            scores = {c: s for c, s in rr}
            for n, c in enumerate(order, 1):
                ranks.setdefault(c, {})["rerank"] = n
        # named sections stay on top: the user asked for them by number, a text model should not demote them
        final = (pinned[:k] + [c for c in order if c not in pinned])[:k]

    rows_by_id = store.get(final)
    out = []
    for cid in final:
        if cid in rows_by_id:
            detail = dict(ranks.get(cid, {}))
            if pq.notes:
                detail["notes"] = pq.notes
            out.append(_to_hit(cid, rows_by_id[cid], float(scores.get(cid, 0.0)), detail))
    return out
