"""Chroma wrapper: persistent cosine collection, scalar metadata, filter building."""
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import chromadb

from config import settings

META_KEYS = ("doc_id", "chunk_type", "doc_type", "act", "legal_status", "court", "unit_label", "unit_number",
             "locator", "cite", "doc_title", "section_type", "content_hash")


def year_of(record: dict) -> int:
    """Year of a chunk: judgment date, else the year in the act id (BNS_2023 -> 2023); 0 if unknown."""
    m = re.match(r"(\d{4})", record.get("date") or "") or re.search(r"(\d{4})\s*$", record.get("act") or "")
    return int(m.group(1)) if m else 0


def to_metadata(record: dict) -> dict[str, Any]:
    meta: dict[str, Any] = {k: (record.get(k) or "") for k in META_KEYS}
    meta["year"] = year_of(record)
    return meta


def build_where(act: Optional[str] = None, court: Optional[str] = None, year: Optional[int] = None,
                year_from: Optional[int] = None, year_to: Optional[int] = None, doc_type: Optional[str] = None,
                legal_status: Optional[str] = None, doc_id: Optional[str] = None,
                include_index: bool = False) -> Optional[dict]:
    conds: list[dict] = []
    for key, val in (("act", act), ("court", court), ("doc_type", doc_type), ("legal_status", legal_status),
                     ("doc_id", doc_id)):
        if val:
            conds.append({key: val})
    if year is not None:
        conds.append({"year": year})
    if year_from is not None:
        conds.append({"year": {"$gte": year_from}})
    if year_to is not None:
        conds.append({"year": {"$lte": year_to}})
    if not include_index:
        conds.append({"chunk_type": {"$nin": ["index", "front_matter"]}})
    return None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}


class Store:
    def __init__(self, name: str, path: Optional[Path] = None) -> None:
        self.client = chromadb.PersistentClient(path=str(path or settings.index_dir))
        self.name = name
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.col.count()

    def existing(self, doc_id: str) -> dict[str, str]:
        """{chunk_id: content_hash} currently indexed for one document."""
        got = self.col.get(where={"doc_id": doc_id}, include=["metadatas"])
        return {i: (m or {}).get("content_hash", "") for i, m in zip(got["ids"], got["metadatas"])}

    def doc_ids(self) -> set[str]:
        got = self.col.get(include=["metadatas"])
        return {(m or {}).get("doc_id", "") for m in got["metadatas"]}

    def upsert(self, records: Sequence[dict], vectors: Sequence[Sequence[float]]) -> None:
        if records:
            self.col.upsert(ids=[r["chunk_id"] for r in records], embeddings=[list(v) for v in vectors],
                            documents=[r["text"] for r in records], metadatas=[to_metadata(r) for r in records])

    def delete(self, ids: Sequence[str]) -> None:
        for i in range(0, len(ids), 500):
            self.col.delete(ids=list(ids[i:i + 500]))

    def delete_doc(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})

    def query(self, vector: Sequence[float], k: int, where: Optional[dict] = None) -> list[dict]:
        if self.col.count() == 0:
            return []
        res = self.col.query(query_embeddings=[list(vector)], n_results=k, where=where,
                             include=["documents", "metadatas", "distances"])
        return [{"chunk_id": i, "text": d, "meta": m, "score": 1.0 - dist}
                for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]
