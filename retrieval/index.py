"""CLI: python -m retrieval.index [--only ID ...] [--rebuild] [--prune] [--batch-size N]

Embeds data/chunks/*.jsonl into the Chroma collection. Safe to re-run: per document, the chunk ids and
hashes of the embedded text are compared with what is already indexed, so only new or changed chunks are
embedded and chunks that disappeared are deleted. --rebuild empties the collection first.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from config import settings, setup_logging

from .embedder import Embedder, passage_text
from .store import Store

log = logging.getLogger(__name__)


def load_chunks(path: Path) -> list[dict]:
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in recs:  # the hash covers exactly what is embedded, so a changed breadcrumb also re-embeds
        r["content_hash"] = hashlib.sha256(passage_text(r).encode("utf-8")).hexdigest()
    return recs


def run(chunks_dir: Path, store: Store, embedder: Embedder, only: Optional[list[str]] = None,
        rebuild: bool = False, prune: bool = False) -> list[dict]:
    if rebuild:
        store.reset()
    report: list[dict] = []
    seen: set[str] = set()
    for path in sorted(chunks_dir.glob("*.jsonl")):
        doc_id = path.stem
        seen.add(doc_id)
        if only and doc_id not in only:
            continue
        try:
            recs = load_chunks(path)
            have = store.existing(doc_id)
            want = {r["chunk_id"]: r["content_hash"] for r in recs}
            todo = [r for r in recs if have.get(r["chunk_id"]) != r["content_hash"]]
            gone = [i for i in have if i not in want]
            if todo:
                batch = embedder.batch_size
                for i in range(0, len(todo), batch):
                    part = todo[i:i + batch]
                    store.upsert(part, embedder.embed_passages([passage_text(r) for r in part]))
            if gone:
                store.delete(gone)
            status = "up_to_date" if not todo and not gone else ("indexed" if not have else "updated")
            report.append({"doc_id": doc_id, "status": status, "chunks": len(recs), "embedded": len(todo),
                           "deleted": len(gone)})
        except Exception as e:  # one bad document must not stop the others
            log.exception("indexing %s failed", doc_id)
            report.append({"doc_id": doc_id, "status": "failed", "error": str(e)})
    for d in sorted(store.doc_ids() - seen - {""}):
        if prune:
            store.delete_doc(d)
            report.append({"doc_id": d, "status": "pruned"})
        else:
            report.append({"doc_id": d, "status": "orphan", "note": "no chunk file; use --prune to delete"})
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", metavar="ID")
    ap.add_argument("--rebuild", action="store_true", help="empty the collection and re-embed everything")
    ap.add_argument("--prune", action="store_true", help="delete indexed documents that have no chunk file")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args(argv)
    setup_logging()
    embedder = Embedder(batch_size=args.batch_size)
    store = Store(embedder.collection_name)
    report = run(settings.chunks_data_dir, store, embedder, args.only, args.rebuild, args.prune)
    failed = False
    for r in report:
        if r["status"] == "failed":
            failed = True
            extra = r["error"]
        elif r["status"] in ("indexed", "updated", "up_to_date"):
            extra = f"{r['chunks']} chunks (+{r['embedded']} embedded, -{r['deleted']} deleted)"
        else:
            extra = r.get("note", "")
        print(f"[{r['status']}] {r['doc_id']}: {extra}")
    print(f"collection {store.name}: {store.count()} vectors")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
