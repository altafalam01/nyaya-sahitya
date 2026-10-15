"""CLI: python -m ingestion.chunk [--only ID ...] [--force] [--prune] [--max-tokens N]

Chunks every document in data/processed into data/chunks/<doc_id>.jsonl. Safe to re-run:

* A manifest (data/chunks/manifest.json) records, per document, a fingerprint of the processed
  text and metadata plus the chunker settings. A document whose fingerprint and settings are
  unchanged is skipped; new documents (e.g. the Constitution, BNSS, BSA once they are parsed)
  are picked up automatically.
* When a document did change, chunk ids are compared with the previous run and reported as
  unchanged / added / removed, so a later embedding step can re-embed only the new ids.
* --prune deletes chunk files for documents that are no longer in data/processed.
"""
import argparse
import hashlib
import json
import logging
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings, setup_logging

from .chunker import CHUNKER_VERSION, ChunkParams, chunk_document

log = logging.getLogger(__name__)
SKIP_FILES = {"parse_report.json"}
FINGERPRINT_KEYS = ("doc_id", "title", "type", "court", "date", "bench", "citation", "source_url", "act",
                    "legal_status", "usage_note", "text")


def fingerprint(doc: dict) -> str:
    """Hash of what chunking depends on; ignores parse timestamps so a re-parse of the same text is a no-op."""
    blob = json.dumps({k: doc.get(k) for k in FINGERPRINT_KEYS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("documents"), dict):
                return data
        except (OSError, json.JSONDecodeError):
            log.warning("manifest unreadable, rebuilding every document")
    return {"documents": {}}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_chunk_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                ids.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def find_processed(processed_dir: Path) -> list[Path]:
    return sorted(p for p in processed_dir.glob("*.json") if p.name not in SKIP_FILES)


def run(processed_dir: Path, chunks_dir: Path, params: ChunkParams, only: Optional[list[str]] = None,
        force: bool = False, prune: bool = False) -> dict:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = chunks_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    docs = manifest["documents"]
    cfg = params.fingerprint()
    report: list[dict] = []
    seen: set[str] = set()

    for path in find_processed(processed_dir):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc_id = doc["doc_id"]
        except (OSError, json.JSONDecodeError, KeyError) as e:
            report.append({"doc_id": path.stem, "status": "failed", "error": f"unreadable processed file: {e}"})
            continue
        seen.add(doc_id)
        if only and doc_id not in only:
            continue
        out = chunks_dir / f"{doc_id}.jsonl"
        fp = fingerprint(doc)
        prev = docs.get(doc_id)
        if not force and prev and prev.get("fingerprint") == fp and prev.get("config") == cfg and out.exists():
            report.append({"doc_id": doc_id, "status": "up_to_date", "chunks": prev["chunks"], "structure": prev.get("structure")})
            continue
        try:
            records, info = chunk_document(doc, params)
        except Exception as e:  # one bad document must not stop the others
            log.exception("chunking %s failed", doc_id)
            report.append({"doc_id": doc_id, "status": "failed", "error": str(e)})
            continue
        old_ids = read_chunk_ids(out)
        new_ids = {r["chunk_id"] for r in records}
        _atomic_write(out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        by_type: dict[str, int] = {}
        for r in records:
            by_type[r["chunk_type"]] = by_type.get(r["chunk_type"], 0) + 1
        warnings = []
        if info.get("structure") == "no_numbering_found" and doc.get("type") != "judgment":
            warnings.append("no numbered sections/articles detected; split by size only")
        if info.get("missing_count"):
            warnings.append(f"{info['missing_count']} number(s) missing from the sequence (omitted, repealed or not detected)")
        entry = {"fingerprint": fp, "config": cfg, "chunks": len(records), "by_type": by_type,
                 "structure": info, "chunked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "chunker_version": CHUNKER_VERSION}
        docs[doc_id] = entry
        status = "chunked" if not old_ids else "rebuilt"
        report.append({"doc_id": doc_id, "status": status, "chunks": len(records), "by_type": by_type,
                       "structure": info, "unchanged": len(old_ids & new_ids), "added": len(new_ids - old_ids),
                       "removed": len(old_ids - new_ids), "warnings": warnings})

    orphans = sorted(d for d in docs if d not in seen)
    for d in orphans:
        if prune:
            (chunks_dir / f"{d}.jsonl").unlink(missing_ok=True)
            del docs[d]
            report.append({"doc_id": d, "status": "pruned"})
        else:
            report.append({"doc_id": d, "status": "orphan", "note": "no longer in data/processed; use --prune to delete"})
    manifest["version"] = CHUNKER_VERSION
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))
    result = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "documents": report}
    _atomic_write(chunks_dir / "chunk_report.json", json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", metavar="ID", help="document ids to chunk (default: all)")
    ap.add_argument("--force", action="store_true", help="re-chunk even if up to date")
    ap.add_argument("--prune", action="store_true", help="delete chunks of documents no longer in data/processed")
    ap.add_argument("--max-tokens", type=int, default=None, help="largest chunk size in estimated tokens (default 500)")
    args = ap.parse_args(argv)
    setup_logging()
    params = ChunkParams()
    if args.max_tokens:
        params = replace(params, max_tokens=args.max_tokens)
    result = run(settings.processed_data_dir, settings.chunks_data_dir, params, args.only, args.force, args.prune)
    failed = False
    for r in result["documents"]:
        extra = ""
        if r["status"] in ("chunked", "rebuilt"):
            extra = f"{r['chunks']} chunks {r['by_type']} (+{r['added']} new, ={r['unchanged']} same, -{r['removed']} gone)"
        elif r["status"] == "up_to_date":
            extra = f"{r['chunks']} chunks, skipped"
        elif r["status"] == "failed":
            extra, failed = r["error"], True
        print(f"[{r['status']}] {r['doc_id']}: {extra}" + (f" WARN {r['warnings']}" if r.get("warnings") else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
