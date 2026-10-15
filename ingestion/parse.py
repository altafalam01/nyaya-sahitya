"""CLI: python -m ingestion.parse [--only ID ...] [--ocr auto|force|off] [--check-ocr]

Reads every document in data/raw that has a .meta.json, verifies the PDF still matches the
SHA-256 recorded there, extracts and cleans the text, and writes one JSON file per document to
data/processed/<doc_id>.json plus data/processed/parse_report.json.

Metadata fields the .meta.json does not provide (court and date for Acts, bench for everything
at present) are left as "" - nothing is guessed from the text.
"""
import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings, setup_logging

from .ocr import OcrStatus, check_tesseract
from .pdf_extract import BLANK, OCR, OCR_SKIPPED, extract_pages
from .textclean import clean_document

log = logging.getLogger(__name__)
PARSER_VERSION = "1.0"

# provenance copied verbatim when the metadata has it
CARRY_OVER = ("act", "legal_status", "usage_note", "in_force_from", "repealed_on", "content_note")
SOURCE_KEYS = ("file_url", "licence_note", "date_fetched", "method", "origin", "mirror_of", "bucket_url",
               "sha256", "filename", "size_bytes", "url_note")


def iso_date(value: str) -> str:
    """DD-MM-YYYY (as the SC metadata writes it) -> YYYY-MM-DD; anything else stays empty."""
    m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", (value or "").strip())
    if not m:
        return ""
    d, mo, y = m.groups()
    try:
        datetime(int(y), int(mo), int(d))
    except ValueError:
        return ""
    return f"{y}-{mo}-{d}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_documents(raw_dir: Path) -> list[tuple[Path, dict]]:
    out = []
    for meta_path in sorted(raw_dir.glob("*/*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pdf = meta_path.with_name(meta_path.name[: -len(".meta.json")])
        if pdf.suffix.lower() == ".pdf" and pdf.exists():
            out.append((pdf, meta))
    return out


def build_record(meta: dict, text: str, extraction: dict) -> dict:
    cm = meta.get("case_metadata") or {}
    record = {
        "doc_id": meta["source_id"],
        "title": cm.get("title") or meta.get("name", ""),
        "type": meta.get("doc_type", ""),
        "court": cm.get("court", ""),
        "date": iso_date(cm.get("decision_date", "")),
        "bench": cm.get("bench", ""),  # not in our metadata: left empty, never guessed
        "citation": cm.get("citation", ""),
        "source_url": meta.get("file_url") or meta.get("url") or "",
    }
    for key in CARRY_OVER:
        record[key] = meta.get(key, "")
    record["source"] = {k: meta[k] for k in SOURCE_KEYS if k in meta}
    record["extraction"] = extraction
    record["text"] = text
    return record


def parse_document(pdf: Path, meta: dict, ocr_mode: str, ocr: Optional[OcrStatus]) -> tuple[dict, list[str]]:
    """Return (record, warnings). Raises ValueError if the file no longer matches its metadata."""
    digest = sha256_of(pdf)
    if meta.get("sha256") and digest != meta["sha256"]:
        raise ValueError(f"{pdf.name}: SHA-256 {digest} does not match metadata {meta['sha256']}")
    pages = extract_pages(pdf, ocr_mode, ocr)
    text, stats = clean_document(pages, meta.get("doc_type", ""))
    by_source = Counter(p.source for p in pages)
    skipped = [p.number for p in pages if p.source == OCR_SKIPPED]
    weak = [p.number for p in pages if p.weak]
    warnings = []
    if skipped:
        warnings.append(f"{len(skipped)} page(s) need OCR but Tesseract is unavailable; text missing for pages {skipped[:20]}")
    if weak:
        warnings.append(f"{len(weak)} page(s) have a poor text layer and were not OCR'd: {weak[:20]}")
    if stats.unrecoverable_chars:
        warnings.append(f"{stats.unrecoverable_chars} unreadable character(s) removed")
    if not text.strip():
        warnings.append("no text extracted")
    extraction = {
        "parser_version": PARSER_VERSION, "parsed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pages": len(pages), "pages_text_layer": by_source["text"], "pages_ocr": by_source[OCR],
        "pages_blank": by_source[BLANK], "pages_ocr_skipped": skipped, "pages_weak_text_layer": weak,
        "ocr_language": ocr.language_arg() if ocr and ocr.available and by_source[OCR] else None,
        "removed": {k: v for k, v in asdict(stats).items() if k != "header_samples"},
        "header_footer_samples": stats.header_samples, "characters": len(text), "warnings": warnings,
    }
    return build_record(meta, text, extraction), warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", metavar="ID", help="document ids to parse (default: all)")
    ap.add_argument("--ocr", choices=("auto", "force", "off"), default="auto",
                    help="auto: OCR only pages with no usable text layer (default)")
    ap.add_argument("--check-ocr", action="store_true", help="report whether Tesseract is usable, then exit")
    args = ap.parse_args(argv)
    setup_logging()

    status = check_tesseract()
    if args.check_ocr:
        print(status.message)
        print("OCR READY" if status.available else "OCR NOT READY")
        return 0 if status.available else 1

    docs = find_documents(settings.raw_data_dir)
    if args.only:
        docs = [(p, m) for p, m in docs if m["source_id"] in args.only]
    out_dir = settings.processed_data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report, failed = [], False
    for pdf, meta in docs:
        try:
            record, warnings = parse_document(pdf, meta, args.ocr, status)
        except (ValueError, OSError, RuntimeError) as e:
            print(f"[failed] {meta['source_id']}: {e}")
            report.append({"doc_id": meta["source_id"], "status": "failed", "error": str(e)})
            failed = True
            continue
        (out_dir / f"{record['doc_id']}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        ex = record["extraction"]
        print(f"[parsed] {record['doc_id']}: {ex['pages']} pages ({ex['pages_ocr']} OCR), {ex['characters']} chars"
              + (f" WARN {warnings}" if warnings else ""))
        report.append({"doc_id": record["doc_id"], "status": "parsed", "type": record["type"],
                       "legal_status": record["legal_status"], **{k: ex[k] for k in
                       ("pages", "pages_text_layer", "pages_ocr", "pages_blank", "characters", "removed", "warnings")}})
    (out_dir / "parse_report.json").write_text(
        json.dumps({"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "ocr": status.message, "documents": report}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
