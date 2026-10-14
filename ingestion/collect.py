"""CLI: python -m ingestion.collect [--dry-run] [--only ID ...]

--dry-run prints each URL without touching the network, so they can be reviewed first.
Always writes data/raw/registry.json summarising every source and its current status.
"""
import argparse
import json
from datetime import datetime, timezone

from config import settings, setup_logging

from .downloader import MANUAL, Downloader, manual_instructions
from .sources import SOURCES, Source, get_source


def write_registry(raw_dir, sources, outcomes) -> None:
    by_id = {o.source_id: o for o in outcomes}
    rows = []
    for s in sources:
        o = by_id.get(s.id)
        rows.append({
            "id": s.id, "name": s.name, "doc_type": s.doc_type, "url": s.url,
            "file_url": s.file_url, "licence_note": s.licence_note,
            "expected_path": str((raw_dir / s.id / s.filename)),
            "status": o.status if o else "not_run",
        })
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "registry.json").write_text(
        json.dumps({"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sources": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", metavar="ID")
    args = ap.parse_args(argv)
    setup_logging()

    sources: list[Source] = [get_source(i) for i in args.only] if args.only else list(SOURCES)
    if args.dry_run:
        for s in sources:
            print(f"{s.id}\n  page: {s.url}\n  file: {s.file_url or '(none verified)'}\n  note: {s.url_note}")
        return 0

    dl = Downloader(settings.raw_data_dir, settings.user_agent, settings.download_delay_seconds)
    outcomes = []
    for s in sources:
        o = dl.collect(s)
        outcomes.append(o)
        print(f"[{o.status}] {s.id}: {o.detail}")
    write_registry(settings.raw_data_dir, sources, outcomes)
    return 1 if any(o.status == MANUAL for o in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
