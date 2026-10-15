"""IPC <-> BNS section map, built from the BNS PDF's own correspondence table (no numbers typed from memory).

CLI: python -m retrieval.mapping   -> writes data/index/section_map.json and prints coverage + spot checks.

The table text in the chunks is layout-messy (columns interleaved, wrapped headings), so a pair is kept only when
it is supported by the corpus: both section numbers exist as real chunks AND either the row is a clean
'BNS | IPC. Heading' row whose headings resemble the real section headings, or the headings alone agree.
Rows that fail are counted as unparsed, never guessed.
"""
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from config import settings, setup_logging

BNS_DOC, IPC_DOC = "bns_2023", "ipc_1860"
MAP_NAME = "section_map.json"
# "123(2) <left heading words> | 456A. <right heading>"  (BNS part, then IPC part after the pipe)
ROW_RE = re.compile(
    r"(?P<bns>\d{1,3}[A-Z]?)(?P<sub>(?:\s?\(\w{1,3}\))*)\.?\s+(?P<lh>[^|\n]{0,160}?)\|\s*(?P<ipc>\d{1,3}[A-Z]?)\s?[.\-–]\s*(?P<rh>[^|\n]{0,200})")
# no pipe on the line: "102. Culpable homicide by causing 301. Culpable homicide by causing ..."
INLINE_RE = re.compile(r"^(?P<bns>\d{1,3}[A-Z]?)\.\s+(?P<lh>.{3,200}?)\s(?P<ipc>\d{1,3}[A-Z]?)\.\s+(?P<rh>[A-Z].{2,200})$")
STOP = {"the", "of", "or", "and", "to", "a", "an", "in", "for", "by", "with", "etc", "on", "any", "is", "be", "as"}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}


def similarity(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / min(len(wa), len(wb)) if wa and wb else 0.0


def section_headings(chunks_dir: Path, doc_id: str) -> dict[str, str]:
    """{section number: heading} for the real section chunks of one document."""
    out: dict[str, str] = {}
    path = chunks_dir / f"{doc_id}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["chunk_type"] == "section" and r["unit_number"]:
            out.setdefault(r["unit_number"], r["heading"])
    return out


def table_text(chunks_dir: Path) -> str:
    recs = [json.loads(l) for l in (chunks_dir / f"{BNS_DOC}.jsonl").read_text(encoding="utf-8").splitlines()]
    idx = [r["text"] for r in recs if r["chunk_type"] == "index"]
    full = "\n\n".join(idx)
    start = full.upper().find("CORRESPONDING SECTION TABLE")
    return full[start:] if start >= 0 else ""


def build_map(chunks_dir: Path) -> dict:
    bns_h, ipc_h = section_headings(chunks_dir, BNS_DOC), section_headings(chunks_dir, IPC_DOC)
    text = table_text(chunks_dir).replace("�", "'")
    ipc_to_bns: dict[str, set[str]] = defaultdict(set)
    rows = kept = 0
    matches = [m for line in text.splitlines() for m in [ROW_RE.search(line) or INLINE_RE.match(line.strip())] if m]
    for m in matches:
        rows += 1
        b, i = m["bns"], m["ipc"]
        if b not in bns_h or i not in ipc_h:
            continue
        # support from headings: the real BNS heading resembles the row's text, or the two real headings agree
        both_sides = min(similarity(bns_h[b], m["lh"]), similarity(ipc_h[i], m["rh"]))  # each side of the row fits its section
        if max(both_sides, similarity(bns_h[b], ipc_h[i])) >= 0.5:
            ipc_to_bns[i].add(b)
            kept += 1
    entries = {i: sorted(bs, key=lambda s: (int(re.match(r"\d+", s).group()), s)) for i, bs in ipc_to_bns.items()}
    bns_to_ipc: dict[str, set[str]] = defaultdict(set)
    for i, bs in entries.items():
        for b in bs:
            bns_to_ipc[b].add(i)
    return {"source": "BNS 2023 'Corresponding Section Table of BNS with Repealed Act' (NCRB PDF), parsed from the indexed text",
            "ipc_to_bns": entries,
            "bns_to_ipc": {b: sorted(v, key=lambda s: (int(re.match(r"\d+", s).group()), s)) for b, v in bns_to_ipc.items()},
            "stats": {"rows_matched": rows, "pairs_kept": kept, "ipc_sections_mapped": len(entries),
                      "ipc_sections_total": len(ipc_h), "bns_sections_total": len(bns_h)}}


def load_map(path: Optional[Path] = None) -> dict:
    p = path or settings.index_dir / MAP_NAME
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"ipc_to_bns": {}, "bns_to_ipc": {}, "stats": {}}


def main() -> int:
    setup_logging()
    data = build_map(settings.chunks_data_dir)
    out = settings.index_dir / MAP_NAME
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    s = data["stats"]
    print(f"rows matched {s['rows_matched']}, pairs kept {s['pairs_kept']}; "
          f"{s['ipc_sections_mapped']}/{s['ipc_sections_total']} IPC sections mapped -> {out}")
    bns_h = section_headings(settings.chunks_data_dir, BNS_DOC)
    ipc_h = section_headings(settings.chunks_data_dir, IPC_DOC)
    for i in ("302", "304", "376", "378", "379", "392", "415", "420", "498A", "124A"):
        bs = data["ipc_to_bns"].get(i)
        print(f"IPC {i} ({ipc_h.get(i, '?')[:40]}) -> " + (", ".join(f"BNS {b} ({bns_h.get(b, '?')[:40]})" for b in bs) if bs else "not mapped"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
