"""Evaluation: recall@5 of dense vs BM25 vs hybrid vs hybrid+rerank on eval/queries.json, printed as a table.

CLI: python -m retrieval.evaluate [--k 5] [--no-rerank]
Gold answers are (doc, unit) pairs for statutes or a whole doc for a judgment; every gold entry is checked
against the chunks first, so a typo in the gold set fails loudly instead of silently lowering a score.
"""
import argparse
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

from config import BASE_DIR, setup_logging

from .search import Hit

QUERIES = BASE_DIR / "eval" / "queries.json"
METHODS = ("dense", "bm25", "hybrid", "hybrid+rerank")


def is_gold(hit: Hit, gold: Sequence[dict]) -> bool:
    unit = hit.locator.split()[-1] if hit.locator.lower().startswith(("section", "article")) else ""
    for g in gold:
        if hit.doc_id == g["doc"] and ("unit" not in g or unit == g["unit"]):
            return True
    return False


def recall_at_k(hits: Sequence[Hit], gold: Sequence[dict], k: int = 5) -> int:
    """1 if any of the top k hits is a gold answer, else 0."""
    return int(any(is_gold(h, gold) for h in hits[:k]))


def check_gold(queries: Sequence[dict], chunks_dir: Path) -> list[str]:
    """Problems with the gold set: a doc or section that does not exist in the chunks."""
    have: dict[str, set[str]] = {}
    for path in chunks_dir.glob("*.jsonl"):
        have[path.stem] = {json.loads(l).get("unit_number", "") for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    problems = []
    for q in queries:
        for g in q["gold"]:
            if g["doc"] not in have:
                problems.append(f"{q['q']!r}: unknown doc {g['doc']}")
            elif "unit" in g and g["unit"] not in have[g["doc"]]:
                problems.append(f"{q['q']!r}: {g['doc']} has no section {g['unit']}")
    return problems


def evaluate(queries: Sequence[dict], runners: dict[str, Callable[[str], list[Hit]]], k: int = 5) -> list[dict]:
    rows = []
    for q in queries:
        row = {"q": q["q"], "type": q["type"]}
        for name, run in runners.items():
            hits = run(q["q"])
            row[name] = recall_at_k(hits, q["gold"], k)
        rows.append(row)
    return rows


def summarise(rows: Sequence[dict], methods: Sequence[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label, sel in (("meaning", lambda r: r["type"] == "meaning"), ("reference", lambda r: r["type"] == "reference"),
                       ("all", lambda r: True)):
        part = [r for r in rows if sel(r)]
        out[label] = {m: (sum(r[m] for r in part) / len(part) if part else 0.0) for m in methods}
    return out


def format_table(rows: Sequence[dict], methods: Sequence[str], k: int = 5) -> str:
    w = max(len(r["q"]) for r in rows) + 1
    lines = [f"{'query':<{w}} {'type':<9} " + " ".join(f"{m:>13}" for m in methods), "-" * (w + 11 + 14 * len(methods))]
    for r in rows:
        lines.append(f"{r['q']:<{w}} {r['type']:<9} " + " ".join(f"{'hit' if r[m] else '-':>13}" for m in methods))
    lines.append("-" * (w + 11 + 14 * len(methods)))
    for label, vals in summarise(rows, methods).items():
        n = sum(1 for r in rows if label == "all" or r["type"] == label)
        lines.append(f"{'recall@' + str(k) + ' ' + label + f' ({n})':<{w}} {'':<9} " + " ".join(f"{vals[m]:>13.2f}" for m in methods))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true", help="skip the (slower) hybrid+rerank column")
    a = ap.parse_args(argv)
    setup_logging()
    from config import settings
    from .bm25 import load_or_build
    from .embedder import Embedder
    from .hybrid import hybrid_search
    from .mapping import load_map
    from .rerank import Reranker
    from .store import Store

    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    problems = check_gold(queries, settings.chunks_data_dir)
    if problems:
        print("GOLD SET PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    emb = Embedder()
    shared = dict(embedder=emb, store=Store(emb.collection_name), bm25=load_or_build(), section_map=load_map(),
                  reranker=Reranker())
    runners: dict[str, Callable[[str], list[Hit]]] = {
        "dense": lambda q: hybrid_search(q, a.k, mode="dense", **shared),
        "bm25": lambda q: hybrid_search(q, a.k, mode="bm25", **shared),
        "hybrid": lambda q: hybrid_search(q, a.k, mode="hybrid", rerank=False, **shared),
    }
    if not a.no_rerank:
        runners["hybrid+rerank"] = lambda q: hybrid_search(q, a.k, mode="hybrid", rerank=True, **shared)
    rows = evaluate(queries, runners, a.k)
    print(format_table(rows, list(runners), a.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
