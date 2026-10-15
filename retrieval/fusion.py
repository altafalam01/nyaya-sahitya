"""Reciprocal rank fusion."""
from typing import Sequence


def rrf(rank_lists: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None) -> list[tuple[str, float]]:
    """Merge best-first id lists: score(id) = sum over lists of weight / (k + rank), rank starting at 1.

    Ties break by the order of first appearance, so the result is deterministic."""
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[str, float] = {}
    for ids, w in zip(rank_lists, weights):
        for rank, cid in enumerate(ids, 1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    order = {cid: i for i, cid in enumerate(dict.fromkeys(c for ids in rank_lists for c in ids))}
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))
