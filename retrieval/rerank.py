"""Cross-encoder reranker (reads query and chunk together), with a safe fallback to the fused order."""
import logging
from typing import Callable, Optional, Sequence

from config import settings

log = logging.getLogger(__name__)
MAX_LENGTH = 512

Scorer = Callable[[Sequence[tuple[str, str]]], Sequence[float]]


class Reranker:
    def __init__(self, model_name: Optional[str] = None, scorer: Optional[Scorer] = None) -> None:
        self.model_name = model_name or settings.rerank_model
        self._scorer = scorer  # injected in tests so no model is downloaded
        self._model = None
        self.failed = False

    def _score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        if self._scorer is not None:
            return self._scorer(pairs)
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=MAX_LENGTH)
        return [float(s) for s in self._model.predict([list(p) for p in pairs], show_progress_bar=False)]

    def rerank(self, query: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, float]]:
        """candidates: [(chunk_id, text)] in fused order. Returns [(chunk_id, score)] best first.
        If the model cannot run, the fused order is kept (scores are 0) and `failed` is set."""
        if not candidates:
            return []
        try:
            scores = self._score([(query, text) for _, text in candidates])
        except Exception as e:  # missing model / offline: degrade, never fail the search
            log.warning("reranker unavailable (%s); keeping fused order", e)
            self.failed = True
            return [(cid, 0.0) for cid, _ in candidates]
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [(candidates[i][0], float(scores[i])) for i in order]
