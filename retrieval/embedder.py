"""Text -> vector with multilingual-E5. E5 expects 'query: ' / 'passage: ' prefixes; vectors are L2-normalised."""
import math
from typing import Callable, Optional, Protocol, Sequence

from config import settings

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
MAX_SEQ_LENGTH = 512
FORMAT_VERSION = "v1"  # bump when the embedded text format changes; it names the Chroma collection


class Encoder(Protocol):
    """Anything that turns prefixed strings into vectors (the real model, or a fake in tests)."""

    def __call__(self, texts: Sequence[str]) -> list[list[float]]: ...


def normalise(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [float(x) for x in vec] if norm == 0 else [float(x) / norm for x in vec]


def passage_text(record: dict) -> str:
    """What gets embedded for a chunk: its breadcrumb (act > chapter > section) plus the text."""
    context = record.get("context") or ""
    return f"{context}\n{record['text']}" if context else record["text"]


class Embedder:
    def __init__(self, model_name: Optional[str] = None, batch_size: Optional[int] = None,
                 encoder: Optional[Callable[[Sequence[str]], list[list[float]]]] = None) -> None:
        self.model_name = model_name or settings.embed_model
        self.batch_size = batch_size or settings.embed_batch_size
        self._encoder = encoder  # injected in tests so no model is downloaded
        self._model = None

    @property
    def collection_name(self) -> str:
        slug = "".join(c if c.isalnum() else "_" for c in self.model_name.split("/")[-1].lower())
        return f"nyaya_{slug}_{FORMAT_VERSION}"

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, only when needed
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = MAX_SEQ_LENGTH
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._encoder is not None:
            return [normalise(v) for v in self._encoder(texts)]
        vecs = self._load().encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                   show_progress_bar=False)
        return [[float(x) for x in v] for v in vecs]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._encode([PASSAGE_PREFIX + t for t in texts[i:i + self.batch_size]]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._encode([QUERY_PREFIX + text])[0]

    def token_count(self, text: str) -> int:
        """Real tokenizer length (loads the model); used to report chunks that would be truncated."""
        return len(self._load().tokenizer.encode(PASSAGE_PREFIX + text))
