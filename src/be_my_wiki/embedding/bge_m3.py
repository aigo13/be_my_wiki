"""BAAI/bge-m3 embedder via sentence-transformers.

bge-m3 is multilingual (100+ languages incl. Korean), accepts up to
8192 input tokens, and produces 1024-dim embeddings. Embeddings are
L2-normalized so dot product equals cosine similarity, which matches
Chroma's default metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Vector

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class BgeM3Embedder:
    """bge-m3 wrapper that loads the model lazily on first use."""

    DEFAULT_MODEL = "BAAI/bge-m3"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def dim(self) -> int:
        return self._ensure_model().get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        arr = self._ensure_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return arr.tolist()

    def embed_query(self, text: str) -> Vector:
        arr = self._ensure_model().encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return arr.tolist()

    def _ensure_model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model
