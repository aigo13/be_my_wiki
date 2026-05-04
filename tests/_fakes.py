"""Test fakes shared across test files.

Filename starts with ``_`` so pytest does not collect it as a test module.
"""

from __future__ import annotations

import hashlib


class FakeEmbedder:
    """Deterministic in-memory embedder.

    Vectors are derived from sha256 of the text, so the same text always
    produces the same vector. Tracks call counts so tests can assert that
    unchanged chunks are not re-embedded.
    """

    DIM = 4

    def __init__(self) -> None:
        self.embed_doc_calls = 0
        self.embed_query_calls = 0

    @property
    def dim(self) -> int:
        return self.DIM

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_doc_calls += 1
        return [self._vec_for(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls += 1
        return self._vec_for(text)

    @classmethod
    def _vec_for(cls, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in h[: cls.DIM]]
