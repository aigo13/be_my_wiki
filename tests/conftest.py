"""Shared pytest fixtures.

Local fixtures inside individual test files override anything declared
here. The ``fake_st`` fixture is defined here so multiple test files can
patch ``sentence_transformers.SentenceTransformer`` consistently without
ever downloading the real bge-m3 model.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest


@pytest.fixture
def fake_st(monkeypatch):
    """Patch sentence_transformers so BgeM3Embedder can run without
    downloading the real bge-m3 model.

    The fake produces deterministic non-zero vectors derived from sha256
    of the input text, which keeps Chroma's cosine search well-defined.
    """

    class _FakeModel:
        DIM = 4

        def __init__(self, *args, **kwargs) -> None:
            pass

        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                return np.array(self._vec(texts), dtype=np.float32)
            return np.array([self._vec(t) for t in texts], dtype=np.float32)

        @classmethod
        def _vec(cls, text: str) -> list[float]:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            return [b / 255.0 for b in h[: cls.DIM]]

        def get_sentence_embedding_dimension(self) -> int:
            return self.DIM

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeModel)
    return _FakeModel
