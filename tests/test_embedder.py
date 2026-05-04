import numpy as np
import pytest

from be_my_wiki.embedding.base import Embedder
from be_my_wiki.embedding.bge_m3 import BgeM3Embedder


class _FakeModel:
    """Stand-in for sentence_transformers.SentenceTransformer.

    Returns deterministic vectors and tracks construction so tests can
    assert lazy loading and parameter plumbing without downloading the
    real 2.3GB model.
    """

    init_count = 0
    DIM = 4

    def __init__(self, name, device=None, **kwargs):
        type(self).init_count += 1
        self.name = name
        self.device = device

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.full(self.DIM, 0.5, dtype=np.float32)
        return np.full((len(texts), self.DIM), 0.5, dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return self.DIM


@pytest.fixture
def fake_st(monkeypatch):
    _FakeModel.init_count = 0
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeModel)
    return _FakeModel


def test_constructor_does_not_load_model(fake_st):
    BgeM3Embedder()
    assert fake_st.init_count == 0


def test_implements_embedder_protocol():
    assert isinstance(BgeM3Embedder(), Embedder)


def test_embed_query_loads_model_once(fake_st):
    emb = BgeM3Embedder()
    vec = emb.embed_query("hello")
    assert isinstance(vec, list)
    assert len(vec) == fake_st.DIM
    assert fake_st.init_count == 1

    emb.embed_query("world")
    assert fake_st.init_count == 1


def test_embed_documents_returns_list_of_vectors(fake_st):
    emb = BgeM3Embedder()
    out = emb.embed_documents(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == fake_st.DIM for v in out)
    assert all(isinstance(x, float) for v in out for x in v)


def test_embed_documents_empty_returns_empty_without_loading(fake_st):
    emb = BgeM3Embedder()
    assert emb.embed_documents([]) == []
    assert fake_st.init_count == 0


def test_dim_loads_model_and_returns_int(fake_st):
    emb = BgeM3Embedder()
    assert emb.dim == fake_st.DIM
    assert fake_st.init_count == 1


def test_device_propagates_to_model(fake_st):
    emb = BgeM3Embedder(device="cuda")
    emb.embed_query("x")
    assert emb._model.device == "cuda"


def test_encode_kwargs_propagate(fake_st, monkeypatch):
    captured: dict = {}
    real_encode = _FakeModel.encode

    def spy_encode(self, texts, **kwargs):
        captured.update(kwargs)
        return real_encode(self, texts, **kwargs)

    monkeypatch.setattr(_FakeModel, "encode", spy_encode)

    emb = BgeM3Embedder(batch_size=32)
    emb.embed_documents(["a"])

    assert captured.get("batch_size") == 32
    assert captured.get("normalize_embeddings") is True
    assert captured.get("show_progress_bar") is False
