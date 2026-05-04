import uuid

import pytest

from be_my_wiki.parsing.chunker import Chunk
from be_my_wiki.store.base import StoreStats, VectorStore
from be_my_wiki.store.chroma import ChromaStore


def _chunk(note_path: str, idx: int, body: str, **md: object) -> Chunk:
    metadata: dict = {"title": "T", "tags": [], "aliases": []}
    metadata.update(md)
    return Chunk(
        note_path=note_path,
        chunk_index=idx,
        text=f"prefix\n\n{body}",
        body=body,
        heading_path=("Section",),
        metadata=metadata,
        content_hash=f"hash-{note_path}-{idx}",
    )


@pytest.fixture
def store() -> ChromaStore:
    # Unique collection per test to isolate state across the chromadb singleton.
    return ChromaStore(collection=f"test_{uuid.uuid4().hex[:8]}")


def test_implements_vector_store_protocol(store):
    assert isinstance(store, VectorStore)


def test_empty_store_stats(store):
    assert store.stats() == StoreStats(total_chunks=0, total_notes=0)


def test_upsert_and_stats(store):
    chunks = [_chunk("a.md", 0, "alpha"), _chunk("a.md", 1, "beta")]
    store.upsert(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    s = store.stats()
    assert s.total_chunks == 2
    assert s.total_notes == 1


def test_upsert_replaces_existing(store):
    c0 = _chunk("a.md", 0, "alpha")
    store.upsert([c0], [[1.0, 0.0, 0.0]])
    store.upsert([_chunk("a.md", 0, "alpha v2")], [[0.5, 0.5, 0.0]])
    assert store.stats().total_chunks == 1
    assert store.get_chunk_hashes("a.md") == {0: "hash-a.md-0"}


def test_search_returns_ranked_hits(store):
    chunks = [
        _chunk("a.md", 0, "alpha"),
        _chunk("b.md", 0, "beta"),
        _chunk("c.md", 0, "gamma"),
    ]
    # query [1, 0] -> a (cos=1.0), c (cos~0.994), b (cos=0)
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])

    hits = store.search([1.0, 0.0], top_k=3)
    assert len(hits) == 3
    assert hits[0].note_path == "a.md"
    assert hits[0].body == "alpha"
    assert hits[1].note_path == "c.md"
    assert hits[2].note_path == "b.md"
    assert hits[0].score > hits[1].score > hits[2].score


def test_search_top_k_zero(store):
    store.upsert([_chunk("a.md", 0, "x")], [[1.0, 0.0]])
    assert store.search([1.0, 0.0], top_k=0) == []


def test_search_empty_store(store):
    assert store.search([1.0, 0.0], top_k=5) == []


def test_get_chunk_hashes(store):
    chunks = [
        _chunk("a.md", 0, "x"),
        _chunk("a.md", 1, "y"),
        _chunk("b.md", 0, "z"),
    ]
    store.upsert(chunks, [[1, 0], [0, 1], [1, 1]])
    assert store.get_chunk_hashes("a.md") == {
        0: "hash-a.md-0",
        1: "hash-a.md-1",
    }
    assert store.get_chunk_hashes("nonexistent.md") == {}


def test_delete_whole_note(store):
    chunks = [
        _chunk("a.md", 0, "x"),
        _chunk("a.md", 1, "y"),
        _chunk("b.md", 0, "z"),
    ]
    store.upsert(chunks, [[1, 0], [0, 1], [1, 1]])
    store.delete("a.md")
    assert store.stats().total_chunks == 1
    assert store.get_chunk_hashes("a.md") == {}


def test_delete_specific_chunks(store):
    chunks = [
        _chunk("a.md", 0, "x"),
        _chunk("a.md", 1, "y"),
        _chunk("a.md", 2, "z"),
    ]
    store.upsert(chunks, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    store.delete("a.md", chunk_indices=[1])
    assert store.get_chunk_hashes("a.md") == {
        0: "hash-a.md-0",
        2: "hash-a.md-2",
    }


def test_metadata_roundtrip_with_lists(store):
    c = Chunk(
        note_path="x.md",
        chunk_index=0,
        text="prefix\n\nbody",
        body="body",
        heading_path=("H1", "H2"),
        metadata={
            "title": "T",
            "tags": ["ml", "project"],
            "aliases": ["alt"],
        },
        content_hash="h",
    )
    store.upsert([c], [[1.0, 0.0]])
    hits = store.search([1.0, 0.0], top_k=1)
    assert hits[0].heading_path == ("H1", "H2")
    assert hits[0].metadata["tags"] == ["ml", "project"]
    assert hits[0].metadata["aliases"] == ["alt"]
    assert hits[0].metadata["title"] == "T"


def test_list_note_paths(store):
    chunks = [
        _chunk("a.md", 0, "x"),
        _chunk("a.md", 1, "y"),
        _chunk("b.md", 0, "z"),
    ]
    store.upsert(chunks, [[1, 0], [0, 1], [1, 1]])
    assert store.list_note_paths() == {"a.md", "b.md"}


def test_list_note_paths_empty(store):
    assert store.list_note_paths() == set()


def test_distinct_note_count(store):
    chunks = [
        _chunk("a.md", 0, "x"),
        _chunk("a.md", 1, "y"),
        _chunk("b.md", 0, "z"),
        _chunk("c.md", 0, "w"),
    ]
    store.upsert(chunks, [[1, 0], [0, 1], [1, 1], [0, 0]])
    s = store.stats()
    assert s.total_chunks == 4
    assert s.total_notes == 3


def test_upsert_empty_is_noop(store):
    store.upsert([], [])
    assert store.stats().total_chunks == 0


def test_upsert_length_mismatch_raises(store):
    with pytest.raises(ValueError, match="same length"):
        store.upsert([_chunk("a.md", 0, "x")], [[1, 0], [0, 1]])


# --- filter pushdown ---


def test_search_with_single_tag_filter(store):
    chunks = [
        _chunk("a.md", 0, "alpha", tags=["ml"]),
        _chunk("b.md", 0, "beta", tags=["project"]),
        _chunk("c.md", 0, "gamma", tags=["ml", "project"]),
    ]
    store.upsert(chunks, [[1, 0], [0, 1], [0.5, 0.5]])
    hits = store.search([1.0, 0.0], top_k=10, tags=["ml"])
    assert {h.note_path for h in hits} == {"a.md", "c.md"}


def test_search_with_multiple_tag_filter_or_semantics(store):
    chunks = [
        _chunk("a.md", 0, "alpha", tags=["ml"]),
        _chunk("b.md", 0, "beta", tags=["project"]),
        _chunk("c.md", 0, "gamma", tags=["other"]),
    ]
    store.upsert(chunks, [[1, 0], [0, 1], [0.5, 0.5]])
    hits = store.search([1.0, 0.0], top_k=10, tags=["ml", "project"])
    assert {h.note_path for h in hits} == {"a.md", "b.md"}


def test_search_with_nonexistent_tag_returns_empty(store):
    chunks = [_chunk("a.md", 0, "x", tags=["ml"])]
    store.upsert(chunks, [[1.0, 0.0]])
    assert store.search([1.0, 0.0], top_k=10, tags=["nonexistent"]) == []


def test_search_with_path_prefix_top_level(store):
    chunks = [
        _chunk("ML/transformer.md", 0, "x"),
        _chunk("ML/basics/intro.md", 0, "y"),
        _chunk("Daily/note.md", 0, "z"),
    ]
    store.upsert(chunks, [[1, 0], [0, 1], [1, 1]])
    hits = store.search([1.0, 0.0], top_k=10, path_prefix="ML")
    assert {h.note_path for h in hits} == {
        "ML/transformer.md",
        "ML/basics/intro.md",
    }


def test_search_with_path_prefix_nested_level(store):
    chunks = [
        _chunk("ML/transformer.md", 0, "x"),
        _chunk("ML/basics/intro.md", 0, "y"),
    ]
    store.upsert(chunks, [[1, 0], [0, 1]])
    hits = store.search([1.0, 0.0], top_k=10, path_prefix="ML/basics")
    assert {h.note_path for h in hits} == {"ML/basics/intro.md"}


def test_search_with_path_prefix_respects_directory_boundary(store):
    """`path_prefix="ML"` must not match `MLops/foo.md`."""
    chunks = [
        _chunk("ML/foo.md", 0, "x"),
        _chunk("MLops/bar.md", 0, "y"),
    ]
    store.upsert(chunks, [[1, 0], [0, 1]])
    hits = store.search([1.0, 0.0], top_k=10, path_prefix="ML")
    assert {h.note_path for h in hits} == {"ML/foo.md"}


def test_search_with_combined_tag_and_path_filter(store):
    chunks = [
        _chunk("ML/a.md", 0, "x", tags=["ml"]),
        _chunk("ML/b.md", 0, "y", tags=["project"]),
        _chunk("Daily/c.md", 0, "z", tags=["ml"]),
    ]
    store.upsert(chunks, [[1, 0], [0, 1], [1, 1]])
    hits = store.search([1.0, 0.0], top_k=10, tags=["ml"], path_prefix="ML")
    assert {h.note_path for h in hits} == {"ML/a.md"}


def test_search_no_filter_returns_all_results(store):
    chunks = [_chunk("a.md", 0, "x"), _chunk("b.md", 0, "y")]
    store.upsert(chunks, [[1, 0], [0, 1]])
    assert len(store.search([1.0, 0.0], top_k=10)) == 2


def test_search_korean_tag_filter(store):
    chunks = [
        _chunk("ko.md", 0, "x", tags=["한국어"]),
        _chunk("en.md", 0, "y", tags=["english"]),
    ]
    store.upsert(chunks, [[1, 0], [0, 1]])
    hits = store.search([1.0, 0.0], top_k=10, tags=["한국어"])
    assert {h.note_path for h in hits} == {"ko.md"}


def test_pushdown_helper_fields_not_leaked_in_metadata(store):
    chunks = [_chunk("ML/a.md", 0, "x", tags=["ml"])]
    store.upsert(chunks, [[1.0, 0.0]])
    hits = store.search([1.0, 0.0], top_k=1)
    md = hits[0].metadata
    assert not any(k.startswith("tag__") for k in md)
    assert not any(k.startswith("dir_lvl") for k in md)
    assert md["tags"] == ["ml"]


# --- multilingual + TeX coverage (per project policy) ---


def test_korean_content_roundtrip(store):
    c = Chunk(
        note_path="ko.md",
        chunk_index=0,
        text="Title: 메모\nTags: 한국어, 머신러닝\n\n이것은 한글 본문입니다.",
        body="이것은 한글 본문입니다.",
        heading_path=("서론",),
        metadata={
            "title": "메모",
            "tags": ["한국어", "머신러닝"],
            "aliases": [],
        },
        content_hash="ko-hash",
    )
    store.upsert([c], [[1.0, 0.0]])
    hits = store.search([1.0, 0.0], top_k=1)
    assert hits[0].body == "이것은 한글 본문입니다."
    assert hits[0].heading_path == ("서론",)
    assert hits[0].metadata["tags"] == ["한국어", "머신러닝"]
    assert hits[0].metadata["title"] == "메모"


def test_tex_content_roundtrip(store):
    body = r"$$\int_0^1 x \, dx = \frac{1}{2}$$"
    c = Chunk(
        note_path="math.md",
        chunk_index=0,
        text=f"Title: Math\n\n{body}",
        body=body,
        heading_path=("적분",),
        metadata={"title": "Math", "tags": [], "aliases": []},
        content_hash="tex-hash",
    )
    store.upsert([c], [[1.0, 0.0]])
    hits = store.search([1.0, 0.0], top_k=1)
    assert hits[0].body == body
    assert r"\int_0^1 x \, dx" in hits[0].body
    assert r"\frac{1}{2}" in hits[0].body
    assert hits[0].heading_path == ("적분",)
