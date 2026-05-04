import uuid

import pytest

from be_my_wiki.embedding.base import Embedder
from be_my_wiki.indexer.pipeline import Indexer, IndexResult
from be_my_wiki.store.chroma import ChromaStore

from tests._fakes import FakeEmbedder


def test_fake_embedder_satisfies_protocol():
    assert isinstance(FakeEmbedder(), Embedder)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> ChromaStore:
    return ChromaStore(collection=f"idx_{uuid.uuid4().hex[:8]}")


@pytest.fixture
def indexer(tmp_path, embedder, store) -> Indexer:
    return Indexer(vault_root=tmp_path, embedder=embedder, store=store)


def test_index_note_fresh(indexer, tmp_path, store):
    note = tmp_path / "test.md"
    note.write_text("## Section\nbody", encoding="utf-8")
    r = indexer.index_note(note)
    assert r == IndexResult(
        note_path="test.md", added=1, updated=0, skipped=0, deleted=0
    )
    assert store.stats().total_chunks == 1


def test_index_note_unchanged_skips_embedding(indexer, tmp_path, embedder):
    note = tmp_path / "test.md"
    note.write_text("## Section\nbody", encoding="utf-8")
    indexer.index_note(note)
    embedder.embed_doc_calls = 0

    r = indexer.index_note(note)
    assert r.skipped == 1
    assert r.added == 0
    assert r.updated == 0
    assert r.deleted == 0
    assert embedder.embed_doc_calls == 0


def test_edit_one_section_updates_only_that(indexer, tmp_path, embedder):
    note = tmp_path / "n.md"
    note.write_text("## A\nalpha\n\n## B\nbeta", encoding="utf-8")
    indexer.index_note(note)
    embedder.embed_doc_calls = 0

    note.write_text("## A\nalpha\n\n## B\nbeta-EDITED", encoding="utf-8")
    r = indexer.index_note(note)
    assert r.skipped == 1
    assert r.updated == 1
    assert r.added == 0
    assert embedder.embed_doc_calls == 1  # one batch with the changed chunk only


def test_inserting_section_shifts_indices(indexer, tmp_path):
    note = tmp_path / "n.md"
    note.write_text("## A\nalpha\n\n## B\nbeta", encoding="utf-8")
    r1 = indexer.index_note(note)
    assert r1.added == 2

    note.write_text("## A\nalpha\n\n## NEW\nnew\n\n## B\nbeta", encoding="utf-8")
    r2 = indexer.index_note(note)
    # 0:A unchanged (skip), 1:B->NEW (update), 2:B (new index)
    assert r2.skipped == 1
    assert r2.updated == 1
    assert r2.added == 1
    assert r2.deleted == 0


def test_removing_section_deletes_chunks(indexer, tmp_path):
    note = tmp_path / "n.md"
    note.write_text(
        "## A\nalpha\n\n## B\nbeta\n\n## C\ngamma", encoding="utf-8"
    )
    indexer.index_note(note)

    note.write_text("## A\nalpha\n\n## C\ngamma", encoding="utf-8")
    r = indexer.index_note(note)
    # 0:A unchanged, 1:B->C (update), 2:gone (delete)
    assert r.skipped == 1
    assert r.updated == 1
    assert r.deleted == 1


def test_delete_note_removes_chunks(indexer, tmp_path, store):
    note = tmp_path / "n.md"
    note.write_text("## A\nalpha", encoding="utf-8")
    indexer.index_note(note)
    assert store.stats().total_chunks == 1

    deleted = indexer.delete_note(note)
    assert deleted == 1
    assert store.stats().total_chunks == 0


def test_delete_nonexistent_note_returns_zero(indexer, tmp_path):
    fake = tmp_path / "ghost.md"
    assert indexer.delete_note(fake) == 0


def test_index_directory_walks_all_md(indexer, tmp_path):
    (tmp_path / "a.md").write_text("## A\nx", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "b.md").write_text("## B\ny", encoding="utf-8")
    (tmp_path / "not-md.txt").write_text("ignore me", encoding="utf-8")

    r = indexer.index_directory()
    assert r.notes_total == 2
    assert r.notes_changed == 2
    assert r.chunks_added == 2


def test_index_directory_skips_ignore_dirs(tmp_path, embedder, store):
    (tmp_path / "good.md").write_text("## A\nx", encoding="utf-8")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "config.md").write_text(
        "## ignored\nz", encoding="utf-8"
    )
    (tmp_path / ".trash").mkdir()
    (tmp_path / ".trash" / "old.md").write_text("## old\nz", encoding="utf-8")

    idx = Indexer(vault_root=tmp_path, embedder=embedder, store=store)
    r = idx.index_directory()
    assert r.notes_total == 1
    assert r.chunks_added == 1


def test_index_directory_prunes_orphans(indexer, tmp_path, store):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("## A\nx", encoding="utf-8")
    b.write_text("## B\ny", encoding="utf-8")
    indexer.index_directory()
    assert store.stats().total_notes == 2

    b.unlink()
    r = indexer.index_directory()
    assert r.notes_pruned == 1
    assert store.stats().total_notes == 1


def test_index_directory_prune_orphans_disabled(indexer, tmp_path, store):
    a = tmp_path / "a.md"
    a.write_text("## A\nx", encoding="utf-8")
    indexer.index_directory()

    a.unlink()
    r = indexer.index_directory(prune_orphans=False)
    assert r.notes_pruned == 0
    assert store.stats().total_notes == 1  # orphan still present


def test_relative_path_uses_forward_slashes(indexer, tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    note = sub / "n.md"
    note.write_text("## A\nx", encoding="utf-8")
    r = indexer.index_note(note)
    assert r.note_path == "subdir/n.md"


# --- multilingual + TeX coverage (per project policy) ---


def test_korean_note_indexed(indexer, tmp_path, store):
    note = tmp_path / "ko.md"
    note.write_text(
        "---\ntags: [한국어, 머신러닝]\n---\n\n## 서론\n이것은 한글 본문입니다.",
        encoding="utf-8",
    )
    r = indexer.index_note(note)
    assert r.added == 1
    hits = store.search([0.5] * 4, top_k=1)
    assert "이것은 한글 본문입니다." in hits[0].body
    assert hits[0].metadata["tags"] == ["한국어", "머신러닝"]
    assert hits[0].heading_path == ("서론",)


def test_tex_note_indexed(indexer, tmp_path, store):
    note = tmp_path / "math.md"
    note.write_text(
        "## Integral\n\n$$\n\\int_0^1 x \\, dx = \\frac{1}{2}\n$$",
        encoding="utf-8",
    )
    r = indexer.index_note(note)
    assert r.added == 1
    hits = store.search([0.5] * 4, top_k=1)
    assert r"\int_0^1 x \, dx" in hits[0].body
    assert r"\frac{1}{2}" in hits[0].body


def test_utf8_bom_file_reads_correctly(indexer, tmp_path):
    note = tmp_path / "bom.md"
    note.write_bytes("﻿## Section\n안녕하세요".encode("utf-8"))
    r = indexer.index_note(note)
    assert r.added == 1
