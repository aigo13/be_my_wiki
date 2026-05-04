"""Tests for VaultWatcher.

Tests bypass the real watchdog Observer by calling ``enqueue`` and
``_flush_pending(force=True)`` directly. One test exercises the worker
thread briefly to verify debouncing actually waits.
"""

import time
import uuid
from pathlib import Path

import pytest

from be_my_wiki.indexer.pipeline import Indexer
from be_my_wiki.indexer.watcher import EventKind, VaultWatcher
from be_my_wiki.store.chroma import ChromaStore

from tests._fakes import FakeEmbedder


@pytest.fixture
def setup(tmp_path):
    embedder = FakeEmbedder()
    store = ChromaStore(collection=f"watch_{uuid.uuid4().hex[:8]}")
    indexer = Indexer(vault_root=tmp_path, embedder=embedder, store=store)
    watcher = VaultWatcher(indexer=indexer, debounce_seconds=0.05)
    return embedder, store, indexer, watcher


def _write(path: Path, content: str = "## A\nbody") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_enqueue_filters_non_md(tmp_path, setup):
    _, store, _, watcher = setup
    txt = _write(tmp_path / "x.txt", "not markdown")
    watcher.enqueue(EventKind.UPSERT, txt)
    watcher._flush_pending(force=True)
    assert store.stats().total_chunks == 0


def test_enqueue_filters_ignore_dirs(tmp_path, setup):
    _, store, _, watcher = setup
    note = _write(tmp_path / ".obsidian" / "config.md")
    watcher.enqueue(EventKind.UPSERT, note)
    watcher._flush_pending(force=True)
    assert store.stats().total_chunks == 0


def test_upsert_existing_file_indexes(tmp_path, setup):
    _, store, _, watcher = setup
    note = _write(tmp_path / "n.md")
    watcher.enqueue(EventKind.UPSERT, note)
    watcher._flush_pending(force=True)
    assert "n.md" in store.list_note_paths()


def test_upsert_missing_file_treated_as_delete(tmp_path, setup):
    _, store, indexer, watcher = setup
    note = _write(tmp_path / "n.md")
    indexer.index_note(note)
    note.unlink()

    watcher.enqueue(EventKind.UPSERT, note)
    watcher._flush_pending(force=True)
    assert "n.md" not in store.list_note_paths()


def test_delete_event_removes_from_store(tmp_path, setup):
    _, store, indexer, watcher = setup
    note = _write(tmp_path / "n.md")
    indexer.index_note(note)
    assert "n.md" in store.list_note_paths()

    watcher.enqueue(EventKind.DELETE, note)
    watcher._flush_pending(force=True)
    assert "n.md" not in store.list_note_paths()


def test_repeated_enqueue_coalesces(tmp_path, setup):
    embedder, _, _, watcher = setup
    note = _write(tmp_path / "n.md")

    for _ in range(5):
        watcher.enqueue(EventKind.UPSERT, note)
    watcher._flush_pending(force=True)

    # All 5 enqueues should resolve to one index_note call → one
    # embed_documents batch.
    assert embedder.embed_doc_calls == 1


def test_processing_exception_does_not_crash_worker(tmp_path, setup, monkeypatch):
    _, _, indexer, watcher = setup
    note = _write(tmp_path / "n.md")

    def boom(_path):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(indexer, "index_note", boom)

    # Should swallow the exception (and log it) rather than propagate.
    watcher.enqueue(EventKind.UPSERT, note)
    watcher._flush_pending(force=True)


def test_debounce_actually_delays(tmp_path, setup):
    _, store, _, watcher = setup
    watcher.debounce_seconds = 0.2
    note = _write(tmp_path / "n.md")

    watcher.enqueue(EventKind.UPSERT, note)
    watcher._flush_pending()  # not force — should NOT fire yet
    assert store.stats().total_chunks == 0

    time.sleep(0.25)
    watcher._flush_pending()  # debounce window passed → fires now
    assert store.stats().total_chunks >= 1


def test_worker_thread_processes_after_debounce(tmp_path, setup):
    _, store, _, watcher = setup
    watcher.debounce_seconds = 0.1
    note = _write(tmp_path / "n.md")

    # Drive the worker thread without starting watchdog Observer.
    import threading

    worker = threading.Thread(target=watcher._worker_loop, daemon=True)
    worker.start()
    try:
        watcher.enqueue(EventKind.UPSERT, note)
        # Wait long enough for debounce + at least one worker pass
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if store.stats().total_chunks >= 1:
                break
            time.sleep(0.05)
    finally:
        watcher._stop.set()
        worker.join(timeout=2)

    assert store.stats().total_chunks >= 1


def test_relative_path_outside_vault_rejected(tmp_path, setup):
    _, store, _, watcher = setup
    outside = tmp_path.parent / "outside.md"
    outside.write_text("body", encoding="utf-8")
    try:
        watcher.enqueue(EventKind.UPSERT, outside)
        watcher._flush_pending(force=True)
        assert store.stats().total_chunks == 0
    finally:
        outside.unlink()


# --- multilingual + TeX coverage (per project policy) ---


def test_korean_filename_with_tex_content(tmp_path, setup):
    _, store, _, watcher = setup
    note = _write(
        tmp_path / "한국어" / "수식.md",
        "## 적분\n$$\n\\int_0^1 x \\, dx = \\frac{1}{2}\n$$",
    )
    watcher.enqueue(EventKind.UPSERT, note)
    watcher._flush_pending(force=True)

    assert "한국어/수식.md" in store.list_note_paths()
    hits = store.search([0.5] * 4, top_k=1)
    assert r"\int_0^1 x \, dx" in hits[0].body
    assert hits[0].heading_path == ("적분",)
