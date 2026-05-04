"""Pipeline orchestration: parse -> chunk -> diff -> embed -> upsert.

Indexer ties the chunker, embedder, and store together.

- ``index_note(abs_path)``: index/refresh a single file with incremental
  diffing (only changed chunks are re-embedded).
- ``delete_note(abs_path)``: remove all chunks for a file.
- ``index_directory()``: walk the vault root, index every ``.md`` file,
  and (optionally) prune entries for files no longer on disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..embedding.base import Embedder
from ..parsing.chunker import Chunk, chunk_note
from ..store.base import VectorStore

logger = logging.getLogger(__name__)

_DEFAULT_IGNORE_DIRS = frozenset({".obsidian", ".trash", ".git", "node_modules"})


@dataclass(frozen=True)
class IndexResult:
    note_path: str
    added: int
    updated: int
    skipped: int
    deleted: int

    @property
    def changed(self) -> bool:
        return self.added > 0 or self.updated > 0 or self.deleted > 0


@dataclass(frozen=True)
class BulkIndexResult:
    notes_total: int
    notes_changed: int
    notes_unchanged: int
    notes_failed: int
    notes_pruned: int
    chunks_added: int
    chunks_updated: int
    chunks_skipped: int
    chunks_deleted: int


class Indexer:
    def __init__(
        self,
        *,
        vault_root: Path,
        embedder: Embedder,
        store: VectorStore,
        max_tokens: int = 512,
        heading_level: int = 2,
        ignore_dirs: set[str] | None = None,
    ) -> None:
        self.vault_root = vault_root
        self.embedder = embedder
        self.store = store
        self.max_tokens = max_tokens
        self.heading_level = heading_level
        self.ignore_dirs = (
            set(_DEFAULT_IGNORE_DIRS) if ignore_dirs is None else set(ignore_dirs)
        )

    def index_note(self, abs_path: Path) -> IndexResult:
        note_path = self._relative_path(abs_path)
        source = abs_path.read_text(encoding="utf-8-sig")
        chunks = chunk_note(
            note_path=note_path,
            source=source,
            max_tokens=self.max_tokens,
            heading_level=self.heading_level,
        )

        existing = self.store.get_chunk_hashes(note_path)
        current_indices = {c.chunk_index for c in chunks}

        to_upsert: list[Chunk] = []
        skipped = 0
        for c in chunks:
            if existing.get(c.chunk_index) == c.content_hash:
                skipped += 1
            else:
                to_upsert.append(c)

        to_delete_indices = sorted(i for i in existing if i not in current_indices)
        added = sum(1 for c in to_upsert if c.chunk_index not in existing)
        updated = len(to_upsert) - added

        if to_upsert:
            vectors = self.embedder.embed_documents([c.text for c in to_upsert])
            self.store.upsert(to_upsert, vectors)
        if to_delete_indices:
            self.store.delete(note_path, chunk_indices=to_delete_indices)

        return IndexResult(
            note_path=note_path,
            added=added,
            updated=updated,
            skipped=skipped,
            deleted=len(to_delete_indices),
        )

    def delete_note(self, abs_path: Path) -> int:
        note_path = self._relative_path(abs_path)
        existing = self.store.get_chunk_hashes(note_path)
        if existing:
            self.store.delete(note_path)
        return len(existing)

    def index_directory(self, *, prune_orphans: bool = True) -> BulkIndexResult:
        md_files = list(self._walk_md_files())
        current_paths = {self._relative_path(p) for p in md_files}

        notes_changed = 0
        notes_unchanged = 0
        notes_failed = 0
        chunks_added = 0
        chunks_updated = 0
        chunks_skipped = 0
        chunks_deleted = 0

        for path in md_files:
            try:
                r = self.index_note(path)
            except Exception as exc:
                logger.warning("Failed to index %s: %s", path, exc)
                notes_failed += 1
                continue

            if r.changed:
                notes_changed += 1
            else:
                notes_unchanged += 1
            chunks_added += r.added
            chunks_updated += r.updated
            chunks_skipped += r.skipped
            chunks_deleted += r.deleted

        notes_pruned = 0
        if prune_orphans:
            stored = self.store.list_note_paths()
            for orphan in stored - current_paths:
                self.store.delete(orphan)
                notes_pruned += 1

        return BulkIndexResult(
            notes_total=len(md_files),
            notes_changed=notes_changed,
            notes_unchanged=notes_unchanged,
            notes_failed=notes_failed,
            notes_pruned=notes_pruned,
            chunks_added=chunks_added,
            chunks_updated=chunks_updated,
            chunks_skipped=chunks_skipped,
            chunks_deleted=chunks_deleted,
        )

    def _walk_md_files(self) -> Iterator[Path]:
        for path in self.vault_root.rglob("*.md"):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(self.vault_root).parts
            if any(part in self.ignore_dirs for part in rel_parts):
                continue
            yield path

    def _relative_path(self, abs_path: Path) -> str:
        return abs_path.relative_to(self.vault_root).as_posix()
