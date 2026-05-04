"""VectorStore protocol and shared data classes.

Concrete implementations (e.g. ``chroma.ChromaStore``) must satisfy this
interface. The indexer and the MCP search tool depend only on this
Protocol so we can swap backends (Chroma -> Qdrant) without touching
call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..parsing.chunker import Chunk

Vector = list[float]


@dataclass(frozen=True)
class SearchHit:
    note_path: str
    chunk_index: int
    body: str
    heading_path: tuple[str, ...]
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class StoreStats:
    total_chunks: int
    total_notes: int


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None: ...

    def delete(
        self,
        note_path: str,
        chunk_indices: list[int] | None = None,
    ) -> None: ...

    def search(
        self,
        query_vec: Vector,
        top_k: int,
        *,
        tags: list[str] | None = None,
        path_prefix: str | None = None,
    ) -> list[SearchHit]: ...

    def get_chunk_hashes(self, note_path: str) -> dict[int, str]: ...

    def list_note_paths(self) -> set[str]: ...

    def stats(self) -> StoreStats: ...
