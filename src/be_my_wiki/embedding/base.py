"""Embedder protocol.

Concrete implementations (e.g. ``bge_m3.BgeM3Embedder``) must satisfy this
interface. The indexer and the MCP search tool depend only on this Protocol,
which lets us swap models without touching call sites.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

Vector = list[float]


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...
