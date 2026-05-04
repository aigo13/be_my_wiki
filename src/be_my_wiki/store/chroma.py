"""ChromaDB-backed VectorStore.

Pass ``persist_path=None`` for an in-memory ephemeral store (used in tests).
The collection is created with cosine-distance space, which matches the
L2-normalized embeddings produced by BgeM3Embedder so dot product equals
cosine similarity.

Filter pushdown
---------------
``tags`` and ``path_prefix`` filters are pushed down to Chroma's ``where``
clause via boolean tag-flag fields and per-level directory string fields:

- ``tag__<sanitized_name>: True`` — one boolean per tag on a chunk.
  The sanitization replaces non-(ASCII-alphanumeric, underscore, Hangul)
  characters with ``_`` so unusual tag names are still queryable.
- ``dir_lvl{n}: "<canonical/path>"`` — one string per ancestor directory
  of the note. ``ML/basics/foo.md`` produces ``dir_lvl1="ML"`` and
  ``dir_lvl2="ML/basics"``. ``path_prefix`` matching uses the level
  derived from the prefix's slash count, so it always honours
  directory boundaries.

These fields are write-only for filtering. Display values (``tags``,
``heading_path``, etc.) still come from the JSON-serialized fields.
"""

from __future__ import annotations

import json
import re
from typing import Any

import chromadb

from ..parsing.chunker import Chunk
from .base import SearchHit, StoreStats, Vector


# Metadata keys whose values are lists. Chroma metadata only accepts
# scalar values, so these are JSON-serialized on write and parsed back
# on read.
_LIST_KEYS = frozenset({"tags", "aliases", "heading_path"})

# Allowed characters in a tag-field key. Anything else becomes ``_``.
# Hangul block (가-힣) is included so Korean tags survive.
_TAG_FIELD_SAFE = re.compile(r"[^A-Za-z0-9_가-힣]")


class ChromaStore:
    def __init__(
        self,
        *,
        persist_path: str | None = None,
        collection: str = "vault",
    ) -> None:
        if persist_path is None:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) "
                "must be the same length"
            )
        if not chunks:
            return
        self._collection.upsert(
            ids=[_chunk_id(c) for c in chunks],
            embeddings=vectors,
            documents=[c.body for c in chunks],
            metadatas=[_build_metadata(c) for c in chunks],
        )

    def delete(
        self,
        note_path: str,
        chunk_indices: list[int] | None = None,
    ) -> None:
        if chunk_indices is None:
            self._collection.delete(where={"note_path": note_path})
        elif chunk_indices:
            self._collection.delete(
                ids=[f"{note_path}::{i}" for i in chunk_indices]
            )

    def search(
        self,
        query_vec: Vector,
        top_k: int,
        *,
        tags: list[str] | None = None,
        path_prefix: str | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0 or self._collection.count() == 0:
            return []
        where = _build_where(tags=tags, path_prefix=path_prefix)
        res = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        if not ids:
            return []
        metadatas = (res.get("metadatas") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for md, doc, dist in zip(metadatas, documents, distances):
            md_d = _deserialize_metadata(dict(md or {}))
            # Strip filter-pushdown helper fields from the user-visible metadata.
            md_d = {
                k: v
                for k, v in md_d.items()
                if not (k.startswith("tag__") or k.startswith("dir_lvl"))
            }
            note_path = md_d.pop("note_path", "")
            chunk_index = int(md_d.pop("chunk_index", 0))
            heading_path = tuple(md_d.pop("heading_path", []))
            hits.append(
                SearchHit(
                    note_path=note_path,
                    chunk_index=chunk_index,
                    body=doc or "",
                    heading_path=heading_path,
                    metadata=md_d,
                    score=1.0 - float(dist),
                )
            )
        return hits

    def get_chunk_hashes(self, note_path: str) -> dict[int, str]:
        res = self._collection.get(
            where={"note_path": note_path},
            include=["metadatas"],
        )
        out: dict[int, str] = {}
        for md in res.get("metadatas") or []:
            if not md:
                continue
            out[int(md["chunk_index"])] = str(md["content_hash"])
        return out

    def list_note_paths(self) -> set[str]:
        if self._collection.count() == 0:
            return set()
        res = self._collection.get(include=["metadatas"])
        return {
            str(md["note_path"])
            for md in (res.get("metadatas") or [])
            if md and "note_path" in md
        }

    def stats(self) -> StoreStats:
        total_chunks = self._collection.count()
        if total_chunks == 0:
            return StoreStats(total_chunks=0, total_notes=0)
        return StoreStats(
            total_chunks=total_chunks,
            total_notes=len(self.list_note_paths()),
        )


def _chunk_id(chunk: Chunk) -> str:
    return f"{chunk.note_path}::{chunk.chunk_index}"


def _build_metadata(chunk: Chunk) -> dict[str, Any]:
    md: dict[str, Any] = {
        "note_path": chunk.note_path,
        "chunk_index": chunk.chunk_index,
        "content_hash": chunk.content_hash,
        "heading_path": list(chunk.heading_path),
    }
    for k, v in chunk.metadata.items():
        if k not in md:
            md[k] = v
    md = _serialize_metadata(md)

    # Filter-pushdown helper fields.
    for tag in chunk.metadata.get("tags") or []:
        if isinstance(tag, str) and tag.strip():
            md[_tag_field(tag.strip())] = True

    parts = chunk.note_path.split("/")[:-1]
    running: list[str] = []
    for i, part in enumerate(parts, 1):
        running.append(part)
        md[f"dir_lvl{i}"] = "/".join(running)

    return md


def _serialize_metadata(md: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in md.items():
        if k in _LIST_KEYS:
            v_list = list(v) if isinstance(v, (list, tuple)) else []
            out[k] = json.dumps(v_list, ensure_ascii=False)
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif v is None:
            continue
        else:
            out[k] = str(v)
    return out


def _deserialize_metadata(md: dict[str, Any]) -> dict[str, Any]:
    out = dict(md)
    for k in _LIST_KEYS:
        if k in out and isinstance(out[k], str):
            try:
                out[k] = json.loads(out[k])
            except json.JSONDecodeError:
                out[k] = []
    return out


def _tag_field(tag: str) -> str:
    return f"tag__{_TAG_FIELD_SAFE.sub('_', tag)}"


def _build_where(
    *,
    tags: list[str] | None,
    path_prefix: str | None,
) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []

    if tags:
        tag_clauses = [{_tag_field(t.strip()): True} for t in tags if t.strip()]
        if len(tag_clauses) == 1:
            clauses.append(tag_clauses[0])
        elif len(tag_clauses) > 1:
            clauses.append({"$or": tag_clauses})

    if path_prefix and path_prefix.strip("/"):
        parts = path_prefix.strip("/").split("/")
        clauses.append({f"dir_lvl{len(parts)}": "/".join(parts)})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
