"""MCP stdio server for be_my_wiki.

Exposes three tools and a single resource template over stdio. The server
is meant to be launched by an MCP client (Claude Desktop, Claude Code) and
then driven by an LLM that calls the tools to surface relevant chunks of
the user's Obsidian vault.

Tools
-----
- ``search(query, top_k=5, tags=None, path_prefix=None)``: semantic search
  over the indexed chunks. Returns a list of hits, each with a
  ``vault://`` URI the LLM can dereference for full content.
- ``get_note(note_path)``: metadata + headings outline + URI for a note.
  Does NOT return full content; that comes from the resource handler.
- ``stats()``: index counts and the configured vault path.

Resource template
-----------------
- ``vault://{note_path}``: full markdown content of a note. Path is
  validated to live inside ``vault_root`` (no traversal, no absolutes).

Layout
------
The pure ``_*_impl`` helpers below take primitive args (embedder, store,
vault root, path) so they are trivially unit-testable. ``create_server``
wires them into FastMCP via decorators.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frontmatter
from mcp.server.fastmcp import FastMCP

from ..config import load_config
from ..embedding.base import Embedder
from ..embedding.bge_m3 import BgeM3Embedder
from ..parsing.chunker import _find_headings, _normalize_list
from ..store.base import VectorStore
from ..store.chroma import ChromaStore


# --- pure logic (unit-testable) ---


def _search_impl(
    embedder: Embedder,
    store: VectorStore,
    query: str,
    top_k: int = 5,
    tags: list[str] | None = None,
    path_prefix: str | None = None,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    vec = embedder.embed_query(query)
    hits = store.search(
        vec, top_k=top_k, tags=tags, path_prefix=path_prefix
    )
    return {
        "hits": [
            {
                "note_path": h.note_path,
                "uri": f"vault://{h.note_path}",
                "chunk_index": h.chunk_index,
                "heading_path": list(h.heading_path),
                "snippet": h.body,
                "score": h.score,
                "title": h.metadata.get("title", ""),
                "tags": h.metadata.get("tags", []),
            }
            for h in hits
        ]
    }


def _get_note_impl(
    vault_root: Path,
    store: VectorStore,
    note_path: str,
) -> dict[str, Any]:
    abs_path = _resolve_safe(vault_root, note_path)
    stat = abs_path.stat()
    source = abs_path.read_text(encoding="utf-8-sig")
    post = frontmatter.loads(source)
    fm = dict(post.metadata) if post.metadata else {}

    title = str(fm["title"]).strip() if fm.get("title") else abs_path.stem
    tags = _normalize_list(fm.get("tags"))
    aliases = _normalize_list(fm.get("aliases"))
    chunk_count = len(store.get_chunk_hashes(note_path))

    outline = [
        {"level": h["level"], "text": h["text"]}
        for h in _find_headings(post.content)
    ]

    return {
        "note_path": note_path,
        "uri": f"vault://{note_path}",
        "title": title,
        "tags": tags,
        "aliases": aliases,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "chunk_count": chunk_count,
        "outline": outline,
    }


def _stats_impl(vault_path: Path, store: VectorStore) -> dict[str, Any]:
    s = store.stats()
    return {
        "total_notes": s.total_notes,
        "total_chunks": s.total_chunks,
        "vault_path": str(vault_path),
    }


def _read_resource_impl(vault_root: Path, note_path: str) -> str:
    abs_path = _resolve_safe(vault_root, note_path)
    return abs_path.read_text(encoding="utf-8-sig")


def _resolve_safe(vault_root: Path, note_path: str) -> Path:
    """Resolve ``note_path`` against ``vault_root`` and verify it stays inside.

    Rejects absolute paths and any path that escapes the vault via ``..``.
    Raises FileNotFoundError if the resolved file does not exist.
    """
    # Reject absolute paths. On Windows ``Path("/etc/passwd").is_absolute()``
    # is False, so we also explicitly reject leading "/" and "\".
    if Path(note_path).is_absolute() or note_path.startswith(("/", "\\")):
        raise ValueError(f"Absolute paths are not allowed: {note_path!r}")

    root_resolved = vault_root.resolve()
    candidate = (vault_root / note_path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise ValueError(f"Path escapes vault: {note_path!r}")

    if not candidate.is_file():
        raise FileNotFoundError(f"Note not found: {note_path!r}")
    return candidate


# --- MCP wiring ---


def create_server(
    config_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7777,
) -> FastMCP:
    cfg = load_config(config_path)
    # MCP server forces CPU; cuda DLL import exceeds stdio tool-call timeout
    # on cold start. Bulk indexing (`my-wiki index`) keeps using cfg device.
    embedder = BgeM3Embedder(
        model_name=cfg.embedding.model,
        device="cpu",
        batch_size=cfg.embedding.batch_size,
    )
    if cfg.storage.backend != "chroma":
        raise ValueError(
            f"Unsupported storage backend: {cfg.storage.backend!r}"
        )
    store = ChromaStore(
        persist_path=str(cfg.storage.chroma_path),
        collection=cfg.storage.collection,
    )

    # host/port only matter for `sse` and `streamable-http` transports;
    # they are ignored when running over stdio.
    mcp = FastMCP("be-my-wiki", host=host, port=port)

    hint = (cfg.mcp.description_hint or "").strip()
    hint_suffix = f"\n\nDomain hint: {hint}" if hint else ""

    @mcp.tool(
        description=(
            "Search the user's personal Obsidian wiki / vault / knowledge base "
            "(be_my_wiki) by natural-language query. "
            "Use this whenever the user asks about content in their own notes, "
            "vault, wiki, second brain, or knowledge base — for example "
            "'what did I write about X', 'find my notes on Y', "
            "'내 노트에서 Z 찾아줘', '내가 정리해둔 ... 알려줘'. "
            "Performs semantic similarity search over markdown chunks "
            "(Korean + English supported, math/TeX preserved). "
            "Returns top_k hits each with a vault:// URI usable for full-content "
            "fetch via the resource handler. Optional filters: `tags` (OR-match), "
            "`path_prefix` (directory-boundary)."
            + hint_suffix
        )
    )
    def search(
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
        path_prefix: str | None = None,
    ) -> dict[str, Any]:
        return _search_impl(
            embedder, store, query,
            top_k=top_k, tags=tags, path_prefix=path_prefix,
        )

    @mcp.tool(
        description=(
            "Fetch metadata, headings outline, and a vault:// URI for a single "
            "note in the user's Obsidian wiki / vault / knowledge base "
            "(be_my_wiki). Use after `search` when you need to decide whether "
            "to read full content, or when the user names a specific note. "
            "Returns: title, tags, aliases, size_bytes, modified_at, "
            "chunk_count, outline (heading tree). Does NOT return full body — "
            "fetch the vault:// URI via the resource handler if needed."
            + hint_suffix
        )
    )
    def get_note(note_path: str) -> dict[str, Any]:
        return _get_note_impl(cfg.vault.path, store, note_path)

    @mcp.tool(
        description=(
            "Index statistics for the user's Obsidian wiki / vault / "
            "knowledge base (be_my_wiki): total notes, total chunks, vault "
            "path. Use when the user asks 'how many notes', 'index status', "
            "'vault 통계' or similar."
            + hint_suffix
        )
    )
    def stats() -> dict[str, Any]:
        return _stats_impl(cfg.vault.path, store)

    @mcp.resource("vault://{note_path}", mime_type="text/markdown")
    def read_note(note_path: str) -> str:
        """Read the full markdown content of a vault note."""
        return _read_resource_impl(cfg.vault.path, note_path)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="my-wiki-mcp",
        description="be_my_wiki MCP server (stdio by default; --transport "
                    "streamable-http for HTTP clients like Claude Desktop's "
                    "custom connectors, which require HTTPS).",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("config.toml"),
        help="Path to config.toml (default: ./config.toml)",
    )
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for http/sse transports (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7777,
        help="Bind port for http/sse transports (default: 7777).",
    )
    parser.add_argument(
        "--ssl-cert",
        type=Path,
        default=None,
        help="Path to TLS certificate (PEM). Enables HTTPS for http/sse.",
    )
    parser.add_argument(
        "--ssl-key",
        type=Path,
        default=None,
        help="Path to TLS private key (PEM). Required with --ssl-cert.",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        if args.ssl_cert or args.ssl_key:
            parser.error(
                "--ssl-cert / --ssl-key only apply to http/sse transports"
            )
        create_server(args.config).run()
        return

    if bool(args.ssl_cert) != bool(args.ssl_key):
        parser.error("--ssl-cert and --ssl-key must be provided together")

    server = create_server(args.config, host=args.host, port=args.port)
    endpoint = (
        server.settings.streamable_http_path
        if args.transport == "streamable-http"
        else server.settings.sse_path
    )
    scheme = "https" if args.ssl_cert else "http"
    print(
        f"be-my-wiki MCP listening on {scheme}://{args.host}:{args.port}{endpoint}"
        f"  (transport={args.transport}{', TLS=on' if args.ssl_cert else ''})",
        flush=True,
    )

    if args.ssl_cert:
        # Run uvicorn directly so we can hand it the SSL cert/key. FastMCP's
        # built-in ``run()`` does not expose those parameters.
        import uvicorn

        app = (
            server.streamable_http_app()
            if args.transport == "streamable-http"
            else server.sse_app()
        )
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ssl_keyfile=str(args.ssl_key),
            ssl_certfile=str(args.ssl_cert),
            log_level="info",
        )
    else:
        server.run(transport=args.transport)


if __name__ == "__main__":
    main()
