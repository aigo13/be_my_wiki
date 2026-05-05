"""Tests for the MCP server.

Covers the pure ``_*_impl`` helpers directly (no MCP protocol required)
and one smoke test that exercises ``create_server`` with the sentence-
transformers fake from conftest so no real model is loaded.
"""

import textwrap
import uuid
from pathlib import Path

import pytest

from be_my_wiki.indexer.pipeline import Indexer
from be_my_wiki.mcp_server.server import (
    _get_note_impl,
    _read_resource_impl,
    _resolve_safe,
    _search_impl,
    _stats_impl,
    create_server,
)
from be_my_wiki.store.chroma import ChromaStore

from tests._fakes import FakeEmbedder


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    return v


@pytest.fixture
def setup(vault):
    embedder = FakeEmbedder()
    store = ChromaStore(collection=f"mcp_{uuid.uuid4().hex[:8]}")
    indexer = Indexer(vault_root=vault, embedder=embedder, store=store)
    return embedder, store, indexer


def _write(vault: Path, name: str, content: str) -> Path:
    p = vault / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# --- _resolve_safe ---


def test_resolve_safe_accepts_relative_inside_vault(vault):
    note = _write(vault, "n.md", "x")
    assert _resolve_safe(vault, "n.md") == note.resolve()


def test_resolve_safe_rejects_absolute(vault):
    with pytest.raises(ValueError, match="Absolute"):
        _resolve_safe(vault, "/etc/passwd")


def test_resolve_safe_rejects_traversal(vault):
    with pytest.raises(ValueError, match="escapes vault"):
        _resolve_safe(vault, "../outside.md")


def test_resolve_safe_missing_file(vault):
    with pytest.raises(FileNotFoundError):
        _resolve_safe(vault, "nope.md")


# --- _search_impl ---


def test_search_impl_returns_hits(vault, setup):
    embedder, store, indexer = setup
    _write(vault, "n.md", "## A\nalpha")
    indexer.index_directory()

    result = _search_impl(embedder, store, "alpha", top_k=5)
    assert "hits" in result
    assert len(result["hits"]) == 1
    h = result["hits"][0]
    assert h["note_path"] == "n.md"
    assert h["uri"] == "vault://n.md"
    assert h["chunk_index"] == 0
    assert h["heading_path"] == ["A"]
    assert "snippet" in h and "score" in h
    assert h["title"] == "n"  # filename fallback
    assert h["tags"] == []


def test_search_impl_with_tag_filter(vault, setup):
    embedder, store, indexer = setup
    _write(vault, "ml.md", "---\ntags: [ml]\n---\n## A\nalpha")
    _write(vault, "other.md", "---\ntags: [other]\n---\n## A\nbeta")
    indexer.index_directory()

    result = _search_impl(embedder, store, "anything", top_k=5, tags=["ml"])
    paths = {h["note_path"] for h in result["hits"]}
    assert paths == {"ml.md"}


def test_search_impl_with_path_prefix(vault, setup):
    embedder, store, indexer = setup
    _write(vault, "ML/a.md", "## A\nalpha")
    _write(vault, "Daily/b.md", "## B\nbeta")
    indexer.index_directory()

    result = _search_impl(embedder, store, "anything", top_k=5, path_prefix="ML")
    paths = {h["note_path"] for h in result["hits"]}
    assert paths == {"ML/a.md"}


def test_search_impl_raises_on_empty_query(setup):
    embedder, store, _ = setup
    with pytest.raises(ValueError, match="empty"):
        _search_impl(embedder, store, "")
    with pytest.raises(ValueError, match="empty"):
        _search_impl(embedder, store, "   ")


def test_search_impl_raises_on_zero_top_k(setup):
    embedder, store, _ = setup
    with pytest.raises(ValueError, match="> 0"):
        _search_impl(embedder, store, "x", top_k=0)


# --- _get_note_impl ---


def test_get_note_impl_returns_full_metadata(vault, setup):
    _, store, indexer = setup
    _write(
        vault,
        "n.md",
        "---\ntitle: My Note\ntags: [ml, project]\naliases: [alt]\n---\n\n# Top\nintro\n\n## Sub\nmore",
    )
    indexer.index_directory()

    info = _get_note_impl(vault, store, "n.md")
    assert info["note_path"] == "n.md"
    assert info["uri"] == "vault://n.md"
    assert info["title"] == "My Note"
    assert info["tags"] == ["ml", "project"]
    assert info["aliases"] == ["alt"]
    assert info["size_bytes"] > 0
    assert "modified_at" in info
    assert info["chunk_count"] >= 1
    outline = info["outline"]
    assert {"level": 1, "text": "Top"} in outline
    assert {"level": 2, "text": "Sub"} in outline


def test_get_note_impl_title_falls_back_to_filename(vault, setup):
    _, store, indexer = setup
    _write(vault, "MyNote.md", "## A\nbody")
    indexer.index_directory()
    info = _get_note_impl(vault, store, "MyNote.md")
    assert info["title"] == "MyNote"


def test_get_note_impl_path_traversal_rejected(vault, setup):
    _, store, _ = setup
    with pytest.raises(ValueError, match="escapes vault"):
        _get_note_impl(vault, store, "../outside.md")


def test_get_note_impl_absolute_path_rejected(vault, setup):
    _, store, _ = setup
    with pytest.raises(ValueError, match="Absolute"):
        _get_note_impl(vault, store, "/etc/passwd")


def test_get_note_impl_missing_file_raises(vault, setup):
    _, store, _ = setup
    with pytest.raises(FileNotFoundError):
        _get_note_impl(vault, store, "nope.md")


# --- _stats_impl ---


def test_stats_impl(vault, setup):
    _, store, indexer = setup
    _write(vault, "a.md", "## A\nx")
    _write(vault, "b.md", "## B\ny")
    indexer.index_directory()

    info = _stats_impl(vault, store)
    assert info["total_notes"] == 2
    assert info["total_chunks"] == 2
    assert info["vault_path"] == str(vault)


# --- _read_resource_impl ---


def test_read_resource_impl_returns_content(vault):
    _write(vault, "n.md", "## Section\nbody content")
    content = _read_resource_impl(vault, "n.md")
    assert "## Section" in content
    assert "body content" in content


def test_read_resource_impl_traversal_rejected(vault):
    with pytest.raises(ValueError, match="escapes vault"):
        _read_resource_impl(vault, "../outside.md")


def test_read_resource_impl_missing_raises(vault):
    with pytest.raises(FileNotFoundError):
        _read_resource_impl(vault, "nope.md")


# --- create_server smoke ---


def test_create_server_returns_fastmcp(tmp_path, fake_st):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            [vault]
            path = "{vault_dir.as_posix()}"

            [storage]
            chroma_path = "{(tmp_path / "chroma").as_posix()}"
            collection = "mcp_smoke"
            """
        ),
        encoding="utf-8",
    )
    server = create_server(cfg_path)
    assert server is not None
    assert server.name == "be-my-wiki"
    # Default host/port for HTTP/SSE transports.
    assert server.settings.host == "127.0.0.1"
    assert server.settings.port == 7777


async def _list_tools(server):
    return await server.list_tools()


def test_tool_descriptions_include_discovery_keywords(tmp_path, fake_st):
    import asyncio

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            [vault]
            path = "{vault_dir.as_posix()}"

            [storage]
            chroma_path = "{(tmp_path / "chroma").as_posix()}"
            collection = "keywords"
            """
        ),
        encoding="utf-8",
    )
    server = create_server(cfg_path)
    tools = {t.name: t for t in asyncio.run(_list_tools(server))}

    # Common discovery keywords appear in at least one tool description.
    blob = " ".join(t.description for t in tools.values()).lower()
    for kw in ("obsidian", "vault", "wiki", "knowledge base", "내 노트"):
        assert kw in blob, f"missing keyword in tool descriptions: {kw}"


def test_description_hint_propagates_to_tool_descriptions(tmp_path, fake_st):
    import asyncio

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            [vault]
            path = "{vault_dir.as_posix()}"

            [storage]
            chroma_path = "{(tmp_path / "chroma").as_posix()}"
            collection = "hint_test"

            [mcp]
            description_hint = "Quantum Computing Theory wiki — algorithms, linear algebra"
            """
        ),
        encoding="utf-8",
    )
    server = create_server(cfg_path)
    tools = asyncio.run(_list_tools(server))
    for t in tools:
        assert "Quantum Computing Theory wiki" in t.description, (
            f"hint missing from tool {t.name}"
        )


def test_create_server_forces_cpu_even_with_cuda_config(
    tmp_path, fake_st, monkeypatch
):
    """MCP server must build the embedder with device='cpu' even when the
    config says cuda — cuda DLL import on cold start exceeds the MCP stdio
    tool-call timeout on Windows. Bulk indexing keeps using cfg device.
    """
    from be_my_wiki.mcp_server import server as server_module

    captured: dict = {}
    real_cls = server_module.BgeM3Embedder

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(server_module, "BgeM3Embedder", spy)

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            [vault]
            path = "{vault_dir.as_posix()}"

            [storage]
            chroma_path = "{(tmp_path / "chroma").as_posix()}"
            collection = "force_cpu"

            [embedding]
            device = "cuda"
            """
        ),
        encoding="utf-8",
    )
    create_server(cfg_path)
    assert captured.get("device") == "cpu"


def test_create_server_accepts_custom_host_port(tmp_path, fake_st):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            [vault]
            path = "{vault_dir.as_posix()}"

            [storage]
            chroma_path = "{(tmp_path / "chroma").as_posix()}"
            collection = "mcp_http"
            """
        ),
        encoding="utf-8",
    )
    server = create_server(cfg_path, host="0.0.0.0", port=9000)
    assert server.settings.host == "0.0.0.0"
    assert server.settings.port == 9000
    assert server.settings.streamable_http_path == "/mcp"


# --- multilingual + TeX coverage (per project policy) ---


def test_korean_note_full_flow(vault, setup):
    embedder, store, indexer = setup
    _write(
        vault,
        "ko.md",
        "---\ntags: [한국어]\n---\n\n## 서론\n안녕하세요. 이것은 한글 본문입니다.",
    )
    indexer.index_directory()

    sresult = _search_impl(embedder, store, "안녕", top_k=5)
    assert any(h["note_path"] == "ko.md" for h in sresult["hits"])

    info = _get_note_impl(vault, store, "ko.md")
    assert info["tags"] == ["한국어"]
    assert {"level": 2, "text": "서론"} in info["outline"]

    content = _read_resource_impl(vault, "ko.md")
    assert "안녕하세요" in content


def test_search_with_korean_tag_filter(vault, setup):
    embedder, store, indexer = setup
    _write(vault, "ko.md", "---\ntags: [한국어]\n---\n\n## A\nalpha")
    _write(vault, "en.md", "---\ntags: [english]\n---\n\n## A\nalpha")
    indexer.index_directory()

    result = _search_impl(
        embedder, store, "alpha", top_k=5, tags=["한국어"]
    )
    assert {h["note_path"] for h in result["hits"]} == {"ko.md"}


def test_tex_note_outline_and_resource(vault, setup):
    _, store, indexer = setup
    _write(
        vault,
        "math.md",
        "## Integral\n\n$$\n\\int_0^1 x \\, dx = \\frac{1}{2}\n$$",
    )
    indexer.index_directory()

    info = _get_note_impl(vault, store, "math.md")
    assert {"level": 2, "text": "Integral"} in info["outline"]

    content = _read_resource_impl(vault, "math.md")
    assert r"\int_0^1 x \, dx" in content
    assert r"\frac{1}{2}" in content
