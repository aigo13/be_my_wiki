import textwrap

import pytest

from be_my_wiki.config import Config, load_config


def _write(path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_load_minimal_config(tmp_path):
    cfg_path = tmp_path / "config.toml"
    _write(
        cfg_path,
        """
        [vault]
        path = "/abs/path/vault"
        """,
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    # Defaults filled in
    assert cfg.storage.backend == "chroma"
    assert cfg.storage.collection == "vault"
    assert cfg.embedding.model == "BAAI/bge-m3"
    assert cfg.embedding.device == "cpu"
    assert cfg.embedding.batch_size == 16
    assert cfg.chunking.max_tokens == 512
    assert cfg.chunking.heading_level == 2
    assert cfg.indexer.debounce_seconds == 2.0
    assert cfg.mcp.default_top_k == 5


def test_load_full_config(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    cfg_path = tmp_path / "config.toml"
    _write(
        cfg_path,
        f"""
        [vault]
        path = "{vault.as_posix()}"
        ignore = [".obsidian/**", ".trash/**"]

        [storage]
        backend = "chroma"
        chroma_path = "./data/chroma"
        collection = "myvault"

        [embedding]
        model = "BAAI/bge-m3"
        device = "cuda"
        batch_size = 32

        [chunking]
        max_tokens = 256
        heading_level = 3

        [indexer]
        debounce_seconds = 0.5

        [mcp]
        default_top_k = 10
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.vault.path == vault.resolve()
    assert cfg.vault.ignore == [".obsidian/**", ".trash/**"]
    assert cfg.storage.collection == "myvault"
    assert cfg.embedding.device == "cuda"
    assert cfg.embedding.batch_size == 32
    assert cfg.chunking.max_tokens == 256
    assert cfg.chunking.heading_level == 3
    assert cfg.indexer.debounce_seconds == 0.5
    assert cfg.mcp.default_top_k == 10


def test_relative_paths_resolve_from_config_dir(tmp_path):
    cfg_path = tmp_path / "config.toml"
    _write(
        cfg_path,
        """
        [vault]
        path = "myvault"

        [storage]
        chroma_path = "data/chroma"
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.vault.path == (tmp_path / "myvault").resolve()
    assert cfg.storage.chroma_path == (tmp_path / "data" / "chroma").resolve()


def test_absolute_paths_left_alone(tmp_path):
    abs_vault = (tmp_path / "abs").resolve()
    cfg_path = tmp_path / "config.toml"
    _write(
        cfg_path,
        f"""
        [vault]
        path = "{abs_vault.as_posix()}"
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.vault.path == abs_vault


def test_missing_required_field_raises(tmp_path):
    cfg_path = tmp_path / "config.toml"
    _write(
        cfg_path,
        """
        [storage]
        collection = "x"
        """,
    )
    with pytest.raises(Exception):
        load_config(cfg_path)


def test_invalid_heading_level_raises(tmp_path):
    cfg_path = tmp_path / "config.toml"
    _write(
        cfg_path,
        """
        [vault]
        path = "/tmp/v"

        [chunking]
        heading_level = 7
        """,
    )
    with pytest.raises(Exception):
        load_config(cfg_path)


def test_invalid_batch_size_raises(tmp_path):
    cfg_path = tmp_path / "config.toml"
    _write(
        cfg_path,
        """
        [vault]
        path = "/tmp/v"

        [embedding]
        batch_size = 0
        """,
    )
    with pytest.raises(Exception):
        load_config(cfg_path)


def test_unknown_backend_passes_pydantic_but_typed(tmp_path):
    # backend is Literal["chroma"]; anything else fails validation.
    cfg_path = tmp_path / "config.toml"
    _write(
        cfg_path,
        """
        [vault]
        path = "/tmp/v"

        [storage]
        backend = "qdrant"
        """,
    )
    with pytest.raises(Exception):
        load_config(cfg_path)


# --- multilingual + TeX coverage (per project policy) ---


def test_korean_path_in_config(tmp_path):
    vault = tmp_path / "내 노트"
    vault.mkdir()
    cfg_path = tmp_path / "config.toml"
    _write(
        cfg_path,
        f"""
        [vault]
        path = "{vault.as_posix()}"
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.vault.path == vault.resolve()


def test_path_with_tex_like_chars(tmp_path):
    # TeX-flavored special characters in paths. Combines CJK + $ symbol.
    vault = tmp_path / "수식$노트"
    vault.mkdir()
    cfg_path = tmp_path / "config.toml"
    _write(
        cfg_path,
        f"""
        [vault]
        path = "{vault.as_posix()}"
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.vault.path == vault.resolve()
