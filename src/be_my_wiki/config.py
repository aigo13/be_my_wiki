"""Configuration loading.

Reads ``config.toml`` and validates it against a Pydantic schema. Relative
paths inside the config are resolved against the directory of the config
file (so a config at ``./config.toml`` with ``chroma_path = "data/chroma"``
points to ``./data/chroma``, regardless of cwd).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class VaultConfig(BaseModel):
    path: Path
    ignore: list[str] = Field(default_factory=list)


class StorageConfig(BaseModel):
    backend: Literal["chroma"] = "chroma"
    chroma_path: Path = Path("./data/chroma")
    collection: str = "vault"


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = Field(default=16, gt=0)


class ChunkingConfig(BaseModel):
    max_tokens: int = Field(default=512, gt=0)
    heading_level: int = Field(default=2, ge=1, le=6)


class IndexerConfig(BaseModel):
    debounce_seconds: float = Field(default=2.0, gt=0)


class McpConfig(BaseModel):
    default_top_k: int = Field(default=5, gt=0)
    # Free-form text appended to each MCP tool's description. Useful for
    # making the tools easier to discover via Claude's tool search — list
    # domain keywords your vault contains (e.g. "Quantum Computing Theory,
    # quantum algorithms, linear algebra"). Leave empty to use only the
    # generic descriptions.
    description_hint: str = ""


class Config(BaseModel):
    vault: VaultConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)


def load_config(config_path: Path) -> Config:
    """Load and validate config from a TOML file.

    Relative paths in the config are resolved against the config file's
    parent directory.
    """
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    cfg = Config.model_validate(data)

    base = config_path.parent
    if not cfg.vault.path.is_absolute():
        cfg.vault.path = (base / cfg.vault.path).resolve()
    if not cfg.storage.chroma_path.is_absolute():
        cfg.storage.chroma_path = (base / cfg.storage.chroma_path).resolve()
    return cfg
