# be_my_wiki

A semantic-search **MCP server** for Obsidian-style markdown vaults. Index
your notes with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) embeddings,
store them in ChromaDB, and query by natural language from
[Claude Desktop](https://claude.ai/download) or [Claude Code](https://claude.com/claude-code)
through the [Model Context Protocol](https://modelcontextprotocol.io/).

> **한국어 안내**
> 이 프로젝트는 Obsidian (또는 비슷한 markdown 기반) vault를 자연어로 시맨틱
> 검색하기 위한 MCP 서버입니다. bge-m3 임베딩 + ChromaDB 벡터 저장소로
> 인덱스를 만들고, Claude Desktop 및 Claude Code에서 도구 호출로 검색할 수
> 있게 노출합니다. README는 영어가 메인이지만 핵심 단계마다 한국어 보조
> 설명을 함께 둡니다.

---

## Features

- 🌐 **Multilingual** — bge-m3 covers 100+ languages; Korean and English
  cross-lingual search works out of the box.
- 🧮 **Math/TeX preserved** — chunker doesn't split inside `$...$` or
  `$$...$$` blocks; YAML frontmatter with math notation is handled
  gracefully (falls back to body-only when frontmatter parse fails).
- 🔎 **Native filter pushdown** — Chroma `where`-clause filters by tag
  (OR-match) and path-prefix (directory boundary).
- ⚡ **Incremental indexing** — only changed chunks are re-embedded
  (sha256 of `prefix + body`).
- 👀 **Filesystem watcher** — `my-wiki watch` re-indexes on file changes
  with event debouncing.
- 🔌 **Two MCP transports** — stdio (Claude Code, MCPB packaging) and
  HTTPS / streamable-http (Claude Desktop Custom Connectors).
- 📦 **MCPB-friendly** — ships an example manifest for Claude Desktop's
  `.mcpb` packaging.

> **한국어**: 한국어 / 영어 cross-lingual 검색, LaTeX 수식 보존, 태그·폴더
> 필터, 변경분만 재임베딩, 파일 변경 자동 감지, Claude Desktop / Code 양쪽
> 등록 지원.

---

## Quickstart

(Assumes Python 3.12 and [uv](https://docs.astral.sh/uv/) installed.)

```bash
git clone https://github.com/aigo13/be_my_wiki.git
cd be_my_wiki
uv venv --python 3.12.3
uv pip install -e ".[dev]"

# (Optional) GPU? Replace torch with a CUDA build:
# uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128

cp config.example.toml config.toml
# Edit config.toml: set [vault] path to your vault directory

my-wiki index            # First run downloads bge-m3 (~2.3 GB) then embeds
my-wiki search "your query" -k 5
```

Then register the server with Claude Code or Claude Desktop — see
[Claude integration](#claude-integration).

---

## Requirements

- **Python 3.12** (3.13 supported; 3.14 not yet — bge-m3's deps lag).
- **~3 GB free disk** for the bge-m3 model on first index. Cached at
  `~/.cache/huggingface/hub/models--BAAI--bge-m3`.
- **(Optional) NVIDIA GPU** + CUDA 12.x or 13.x driver for ~10–20× faster
  indexing. CPU works fine for small/medium vaults.
- **Tested on Windows 11**. macOS / Linux should work but are unverified;
  bug reports welcome.

---

## Step-by-step setup

### 1. Install Python with `uv`

[uv](https://docs.astral.sh/uv/) is a fast Python package manager from
Astral that can also manage Python interpreter versions per project.

```bash
# Install uv if you don't have it (Windows)
winget install astral-sh.uv
# or:  irm https://astral.sh/uv/install.ps1 | iex

# Install Python 3.12.3 if needed
uv python install 3.12.3
```

> **한국어**: 시스템 Python과 충돌 없이 프로젝트별로 정확한 버전을 쓰려면
> uv가 깔끔합니다. uv 설치 후 위 명령으로 Python 3.12.3을 자동 다운로드.

### 2. Clone and create a virtual environment

```bash
git clone https://github.com/aigo13/be_my_wiki.git
cd be_my_wiki

# Default: venv lives at ./.venv
uv venv --python 3.12.3

# Or store the venv elsewhere (useful if you keep all venvs in one place)
# uv venv /path/to/your/venvs/be_my_wiki --python 3.12.3
```

### 3. Install the package and dependencies

```bash
uv pip install -e ".[dev]"
```

This installs runtime deps (`mcp`, `chromadb`, `sentence-transformers`,
`watchdog`, `markdown-it-py`, `python-frontmatter`, `pydantic`, `typer`),
dev tools (`pytest`, `ruff`), and registers the CLI entry points
(`my-wiki`, `my-wiki-mcp`) inside the venv.

PyTorch is pulled in by `sentence-transformers` as the **CPU-only** wheel
by default (~750 MB).

### 4. (Optional) Switch PyTorch to a CUDA build

```bash
# Example: CUDA 12.8 (works with CUDA 12.x or 13.x driver — backward-compat)
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

Pick the right index URL for your CUDA toolkit (`cu124`, `cu126`, `cu128`).
Verify:

```bash
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"
# Should print True if CUDA wheel + driver are compatible
```

> **한국어**: GPU 없으면 이 단계 건너뛰면 됩니다. 미니 vault에선 CPU도 문제
> 없고, 첫 다운로드만 끝나면 임베딩 자체는 빠릅니다.

### 5. Configure your vault

```bash
cp config.example.toml config.toml
```

Open `config.toml` and edit at least the vault path. Use forward slashes
on Windows (TOML strings):

```toml
[vault]
path = "D:/Notes/MyVault"
ignore = [".obsidian/**", ".trash/**", "**/node_modules/**"]

[storage]
backend = "chroma"
chroma_path = "./data/chroma"
collection = "vault"

[embedding]
model = "BAAI/bge-m3"
device = "cpu"          # or "cuda"
batch_size = 16

[chunking]
max_tokens = 512
heading_level = 2

[mcp]
default_top_k = 5
# Optional: domain keywords appended to MCP tool descriptions, helps
# Claude's tool-search match the right server. Example:
# description_hint = "Quantum Computing wiki — algorithms, linear algebra, 양자 컴퓨팅"
```

> **한국어**: `[vault] path` 만 본인 vault로 바꾸면 일단 동작합니다. Obsidian
> 내부 파일이 들어있는 `.obsidian/` 디렉터리는 기본으로 무시합니다. 나머지는
> 필요할 때 조정.

### 6. First index (downloads bge-m3)

```bash
my-wiki index
```

The first run downloads bge-m3 (~2.3 GB) into the HuggingFace cache, then
walks your vault and embeds every chunk. Watch for output like:

```
Indexing vault: D:/Notes/MyVault
Notes: total=117 changed=117 unchanged=0 failed=0 pruned=0
Chunks: added=625 updated=0 skipped=0 deleted=0
```

Re-running is incremental — only changed chunks are re-embedded:

```
Notes: total=117 changed=1 unchanged=116 failed=0 pruned=0
Chunks: added=2 updated=0 skipped=623 deleted=1
```

### 7. Smoke test from the terminal

```bash
my-wiki stats
my-wiki search "your query here" -k 5
```

### 8. (Optional) Auto-sync as you edit

```bash
my-wiki watch
```

Runs an initial index, then keeps watching the vault for file changes
(debounced, 2 seconds by default) and re-indexes incrementally.

---

## Claude integration

`be_my_wiki` is meant to be driven by an LLM through MCP. Claude Code and
Claude Desktop are the most common clients today; both run their MCP
servers as **stdio subprocesses** behind the scenes. Just the registration
mechanism differs.

### Claude Code (stdio)

Register with **user scope** so it's available across all projects:

```bash
# Replace paths with your absolute paths.
# On Windows the binary is .venv/Scripts/my-wiki-mcp.exe
claude mcp add --scope user --transport stdio be-my-wiki -- \
    /absolute/path/to/.venv/bin/my-wiki-mcp \
    --config /absolute/path/to/config.toml
```

Verify:

```bash
claude mcp list
# Look for: be-my-wiki   stdio   ...my-wiki-mcp ...
```

Open a new Claude Code session and ask something like *"내 노트에서 X 찾아줘"* —
Claude should call `mcp__be_my_wiki__search` (or `__get_chunk` / `__get_note`).

### Claude Desktop (MCPB)

Recent Claude Desktop manages MCP servers through a centralized DXT
registry, so direct edits to `claude_desktop_config.json` get rewritten on
launch. The supported path for **local** stdio servers is **MCPB
(Desktop Extensions)**, which packages a manifest (and optionally code) as
a `.mcpb` zip file.

Create a directory `mcpb/` next to your repo (or anywhere) with a single
`manifest.json`:

```json
{
  "manifest_version": "0.3",
  "name": "be-my-wiki",
  "display_name": "be_my_wiki",
  "version": "0.1.0",
  "description": "Semantic search over a local Obsidian vault.",
  "author": {"name": "Your Name"},
  "server": {
    "type": "binary",
    "entry_point": "/absolute/path/to/.venv/bin/my-wiki-mcp",
    "mcp_config": {
      "command": "/absolute/path/to/.venv/bin/my-wiki-mcp",
      "args": ["--config", "/absolute/path/to/config.toml"],
      "env": {}
    }
  },
  "tools": [
    {"name": "search", "description": "Semantic search over your vault."},
    {"name": "get_chunk", "description": "Fetch one chunk's full body by (note_path, chunk_index)."},
    {"name": "get_note", "description": "Note metadata, outline, and chunks index."},
    {"name": "stats", "description": "Index statistics."}
  ]
}
```

Pack it into a `.mcpb` (a zip with a different extension):

```bash
cd path/to/mcpb-dir
python -c "import zipfile; zipfile.ZipFile('be-my-wiki.mcpb','w').write('manifest.json')"
```

Then either **double-click** the `.mcpb`, **drag-drop** it onto Claude
Desktop, or use **Settings → Extensions → Advanced settings → Install
Extension…**. Claude Desktop spawns `my-wiki-mcp` over stdio whenever a
tool is invoked — no terminal needs to stay open.

> **한국어**: `manifest.json`의 두 절대경로(venv binary, config.toml)만
> 본인 환경으로 채우고 zip으로 묶어 `.mcpb`로 저장. Desktop UI에 드래그하면
> 끝.

### Claude Desktop (HTTPS Custom Connector — alternative)

If MCPB isn't available on your plan or you prefer the HTTP path, the
project ships a self-signed cert generator and an HTTPS-capable server.

```bash
# 1. Generate root CA + server cert
my-wiki ssl-init

# 2. Run the printed PowerShell command as admin to install the root CA
#    in Windows trust store. Example:
#    Import-Certificate -FilePath "...\data\certs\be-my-wiki-ca.pem" \
#                       -CertStoreLocation Cert:\LocalMachine\Root

# 3. Start the HTTPS MCP server
my-wiki-mcp --transport streamable-http --port 7777 \
            --config config.toml \
            --ssl-cert data/certs/server.pem \
            --ssl-key  data/certs/server-key.pem
```

Then in Claude Desktop add a Custom Connector with URL
`https://localhost:7777/mcp`. The server process must keep running for
queries to work.

---

## CLI reference

```text
my-wiki index   [--config PATH] [--prune/--no-prune]
my-wiki watch   [--config PATH] [--initial/--no-initial]
my-wiki search  QUERY [--top-k N] [--config PATH]
my-wiki stats   [--config PATH]
my-wiki ssl-init [--out DIR] [--host NAME ...]

my-wiki-mcp [--config PATH]
            [--transport {stdio|sse|streamable-http}]
            [--host HOST] [--port PORT]
            [--ssl-cert FILE --ssl-key FILE]
```

---

## MCP surface

The MCP server exposes four tools and one resource template:

| Kind | Name / URI | Purpose |
|---|---|---|
| Tool | `search(query, top_k=5, tags?, path_prefix?)` | Semantic search; each hit carries the full chunk `body` plus a `vault://` URI |
| Tool | `get_chunk(note_path, chunk_index)` | Fetch the body of one specific chunk (heading-aware section) — saves tokens vs. reading the whole note |
| Tool | `get_note(note_path)` | Metadata, headings `outline`, and `chunks` index (chunk_index + heading_path per indexed chunk) — pair with `get_chunk` to drill in |
| Tool | `stats()` | total_notes, total_chunks, vault_path |
| Resource | `vault://{note_path}` | Full markdown content of a note |

Path traversal is rejected: the server only reads files inside the
configured `vault.path`.

> **Token-saving flow**: `search` already returns the matched chunk's
> full body. For neighbouring/specific chunks, call `get_note` to see
> the chunk index, then `get_chunk` for just that chunk — no need to
> fetch the entire note via `vault://`.

---

## Tips for good search results

- **Use H2 headings** (or whatever you set `chunking.heading_level` to) to
  carve notes into searchable sections. The chunker indexes per-section,
  and section titles flow into the embedding via the `Title:` / `Section:`
  prefix the chunker injects.
- **Tag in YAML frontmatter** (`tags: [foo, bar]`) — these become
  filterable via the `search(... tags=[...])` argument, and also appear
  in the embedded prefix so semantic search can pick them up.
- **Math is fine** — both inline (`$E=mc^2$`) and display (`$$...$$`)
  blocks survive chunking. If frontmatter contains math (e.g.
  `summary: T : V → W`), YAML parsing may fail; the chunker logs a
  warning and indexes the body anyway.
- **Set `description_hint`** in `[mcp]` with your domain keywords — helps
  Claude's tool-search match the right server.

---

## Architecture

```
                  parsing/chunker         embedding/bge_m3
                        │                       │
[markdown notes] ──▶ Chunk(s) ──embed──▶  Vector(s)
                        │                       │
                        └────upsert────▶ store/chroma  (ChromaDB, cosine)
                                              │
                                       VectorStore
                                              │
                       search / get_chunk / get_note / stats
                                              │
                                   mcp_server (stdio | https)
                                              │
                                   Claude Desktop / Code
```

Modules under `src/be_my_wiki/`:

- `parsing/chunker.py` — markdown → chunks; frontmatter + heading-aware,
  prefix-injection (A2 hybrid: title/tags/heading_path become embedded
  text so semantic similarity is metadata-sensitive).
- `embedding/{base,bge_m3}.py` — `Embedder` Protocol + bge-m3 wrapper
  (lazy load, L2 normalized).
- `store/{base,chroma}.py` — `VectorStore` Protocol + ChromaDB backend.
  Filter pushdown via per-tag boolean fields and per-level directory
  fields (so `tags=["ml"]` and `path_prefix="ML/basics"` translate into
  Chroma `where` clauses, not Python post-filters).
- `indexer/pipeline.py` — orchestrates parse → chunk → diff → embed →
  upsert, with content-hash diffing for incremental re-index. Optional
  orphan pruning for files removed from disk.
- `indexer/watcher.py` — watchdog filesystem watcher with worker-thread
  debouncing.
- `mcp_server/server.py` — FastMCP server. Pure `_*_impl` functions for
  testability; transport selectable at startup; SSL via uvicorn for
  HTTPS.
- `cli.py` — typer CLI (`index`, `watch`, `search`, `stats`, `ssl-init`).
- `config.py` — Pydantic 2 schema for `config.toml`.

---

## Development

```bash
uv pip install -e ".[dev]"
pytest tests/        # 130+ unit tests

# Lint / format (optional, ruff is in dev deps)
ruff check src tests
ruff format src tests
```

The test suite includes Korean and TeX coverage cases for every module
that handles content (`tests/test_chunker.py`, `tests/test_store.py`,
`tests/test_indexer.py`, `tests/test_mcp_server.py`, `tests/test_cli.py`).

The bge-m3 model is **not** loaded during the unit tests — a
`fake_st` fixture in `tests/conftest.py` patches
`sentence_transformers.SentenceTransformer` with a deterministic stand-in.
The full model is exercised when you run `my-wiki index` against a real
vault.

---

## Acknowledgements

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — multilingual embedder
- [ChromaDB](https://www.trychroma.com/) — vector storage
- [Model Context Protocol](https://modelcontextprotocol.io/) — the
  protocol Claude uses to talk to local servers
- [FastMCP](https://github.com/jlowin/fastmcp) — high-level Python MCP
  server framework

---

## License

MIT. See [LICENSE](LICENSE).
