"""Chunk Obsidian markdown notes for embedding (A2 hybrid strategy).

For each note we:
1. Strip YAML frontmatter and capture it as metadata.
2. Split the body at markdown headings up to a configured level (default H2).
3. Sub-split sections that exceed max_tokens by paragraph (or hard char-cap).
4. Prepend a structural prefix (title, tags, heading path) to each chunk's
   embeddable text so frontmatter influences semantic similarity.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
from markdown_it import MarkdownIt

logger = logging.getLogger(__name__)


# bge-m3 uses an XLM-RoBERTa tokenizer. ~3 chars/token is a conservative
# estimate for mixed Korean/English text. We trade precision for keeping
# the chunker free of model-loading dependencies at runtime.
_CHARS_PER_TOKEN = 3


@dataclass(frozen=True)
class Chunk:
    note_path: str
    chunk_index: int
    text: str                       # prefix + body, embedded as-is
    body: str                       # original body fragment, for snippets
    heading_path: tuple[str, ...]
    metadata: dict[str, Any]
    content_hash: str               # sha256 of `text`


def chunk_note(
    *,
    note_path: str,
    source: str,
    max_tokens: int = 512,
    heading_level: int = 2,
) -> list[Chunk]:
    """Parse a markdown note and return its chunks.

    `note_path` is used for the title fallback and as `Chunk.note_path`.
    `source` is the raw file content (frontmatter + body).
    """
    fm, body = _safe_load_frontmatter(source, note_path=note_path)

    if not body.strip():
        return []

    title = _resolve_title(fm, note_path)
    tags = _normalize_list(fm.get("tags"))
    aliases = _normalize_list(fm.get("aliases"))

    sections = _split_by_heading(body, heading_level)
    char_cap = max_tokens * _CHARS_PER_TOKEN

    base_metadata: dict[str, Any] = {
        "title": title,
        "tags": tags,
        "aliases": aliases,
    }
    for k, v in fm.items():
        if k not in base_metadata:
            base_metadata[k] = v

    chunks: list[Chunk] = []
    for section in sections:
        for body_fragment in _enforce_size(section.body, char_cap):
            prefix = _build_prefix(
                title=title,
                tags=tags,
                heading_path=section.heading_path,
            )
            text = f"{prefix}\n\n{body_fragment}" if prefix else body_fragment
            chunks.append(
                Chunk(
                    note_path=note_path,
                    chunk_index=len(chunks),
                    text=text,
                    body=body_fragment,
                    heading_path=tuple(section.heading_path),
                    metadata=dict(base_metadata),
                    content_hash=_sha256(text),
                )
            )

    return chunks


# --- internals ---


@dataclass
class _Section:
    heading_path: list[str]
    body: str


def _split_by_heading(body: str, heading_level: int) -> list[_Section]:
    """Split body at every heading whose level <= heading_level.

    Heading detection uses markdown-it-py so fenced code blocks do not
    produce false positives. Sections track the chain of ancestor headings
    in heading_path.
    """
    headings = _find_headings(body)
    split_points = [h for h in headings if h["level"] <= heading_level]
    lines = body.split("\n")

    sections: list[_Section] = []

    pre_end = split_points[0]["line_start"] if split_points else len(lines)
    pre_body = "\n".join(lines[:pre_end]).strip()
    if pre_body:
        sections.append(_Section(heading_path=[], body=pre_body))

    stack: list[tuple[int, str]] = []
    for i, sp in enumerate(split_points):
        while stack and stack[-1][0] >= sp["level"]:
            stack.pop()
        stack.append((sp["level"], sp["text"]))

        body_start = sp["line_end"]
        body_end = (
            split_points[i + 1]["line_start"]
            if i + 1 < len(split_points)
            else len(lines)
        )
        section_body = "\n".join(lines[body_start:body_end]).strip()
        if section_body:
            sections.append(
                _Section(
                    heading_path=[text for _, text in stack],
                    body=section_body,
                )
            )

    return sections


def _find_headings(body: str) -> list[dict[str, Any]]:
    md = MarkdownIt()
    tokens = md.parse(body)
    out: list[dict[str, Any]] = []
    for i, tok in enumerate(tokens):
        if tok.type != "heading_open":
            continue
        level = int(tok.tag[1])
        inline = tokens[i + 1] if i + 1 < len(tokens) else None
        text = inline.content.strip() if inline else ""
        if tok.map is None:
            continue
        line_start, line_end = tok.map
        out.append(
            {
                "line_start": line_start,
                "line_end": line_end,
                "level": level,
                "text": text,
            }
        )
    return out


def _enforce_size(body: str, char_cap: int) -> list[str]:
    body = body.strip()
    if not body:
        return []
    if len(body) <= char_cap:
        return [body]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for p in paragraphs:
        if len(p) > char_cap:
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer, buffer_len = [], 0
            for i in range(0, len(p), char_cap):
                chunks.append(p[i : i + char_cap])
        elif buffer and buffer_len + len(p) + 2 > char_cap:
            chunks.append("\n\n".join(buffer))
            buffer = [p]
            buffer_len = len(p)
        else:
            buffer.append(p)
            buffer_len += len(p) + 2

    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def _build_prefix(
    *,
    title: str,
    tags: list[str],
    heading_path: list[str],
) -> str:
    parts = [f"Title: {title}"]
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    if heading_path:
        parts.append(f"Section: {' > '.join(heading_path)}")
    return "\n".join(parts)


def _resolve_title(fm: dict[str, Any], note_path: str) -> str:
    raw = fm.get("title")
    if raw:
        return str(raw).strip()
    return Path(note_path).stem


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in re.split(r"[,\s]+", value) if s.strip()]
    if isinstance(value, list):
        return [str(s).strip() for s in value if str(s) and str(s).strip()]
    return []


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_load_frontmatter(
    source: str,
    *,
    note_path: str = "",
) -> tuple[dict[str, Any], str]:
    """Parse frontmatter; on YAML error, strip the block and return empty meta.

    Real-world Obsidian vaults sometimes have unquoted frontmatter values
    that include math notation (``{x : f(x)=0}``, ``T : V -> W``, etc.).
    Those break YAML parsing. Rather than dropping the whole note from the
    index we log a warning and fall back to indexing the body alone.
    """
    try:
        post = frontmatter.loads(source)
    except Exception as exc:
        logger.warning(
            "frontmatter parse failed for %s (%s); indexing body only",
            note_path or "<unknown>",
            exc,
        )
        return {}, _strip_frontmatter_block(source)
    fm = dict(post.metadata) if post.metadata else {}
    return fm, post.content


def _strip_frontmatter_block(source: str) -> str:
    """Remove a leading ``---\\n...\\n---\\n`` block, if any."""
    if not source.startswith("---"):
        return source
    # Look for the closing delimiter on its own line.
    after_open = source[3:].lstrip("\r\n")
    end_marker = re.search(r"^---\s*$", after_open, flags=re.MULTILINE)
    if not end_marker:
        return source
    cutoff = (len(source) - len(after_open)) + end_marker.end()
    return source[cutoff:].lstrip("\r\n")
