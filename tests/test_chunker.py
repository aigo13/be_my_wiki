from be_my_wiki.parsing.chunker import chunk_note


def test_empty_source():
    assert chunk_note(note_path="empty.md", source="") == []


def test_only_frontmatter_no_body():
    src = "---\ntitle: T\ntags: [a]\n---\n"
    assert chunk_note(note_path="x.md", source=src) == []


def test_plain_body_no_headings():
    src = "Just some content.\nNo headings here."
    chunks = chunk_note(note_path="plain.md", source=src)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.heading_path == ()
    assert "Just some content." in c.body
    assert c.metadata["title"] == "plain"


def test_title_from_filename_with_spaces():
    chunks = chunk_note(note_path="My Cool Note.md", source="content")
    assert chunks[0].metadata["title"] == "My Cool Note"
    assert "Title: My Cool Note" in chunks[0].text


def test_h2_split():
    src = "## A\nalpha\n\n## B\nbeta"
    chunks = chunk_note(note_path="ab.md", source=src, heading_level=2)
    assert len(chunks) == 2
    assert chunks[0].heading_path == ("A",)
    assert chunks[0].body.strip() == "alpha"
    assert chunks[1].heading_path == ("B",)
    assert chunks[1].body.strip() == "beta"


def test_nested_h1_h2():
    src = "# Top\nintro\n\n## A\nalpha\n\n## B\nbeta"
    chunks = chunk_note(note_path="x.md", source=src, heading_level=2)
    assert len(chunks) == 3
    assert chunks[0].heading_path == ("Top",)
    assert chunks[0].body.strip() == "intro"
    assert chunks[1].heading_path == ("Top", "A")
    assert chunks[2].heading_path == ("Top", "B")


def test_h3_kept_as_content_when_level_2():
    src = "## A\nalpha\n\n### Sub\nsubtext\n\n## B\nbeta"
    chunks = chunk_note(note_path="x.md", source=src, heading_level=2)
    assert len(chunks) == 2
    assert "### Sub" in chunks[0].body
    assert "subtext" in chunks[0].body
    assert chunks[1].body.strip() == "beta"


def test_frontmatter_prefix_injection():
    src = """---
title: My Note
tags: [project, ml]
aliases: [Other]
---

## Section
content here"""
    chunks = chunk_note(note_path="my.md", source=src)
    assert len(chunks) == 1
    c = chunks[0]
    assert "Title: My Note" in c.text
    assert "Tags: project, ml" in c.text
    assert "Section: Section" in c.text
    assert c.body.strip() == "content here"
    assert c.metadata["tags"] == ["project", "ml"]
    assert c.metadata["aliases"] == ["Other"]


def test_long_section_subsplits():
    para1 = "alpha " * 100
    para2 = "beta " * 100
    src = f"## S\n{para1}\n\n{para2}"
    chunks = chunk_note(note_path="x.md", source=src, max_tokens=80)
    assert len(chunks) > 1
    char_cap = 80 * 3
    for c in chunks:
        assert len(c.body) <= char_cap


def test_hash_deterministic_and_unique():
    src1 = "## A\nalpha"
    src2 = "## A\nbeta"
    a = chunk_note(note_path="x.md", source=src1)
    b = chunk_note(note_path="x.md", source=src1)
    c = chunk_note(note_path="x.md", source=src2)
    assert a[0].content_hash == b[0].content_hash
    assert a[0].content_hash != c[0].content_hash


def test_korean_content():
    src = """---
tags: [한국어, 머신러닝]
---

## 서론
이것은 한글 본문입니다."""
    chunks = chunk_note(note_path="ko.md", source=src)
    assert len(chunks) == 1
    c = chunks[0]
    assert "Tags: 한국어, 머신러닝" in c.text
    assert "이것은 한글 본문입니다." in c.body
    assert c.metadata["tags"] == ["한국어", "머신러닝"]


def test_inline_tex_preserved():
    src = "## Theorem\nEinstein showed $E = mc^2$ in 1905."
    chunks = chunk_note(note_path="phys.md", source=src)
    assert len(chunks) == 1
    assert "$E = mc^2$" in chunks[0].body


def test_display_math_block_preserved():
    src = r"""## Integral

The fundamental theorem:

$$
\int_0^1 x \, dx = \frac{1}{2}
$$

is well known."""
    chunks = chunk_note(note_path="math.md", source=src)
    assert len(chunks) == 1
    body = chunks[0].body
    assert "$$" in body
    assert r"\int_0^1 x \, dx" in body
    assert r"\frac{1}{2}" in body


def test_hash_inside_math_not_treated_as_heading():
    src = r"""## Real
Let $\#A$ denote the cardinality of A.

More."""
    chunks = chunk_note(note_path="x.md", source=src)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Real",)
    assert r"$\#A$" in chunks[0].body


def test_korean_with_tex():
    src = """## 정리
페르마의 마지막 정리에 따르면 $a^n + b^n = c^n$은 $n > 2$일 때 정수해가 없다."""
    chunks = chunk_note(note_path="ko.md", source=src)
    assert len(chunks) == 1
    body = chunks[0].body
    assert "페르마의 마지막 정리" in body
    assert "$a^n + b^n = c^n$" in body


def test_code_block_with_hash_not_a_heading():
    src = '''## Real
content

```python
# this is a comment, not a heading
print("hi")
```

more content'''
    chunks = chunk_note(note_path="x.md", source=src)
    assert len(chunks) == 1
    assert "# this is a comment" in chunks[0].body


def test_tags_string_is_split():
    src = "---\ntags: project, ml\n---\n\nbody"
    chunks = chunk_note(note_path="x.md", source=src)
    assert chunks[0].metadata["tags"] == ["project", "ml"]


def test_chunk_index_is_sequential():
    src = "## A\na\n\n## B\nb\n\n## C\nc"
    chunks = chunk_note(note_path="x.md", source=src)
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_malformed_frontmatter_falls_back_to_body_only(caplog):
    # Real-world case: math notation with `{` and `:` in an unquoted YAML value
    # breaks PyYAML. We should still index the body.
    src = """---
tags: [math]
summary: T : V -> W. annihilator U^0 = {x : f(x)=0}.
---

## Theorem

Body content here."""
    chunks = chunk_note(note_path="DualMap.md", source=src)
    assert len(chunks) == 1
    assert chunks[0].body.strip() == "Body content here."
    assert chunks[0].heading_path == ("Theorem",)
    # Metadata is empty (frontmatter dropped) but title falls back to filename.
    assert chunks[0].metadata["title"] == "DualMap"
    assert chunks[0].metadata["tags"] == []
    # A warning was logged.
    assert any(
        "frontmatter parse failed" in r.message for r in caplog.records
    )


def test_malformed_frontmatter_without_closing_delimiter():
    # If the opening `---` has no matching close, leave source intact.
    src = """---
not really frontmatter

## Heading

body"""
    chunks = chunk_note(note_path="x.md", source=src)
    # The leading `---` line isn't valid YAML closing → frontmatter library
    # treats the whole thing as content. We should still produce a chunk.
    assert len(chunks) >= 1


def test_frontmatter_scalars_passthrough():
    src = """---
title: T
date: 2026-05-04
custom_field: hello
---

body"""
    chunks = chunk_note(note_path="x.md", source=src)
    md = chunks[0].metadata
    assert md["title"] == "T"
    assert "date" in md
    assert md["custom_field"] == "hello"
