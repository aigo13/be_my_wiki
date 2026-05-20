---
name: pdf-fig-extractor
description: Extract figures from a PDF and embed them in an Obsidian-vault markdown note. Use whenever the user wants to ingest a paper or book PDF into their vault and the note should embed the actual figure images (not ASCII transcriptions or paraphrases). For each detected figure you decide whether it belongs in the note; chosen figures are saved to the vault's attachments folder as `<abbrev>_fig<N>.<ext>` (PNG by default, SVG when the figure is mostly vector) and embedded with an Obsidian wikilink.
---

# pdf-fig-extractor

For each figure in the PDF you decide whether it belongs in the note, then
this skill turns it into a real embedded image. Figure detection and
file extraction are handled by `extract_figures.py`; figure *judgement* is
your job.

## Helper script

Always invoke it through the be_my_wiki venv so PyMuPDF is on the path:

```bash
PY=/Users/sj.hwang/MyApps/be_my_wiki/.venv/bin/python
SCRIPT=/Users/sj.hwang/MyApps/be_my_wiki/.claude/skills/pdf-fig-extractor/extract_figures.py
```

Subcommands:

- `list <pdf>` — prints JSON: a filename `abbrev` and every detected
  figure with `id`, `page`, `label`, `number`, `caption`,
  `vector_fraction`, and `recommend_format` (`"png"` or `"svg"`).
- `extract <pdf> <figure_id> <out_path>` — writes the figure. Format is
  taken from the extension (`.png` or `.svg`). `--dpi N` for raster.
- `abbrev <pdf>` — prints the filename abbreviation alone.

## Workflow

1. **Confirm context.** You need three things; ask the user for any you
   don't already have:
   - the PDF path,
   - the vault's attachments folder (an absolute path — typically
     something like `<vault>/_attachments/`),
   - the markdown note you are authoring (new file or existing).

2. **List figures.** Run `list <pdf>`. Capture the `abbrev` field — every
   filename will use it.

3. **Per figure, decide if it belongs.** A figure belongs in the note when:
   - the note's narrative actually depends on it (it is the diagram or
     result being discussed, not a passing mention), and
   - omitting it would make the explanation harder to follow.

   Skip decorative figures, redundant plots, and anything tangential to
   what you're writing.

4. **Extract and save.** For each figure you keep:
   - Filename: `<abbrev>_fig<number>.<ext>` where `<number>` is the
     figure's `number` field (the label number from the PDF, *not* the
     `fig-N` id) and `<ext>` comes from `recommend_format`. Example:
     `circ_fig3.png`, `qaoa_fig1.svg`.
   - Run `extract <pdf> <figure_id> <attachments>/<filename>`.

5. **Embed in the note.** Place `![[<filename>]]` at the spot in the
   markdown where the figure belongs — directly under or next to the
   sentence it supports.

## Format choice (PNG vs SVG)

Default to `recommend_format` from `list`:

- **PNG** — figures that include raster content (photos, scans, screenshots)
  or are mostly text glyphs with thin vector marks (small circuit diagrams,
  Bell-state-style schematics). `vector_fraction` is low.
- **SVG** — figures that are mostly vector drawings (`vector_fraction` >
  0.5). Lossless at any zoom; preserves clean lines in vector circuit
  diagrams. Obsidian renders SVG embeds natively.

You may override the recommendation when you have a reason (e.g. user
prefers a single format for the whole note).

## Conventions

- One figure per filename — the `<number>` ties the file to the PDF's
  figure label, so cross-referencing stays obvious.
- Never embed ASCII-art or a hand-typed reproduction of a figure that
  exists in the PDF. If a figure is worth referencing, embed the image.
- Code listings shown in the PDF as numbered figures are intentionally
  skipped by `list` — reproduce those as fenced code blocks from the
  PDF's body text rather than as images.
