#!/usr/bin/env python3
"""Detect and extract figures from a PDF for Obsidian-vault ingestion.

This is the helper that backs the ``pdf-fig-extractor`` skill. It is
deliberately minimal — figure judgement is left to the model.

Subcommands
-----------
``list <pdf>``                       JSON: ``abbrev`` + figures with id,
                                     page, label, caption, vector_fraction,
                                     recommend_format ("png" or "svg").

``extract <pdf> <figure_id> <out>``  Crop the figure to ``out``. Output
                                     format is taken from the extension —
                                     ``.png`` (raster) or ``.svg`` (vector).

``abbrev <pdf>``                     Print the filename abbreviation only.

Requires ``pymupdf`` (already in the be_my_wiki venv).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf

# Caption opener with a separator after the number, so body sentences that
# merely start with "Figure 3 shows ..." are not mistaken for captions.
_CAPTION_RE = re.compile(
    r"^\s*(fig(?:ure)?\.?|그림)\s*\.?\s*(\d+)\s*[.:|)–—]",
    re.IGNORECASE,
)
_MIN_PROSE_WORDS = 12   # short blocks above the caption never bound the region
_HEADER_ZONE = 95.0     # running headers/page numbers above this y do bound it
_TOP_MARGIN = 36.0
_PAD = 4.0
_MIN_FIGURE_SPAN = 24.0  # below this, the "figure" is a code listing


def _block_text(block: dict) -> str:
    return " ".join(
        "".join(s.get("text", "") for s in ln.get("spans", []))
        for ln in block.get("lines", [])
    )


def _find_captions(blocks: list[dict]) -> list[dict]:
    out: list[dict] = []
    for b in blocks:
        if b.get("type") != 0:
            continue
        text = _block_text(b).strip()
        m = _CAPTION_RE.match(text)
        if not m:
            continue
        out.append(
            {
                "number": m.group(2),
                "label": f"{m.group(1).rstrip('.').title()} {m.group(2)}",
                "caption": text,
                "bbox": tuple(b["bbox"]),
            }
        )
    return out


def _page_graphics(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects: list[pymupdf.Rect] = []
    for img in page.get_images(full=True):
        rects.extend(page.get_image_rects(img[0]))
    for d in page.get_drawings():
        rects.append(d["rect"])
    return [r for r in rects if not (r.is_empty or r.is_infinite)]


def _figure_region(
    caption: dict,
    captions: list[dict],
    blocks: list[dict],
    graphics: list[pymupdf.Rect],
    page_width: float,
) -> tuple[float, float, float, float]:
    """Band from the nearest body-prose block above the caption to the caption."""
    cx0, cy0, cx1, _ = caption["bbox"]

    top = _TOP_MARGIN
    for b in blocks:
        if b.get("type") != 0:
            continue
        bx0, _, bx1, by1 = b["bbox"]
        if by1 > cy0 - 1 or min(bx1, cx1) - max(bx0, cx0) <= 0:
            continue
        is_prose = len(_block_text(b).split()) >= _MIN_PROSE_WORDS
        if is_prose or by1 < _HEADER_ZONE:
            top = max(top, by1)
    for other in captions:
        if other["bbox"] == caption["bbox"]:
            continue
        ox0, _, ox1, oy1 = other["bbox"]
        if oy1 > cy0 - 1:
            continue
        if min(ox1, cx1) - max(ox0, cx0) > 0:
            top = max(top, oy1)

    xs0, xs1 = [cx0], [cx1]
    for b in blocks:
        if b.get("type") != 0 or not _block_text(b).strip():
            continue
        bx0, by0, bx1, by1 = b["bbox"]
        if by0 >= top - 2 and by1 <= cy0 + 2:
            xs0.append(bx0)
            xs1.append(bx1)
    for r in graphics:
        if r.y0 >= top - 2 and r.y1 <= cy0 + 2:
            xs0.append(r.x0)
            xs1.append(r.x1)

    y0 = min(top + _PAD, cy0 - 1)
    return (
        max(min(xs0) - _PAD, 0.0),
        y0,
        min(max(xs1) + _PAD, page_width),
        max(cy0, y0 + 1),
    )


def _vector_fraction(page: pymupdf.Page, region: tuple[float, float, float, float]) -> float:
    """Fraction of the region area covered by vector drawings (SVG-worthiness)."""
    rect = pymupdf.Rect(region)
    area = rect.get_area()
    if area <= 0:
        return 0.0
    vec_area = 0.0
    for d in page.get_drawings():
        r = d["rect"]
        if r.is_empty or r.is_infinite:
            continue
        inter = r & rect
        if not inter.is_empty:
            vec_area += inter.get_area()
    return min(1.0, vec_area / area)


def abbrev_from_filename(filename: str) -> str:
    """Compact filename abbreviation for figure filenames.

    Word-initials for multi-word stems; the truncated stem for short or
    purely-numeric ones (e.g. arXiv IDs).
    """
    stem = Path(filename).stem
    words = [w for w in re.split(r"[\s_\-.]+", stem) if w]
    if not words:
        return "fig"
    if len(stem) <= 4:
        return stem.lower()
    initials = "".join(w[0] for w in words if w[0].isalnum()).lower()
    if len(initials) >= 2:
        return initials[:6]
    return re.sub(r"[^a-z0-9]", "", stem.lower())[:4] or "fig"


def list_figures(pdf_path: str) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    out: list[dict] = []
    try:
        for pno in range(doc.page_count):
            page = doc[pno]
            blocks = page.get_text("dict")["blocks"]
            caps = _find_captions(blocks)
            graphics = _page_graphics(page)
            for cap in caps:
                region = _figure_region(
                    cap, caps, blocks, graphics, page.rect.width
                )
                if min(region[2] - region[0], region[3] - region[1]) < _MIN_FIGURE_SPAN:
                    continue  # code listing — let Claude reproduce from text
                vf = _vector_fraction(page, region)
                out.append(
                    {
                        "id": f"fig-{len(out) + 1}",
                        "page": pno + 1,
                        "label": cap["label"],
                        "number": cap["number"],
                        "caption": cap["caption"],
                        "bbox": [round(v, 1) for v in region],
                        "vector_fraction": round(vf, 3),
                        "recommend_format": "svg" if vf > 0.5 else "png",
                    }
                )
    finally:
        doc.close()
    return out


def extract(pdf_path: str, figure_id: str, out_path: str, dpi: int = 200) -> None:
    figs = list_figures(pdf_path)
    fig = next((f for f in figs if f["id"] == figure_id), None)
    if fig is None:
        sys.exit(f"figure {figure_id!r} not found in {pdf_path}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[fig["page"] - 1]
        rect = pymupdf.Rect(fig["bbox"])
        ext = out.suffix.lower()
        if ext == ".svg":
            svg = page.get_svg_image()
            out.write_text(_crop_svg(svg, rect), encoding="utf-8")
        elif ext in (".png", ".jpg", ".jpeg"):
            page.get_pixmap(clip=rect, dpi=dpi).save(out)
        else:
            sys.exit(f"unsupported output extension: {ext!r}")
    finally:
        doc.close()


def _crop_svg(svg_text: str, rect: pymupdf.Rect) -> str:
    """Crop a full-page SVG to ``rect`` by rewriting viewBox / width / height.

    Off-region content stays in the file but is clipped out of view.
    """
    rx, ry, rw, rh = rect.x0, rect.y0, rect.width, rect.height
    svg_text = re.sub(
        r'viewBox="[^"]*"', f'viewBox="{rx} {ry} {rw} {rh}"', svg_text, count=1
    )
    svg_text = re.sub(
        r'\bwidth="[^"]*"', f'width="{rw}"', svg_text, count=1
    )
    svg_text = re.sub(
        r'\bheight="[^"]*"', f'height="{rh}"', svg_text, count=1
    )
    return svg_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="JSON list of detected figures")
    p_list.add_argument("pdf")

    p_ex = sub.add_parser("extract", help="Extract one figure to file")
    p_ex.add_argument("pdf")
    p_ex.add_argument("figure_id")
    p_ex.add_argument("out_path")
    p_ex.add_argument("--dpi", type=int, default=200)

    p_ab = sub.add_parser("abbrev", help="Print filename abbreviation")
    p_ab.add_argument("pdf")

    args = ap.parse_args()
    if args.cmd == "list":
        print(
            json.dumps(
                {
                    "pdf": str(Path(args.pdf).resolve()),
                    "abbrev": abbrev_from_filename(args.pdf),
                    "figures": list_figures(args.pdf),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.cmd == "extract":
        extract(args.pdf, args.figure_id, args.out_path, dpi=args.dpi)
    elif args.cmd == "abbrev":
        print(abbrev_from_filename(args.pdf))


if __name__ == "__main__":
    main()
