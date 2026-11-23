"""Document processing and modality extraction (report section 3.2.1).

PDF  -> text chunks (T), figures (I), tables (A), charts (C) + full-page renders
Image-> a single image / chart element
Text -> text chunks
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from .config import Config
from .schema import Element
from .utils import rows_to_markdown

CAPTION_RE = re.compile(r"^\s*(figure|fig\.?|chart|graph|table|plot|exhibit)\s*[\dIVX]*[\.:\-\u2013]?", re.I)
CHART_WORDS = re.compile(
    r"\b(chart|graph|plot|axis|axes|bar|histogram|trend|curve|scatter|heatmap|pie|over time|vs\.?)\b", re.I)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


# --------------------------------------------------------------------------- text
def chunk_text(text: str, max_words: int = 180, overlap: int = 30) -> list[str]:
    """Sentence-aware chunking: at most `max_words` words per chunk, `overlap` words carried over (eq. 4)."""
    text = re.sub(r"[ \t]+", " ", text)
    sents = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks: list[str] = []
    cur: list[str] = []
    for s in sents:
        w = s.split()
        if not w:
            continue
        if len(w) > max_words:                      # one giant sentence: hard split
            if cur:
                chunks.append(" ".join(cur))
                cur = []
            step = max(1, max_words - overlap)
            for i in range(0, len(w), step):
                chunks.append(" ".join(w[i:i + max_words]))
                if i + max_words >= len(w):
                    break
            continue
        if len(cur) + len(w) > max_words and cur:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.extend(w)
    if cur and (not chunks or len(cur) > overlap):
        chunks.append(" ".join(cur))
    return chunks


# ------------------------------------------------------------------ chart heuristic
def looks_like_chart(path: str, caption: str = "") -> bool:
    """Cheap chart-vs-photo classifier: caption keywords, or mostly-white canvas with few distinct colours."""
    if caption and CHART_WORDS.search(caption):
        return True
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        im.thumbnail((128, 128))
        px = list(im.getdata())
        near_white = sum(1 for p in px if min(p) > 235) / max(1, len(px))
        buckets = {(r // 32, g // 32, b // 32) for r, g, b in px}
        return near_white > 0.45 and len(buckets) < 60
    except Exception:
        return False


# ------------------------------------------------------------------------- PDF
def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def find_caption(blocks, bbox, max_gap: float = 70.0) -> str:
    """Nearest 'Figure N / Table N ...' text block directly above or below a bbox."""
    x0, y0, x1, y1 = bbox
    best, best_d = "", 1e9
    for b in blocks:
        if b[6] != 0:
            continue
        t = " ".join(b[4].split())
        if not CAPTION_RE.match(t):
            continue
        bx0, by0, bx1, by1 = b[:4]
        if min(x1, bx1) - max(x0, bx0) <= 0:
            continue
        if by0 >= y1 - 5:
            d = by0 - y1
        elif by1 <= y0 + 5:
            d = y0 - by1
        else:
            continue
        if d <= max_gap and d < best_d:
            best, best_d = t, d
    return best


def _render(page, bbox, out: Path, dpi: int) -> str:
    import fitz
    pix = page.get_pixmap(clip=fitz.Rect(bbox), dpi=dpi)
    pix.save(str(out))
    return str(out)


def ingest_pdf(path: str, doc_id: str, asset_dir: Path, cfg: Config) -> list[Element]:
    import fitz  # PyMuPDF

    asset_dir.mkdir(parents=True, exist_ok=True)
    els: list[Element] = []
    doc = fitz.open(path)
    for pno, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")

        # ---- tables: dual format (markdown text + cropped raster), eq. 6
        table_boxes = []
        try:
            tables = page.find_tables().tables
        except Exception:
            tables = []
        for ti, tab in enumerate(tables):
            try:
                rows = tab.extract()
            except Exception:
                continue
            md = rows_to_markdown(rows)
            if not md or len(rows) < 2:
                continue
            bbox = tuple(round(v, 1) for v in tab.bbox)
            table_boxes.append(bbox)
            cap = find_caption(blocks, bbox)
            img = _render(page, bbox, asset_dir / f"p{pno}_table{ti}.png", cfg.render_dpi)
            els.append(Element("table", (cap + "\n" if cap else "") + md, img,
                               {"page": pno, "bbox": bbox, "caption": cap}))

        # ---- text chunks (skip text that lives inside tables), eq. 4
        parts = [b[4] for b in blocks if b[6] == 0 and not any(_overlap(b[:4], tb) for tb in table_boxes)]
        for ci, ch in enumerate(chunk_text("\n\n".join(parts), cfg.chunk_words, cfg.chunk_overlap)):
            els.append(Element("text", ch, None, {"page": pno, "chunk": ci}))

        # ---- embedded raster figures -> image (I) or chart (C), eq. 5 / 7
        for ii, info in enumerate(page.get_images(full=True)):
            try:
                rects = page.get_image_rects(info[0])
            except Exception:
                rects = []
            for ri, r in enumerate(rects):
                if r.width < cfg.min_image_px or r.height < cfg.min_image_px:
                    continue
                bbox = tuple(round(v, 1) for v in r)
                out = _render(page, r, asset_dir / f"p{pno}_img{ii}_{ri}.png", cfg.render_dpi)
                cap = find_caption(blocks, bbox)
                mod = "chart" if looks_like_chart(out, cap) else "image"
                els.append(Element(mod, cap, out, {"page": pno, "bbox": bbox, "caption": cap}))

        # ---- whole page (catches vector-drawn charts that have no embedded bitmap)
        if cfg.index_pages:
            out = asset_dir / f"p{pno}_page.png"
            page.get_pixmap(dpi=cfg.page_dpi).save(str(out))
            els.append(Element("page", "", str(out), {"page": pno}))
    doc.close()
    return els


# ----------------------------------------------------------------- images / text files
def ingest_image(path: str, asset_dir: Path, label: str | None = None) -> list[Element]:
    """A user-uploaded picture. `label` can force 'chart' or 'image'; otherwise a heuristic decides."""
    from PIL import Image

    asset_dir.mkdir(parents=True, exist_ok=True)
    out = asset_dir / (Path(path).stem + ".png")
    Image.open(path).convert("RGB").save(out)
    mod = label if label in ("chart", "image") else ("chart" if looks_like_chart(str(out)) else "image")
    return [Element(mod, "", str(out), {"page": 1, "caption": ""})]


def ingest_textfile(path: str, cfg: Config) -> list[Element]:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [Element("text", c, None, {"page": 1, "chunk": i})
            for i, c in enumerate(chunk_text(raw, cfg.chunk_words, cfg.chunk_overlap))]


def copy_into(path: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(path).name
    if Path(path).resolve() != dest.resolve():
        shutil.copy2(path, dest)
    return dest
