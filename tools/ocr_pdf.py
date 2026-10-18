"""
OCR a scanned (image-only) PDF into a sidecar file used by math_tutor.ingest.

Some CBSE papers are scans with no text layer, so the normal PDF reader returns nothing
and the paper never reaches the search index. Run this once per scanned PDF, then rebuild
the vector store (`python -m math_tutor.ingest`).

    pip install rapidocr onnxruntime pymupdf numpy
    python tools/ocr_pdf.py "PYQ/65-1-1_Mathematics 2025.pdf"

Writes "<pdf name>.ocr.json" next to the PDF (a JSON list with one string per page).
Commit that file: Streamlit Cloud does not run OCR.

Limits: the OCR model reads English only. Hindi lines are dropped, and mathematical
notation (fractions, matrices, integrals) is only approximately recovered.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LINE_TOLERANCE = 18          # pixels at 200 dpi: boxes closer than this share a line
_NON_ENGLISH = re.compile(r"[\u0900-\u097F\u3000-\u9fff\uff00-\uffef]")


def ocr_pdf(pdf: Path, dpi: int = 200) -> list[str]:
    import fitz  # pymupdf
    import numpy as np
    from rapidocr import RapidOCR

    engine = RapidOCR()
    pages: list[str] = []
    with fitz.open(str(pdf)) as doc:
        for number, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
            result = engine(image)
            boxes = result.boxes if result.boxes is not None else []
            items = [(sum(p[1] for p in b) / 4, min(p[0] for p in b), t)
                     for b, t in zip(boxes, result.txts or [])]
            items.sort(key=lambda r: (round(r[0] / LINE_TOLERANCE), r[1]))
            lines, current, anchor = [], [], None
            for y, x, text in items:
                if anchor is None or abs(y - anchor) <= LINE_TOLERANCE:
                    current.append((x, text))
                    anchor = y if anchor is None else anchor
                else:
                    lines.append(" ".join(t for _, t in sorted(current)))
                    current, anchor = [(x, text)], y
            if current:
                lines.append(" ".join(t for _, t in sorted(current)))
            lines = [ln for ln in lines if ln.strip() and not _NON_ENGLISH.search(ln)]
            pages.append("\n".join(lines))
            print(f"  page {number}: {len(lines)} lines", flush=True)
    return pages


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    source = Path(sys.argv[1])
    target = source.with_name(source.name + ".ocr.json")
    pages = ocr_pdf(source)
    target.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {target} ({sum(len(p) for p in pages)} characters)")
