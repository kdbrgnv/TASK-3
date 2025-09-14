# src/pipeline.py
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image, ImageDraw
import re

from src.preprocess import pdf_to_images
from utils.ocr_utils import preprocess_for_ocr
from src.ocr_paddle import PaddleEngine
from src.vt_donut import DonutEngine
from src.post_llm import LLMClient
from src.post_rules import fix_fields


RE_IBAN = re.compile(r"\bKZ\d{20}\b", flags=re.I)
RE_BIN = re.compile(r"\b\d{12}\b")

paddle = PaddleEngine(lang="ru")
donut = DonutEngine()
llm = LLMClient()


def _pages_from_file(path: str | Path) -> List[Image.Image]:
    """Загружает изображения из PDF или отдельного файла-изображения."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return pdf_to_images(p)
    return [Image.open(p).convert("RGB")]


def _poly_to_ltrb(poly: List[List[float]]) -> List[int]:
    """Конвертирует список точек в прямоугольник [x1, y1, x2, y2]."""
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def run_pipeline(path: str | Path, doc_type_hint: str | None = None):
    """
    Основной конвейер:
    1. Загружает страницы из PDF/изображений
    2. Выполняет OCR (Paddle)
    3. Пробует классификацию документа (Donut)
    4. Передаёт текст в LLM для маппинга в структурированные поля
    """
    pages = _pages_from_file(path)
    out_pages: List[Image.Image] = []
    ocr_pages: List[List[dict]] = []
    all_text: List[str] = []
    donut_guess = None

    for img in pages:
        img2 = preprocess_for_ocr(img)

        ocr_raw = paddle.run(img2)

        ocr_norm: List[dict] = []
        for o in ocr_raw:
            bb = o.get("bbox")
            if bb and isinstance(bb, list) and len(bb) == 4 and isinstance(bb[0], (list, tuple)):
                bb = _poly_to_ltrb(bb)
            elif bb and isinstance(bb, dict):
                x = bb.get("x", bb.get("left", 0))
                y = bb.get("y", bb.get("top", 0))
                w = bb.get("w", bb.get("width", 0))
                h = bb.get("h", bb.get("height", 0))
                bb = [int(x), int(y), int(x + w), int(y + h)]
            elif bb and isinstance(bb, list) and len(bb) == 4:
                bb = list(map(int, bb))
            else:
                bb = None

            conf = o.get("conf", o.get("score", 0.0))
            ocr_norm.append({
                "text": o.get("text", ""),
                "bbox": bb,
                "conf": float(conf) if conf is not None else 0.0
            })

        ocr_pages.append(ocr_norm)
        all_text.append(" ".join(o["text"] for o in ocr_norm if o.get("text")))

        dj = donut.infer(img2)
        if isinstance(dj, dict) and not donut_guess:
            donut_guess = dj.get("document_type") or dj.get("doctype")

        draw = ImageDraw.Draw(img2)
        for o in ocr_norm:
            bb = o.get("bbox")
            if bb:
                x1, y1, x2, y2 = bb
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=1)

        out_pages.append(img2)

    raw_text = " ".join(all_text)
    doc_type = doc_type_hint or donut_guess or "receipt"

    hints = {
        "iban_candidates": list(set(RE_IBAN.findall(raw_text))),
        "bin_candidates": list(set(RE_BIN.findall(raw_text))),
    }

    fields = llm.map_to_fields(doc_type, raw_text, hints).get("fields", {})
    fields = fix_fields(fields)

    result: Dict[str, Any] = {
        "docType": doc_type,
        "meta": {"pages": len(pages), "lang": "ru", "confidence": 0.0},
        "fields": fields,
        "lineItems": [],
        "debug": {"ocr": ocr_pages},
    }
    return result, out_pages
