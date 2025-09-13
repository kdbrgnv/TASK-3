# src/pipeline.py
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Dict, Any, List
import re

from src.preprocess import pdf_to_images, preprocess_light
from src.ocr_paddle import PaddleEngine
from src.vt_donut import DonutEngine
from src.post_llm import LLMClient
from src.post_rules import fix_fields

# Регулярки для подсказок LLM
RE_IBAN = re.compile(r"\bKZ\d{20}\b", flags=re.I)   # KZ + 20 цифр (строго)
RE_BIN  = re.compile(r"\b\d{12}\b")

# Инициализация движков
paddle = PaddleEngine(lang="ru")
donut = DonutEngine()
llm = LLMClient()


def _pages_from_file(path: str | Path) -> List[Image.Image]:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return pdf_to_images(p)
    return [Image.open(p).convert("RGB")]


def run_pipeline(path: str | Path, doc_type_hint: str | None = None):
    pages = _pages_from_file(path)
    out_pages: List[Image.Image] = []
    ocr_pages: List[List[dict]] = []
    all_text: List[str] = []
    donut_guess = None

    for img in pages:
        # лёгкая предобработка перед OCR/Donut
        img2 = preprocess_light(img)

        # OCR (Paddle)
        ocr = paddle.run(img2)
        ocr_pages.append(ocr)
        all_text.append(" ".join(o["text"] for o in ocr))

        # Donut (DocVQA) — пробуем угадать тип документа
        dj = donut.infer(img2)
        if isinstance(dj, dict) and not donut_guess:
            donut_guess = dj.get("document_type") or dj.get("doctype")

        # отрисовка bbox'ов OCR
        draw = ImageDraw.Draw(img2)
        for o in ocr:
            x1, y1, x2, y2 = o["bbox"]
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=1)

        # добавляем страницу 1 раз (после цикла по bbox)
        out_pages.append(img2)

    raw_text = " ".join(all_text)
    doc_type = doc_type_hint or donut_guess or "receipt"

    # Подсказки для LLM
    hints = {
        "iban_candidates": list(set(RE_IBAN.findall(raw_text))),
        "bin_candidates":  list(set(RE_BIN.findall(raw_text))),
    }

    # Маппинг текста в поля (через Gemini-клиент с офлайн-фолбэком)
    fields = llm.map_to_fields(doc_type, raw_text, hints).get("fields", {})
    fields = fix_fields(fields)  # нормализация/валидация полей

    result: Dict[str, Any] = {
        "docType": doc_type,
        "meta": {"pages": len(pages), "lang": "ru", "confidence": 0.0},
        "fields": fields,
        "lineItems": None,
        "debug": {"ocr": ocr_pages},  # для Spy-панели
    }
    return result, out_pages
