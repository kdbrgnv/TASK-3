# src/pipeline.py
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image, ImageDraw
import re
import unicodedata
import logging

from src.preprocess import pdf_to_images
from utils.ocr_utils import preprocess_for_ocr
from src.ocr_paddle import PaddleEngine
from src.vt_donut import DonutEngine
from src.post_llm import LLMClient
from src.post_rules import fix_fields
from src.section_parser import build_sections  # <-- парсер разделов
from src.post_ocr_corrector import PostCorrector  # <-- автокорректор OCR-текста

# Регулярки для быстрых подсказок LLM
RE_IBAN = re.compile(r"\bKZ\d{20}\b", flags=re.I)
RE_BIN = re.compile(r"\b\d{12}\b")

# Инициализация движков
paddle = PaddleEngine(lang="ru")
donut = DonutEngine()
llm = LLMClient()
corrector = PostCorrector(enable_headings=True, enable_terms=True)


def _pages_from_file(path: str | Path) -> List[Image.Image]:
    """Загружает изображения из PDF или из файла-изображения."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return pdf_to_images(p)
    return [Image.open(p).convert("RGB")]


def _poly_to_ltrb(poly: List[List[float]]) -> List[int]:
    """Конвертирует список точек (полигон) в bbox [x1,y1,x2,y2]."""
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _sanitize_for_llm(s: str, *, max_len: int = 8000) -> str:
    """
    Чистим текст от «опасных» символов для sentencepiece/токенизаторов:
    - NFKC нормализация
    - удаление управляющих \x00..\x08, \x0B, \x0C, \x0E..\x1F и суррогатов
    - обрезка слишком длинных строк
    """
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    s = "".join(
        ch
        for ch in s
        if ch in ("\n", "\r", "\t")
        or (ord(ch) >= 0x20 and not (0xD800 <= ord(ch) <= 0xDFFF))
    )
    s = s.replace("\u200b", "").replace("\uFEFF", "")
    if len(s) > max_len:
        s = s[:max_len]
    return s


def run_pipeline(
    path: str | Path,
    doc_type_hint: str | None = None,
    *,
    conf_threshold: float = 0.5,
    preproc_mode: str = "soft"
):
    """
    Основной конвейер:
    - Загружает страницы
    - Выполняет OCR (Пaddle) + автокоррекция русских OCR-ошибок
    - Пробует классификацию документа (Donut) — безопасно
    - Передаёт текст в LLM — безопасно
    - Строит разделы
    """
    pages = _pages_from_file(path)
    out_pages: List[Image.Image] = []
    ocr_pages: List[List[dict]] = []
    all_text: List[str] = []
    donut_guess = None
    donut_error = None
    llm_error = None

    for img in pages:
        # --- выбор препроцессинга ---
        if preproc_mode == "soft":
            img2 = preprocess_for_ocr(img)
        else:
            img2 = preprocess_for_ocr(img, use_clahe=False, do_unsharp=True)

        # --- OCR ---
        ocr_raw = paddle.run(img2)

        # Нормализация bbox + фильтр по уверенности
        ocr_norm: List[dict] = []
        for o in ocr_raw:
            conf = float(o.get("conf", o.get("score", 0.0)) or 0.0)
            if conf < conf_threshold:
                continue

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

            ocr_norm.append({
                "text": o.get("text", "") or "",
                "bbox": bb,
                "conf": conf,
            })

        # --- Автокоррекция русских OCR-ошибок (латиница→кириллица, частые опечатки, заголовки) ---
        try:
            ocr_fixed = corrector.correct_items(ocr_norm)
        except Exception as _:
            ocr_fixed = ocr_norm  # не валим пайплайн

        ocr_pages.append(ocr_fixed)
        all_text.append(" ".join(o["text"] for o in ocr_fixed if o.get("text")))

        # --- Donut (классификатор) безопасно ---
        try:
            dj = donut.infer(img2)
            if isinstance(dj, dict) and not donut_guess:
                donut_guess = dj.get("document_type") or dj.get("doctype")
        except Exception as e:
            logging.warning("DonutEngine failed: %s", e)
            donut_error = str(e)

        # --- отрисовка bbox ---
        draw = ImageDraw.Draw(img2)
        for o in ocr_fixed:
            bb = o.get("bbox")
            if bb:
                x1, y1, x2, y2 = bb
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=1)

        out_pages.append(img2)

    # --- агрегация текста и подсказок ---
    raw_text = " ".join(all_text)
    doc_type = doc_type_hint or donut_guess or "receipt"

    hints = {
        "iban_candidates": list(set(RE_IBAN.findall(raw_text))),
        "bin_candidates": list(set(RE_BIN.findall(raw_text))),
    }

    # --- LLM безопасно (санитайзер + try/except) ---
    text_clean = _sanitize_for_llm(raw_text)
    try:
        fields = llm.map_to_fields(doc_type, text_clean, hints).get("fields", {})
    except Exception as e:
        logging.warning("LLMClient.map_to_fields failed: %s", e)
        llm_error = str(e)
        fields = {}

    fields = fix_fields(fields)

    # --- разделы по исправленному OCR ---
    try:
        sections = build_sections(ocr_pages)
    except Exception:
        sections = []

    result: Dict[str, Any] = {
        "docType": doc_type,
        "meta": {
            "pages": len(pages),
            "lang": "ru",
            "confidence": 0.0,
            "preproc_mode": preproc_mode,
            "conf_threshold": conf_threshold,
        },
        "fields": fields,
        "lineItems": [],
        "sections": sections,
        "debug": {
            "ocr": ocr_pages,
            "llm_text_len": len(text_clean),
            "donut_error": donut_error,
            "llm_error": llm_error,
        },
    }
    return result, out_pages
