# src/ocr_paddle.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

try:
    from paddleocr import PaddleOCR
except ImportError as e:
    raise RuntimeError("PaddleOCR не установлен. Установи: pip install 'paddleocr>=2.7'") from e

# --- нормализация: латиница ↔ кириллица, частые путаницы ---
LATIN_TO_CYR = str.maketrans({
    "A":"А", "B":"В", "C":"С", "E":"Е", "H":"Н", "K":"К", "M":"М", "O":"О", "P":"Р", "T":"Т", "X":"Х", "Y":"У",
    "a":"а", "c":"с", "e":"е", "o":"о", "p":"р", "x":"х", "y":"у",
})
DIGIT_CONFUSIONS = {
    "0": "О",  # ноль → О (если рядом буквы)
    "1": "І",  # единица → І (реже нужно; при желании отключи)
}

ALLOWED = (
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЬЪЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщыьъэюя"
    "0123456789 -.,:;()/%№\"«»"
)

def _keep_allowed(s: str) -> str:
    if not s:
        return s
    return "".join(ch for ch in s if ch in ALLOWED)

def _smart_digit_fix(text: str) -> str:
    # простая эвристика: если вокруг цифры буквы — заменим на похожую кириллицу
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in DIGIT_CONFUSIONS:
            left = chars[i-1] if i > 0 else " "
            right = chars[i+1] if i+1 < len(chars) else " "
            if left.isalpha() or right.isalpha():
                chars[i] = DIGIT_CONFUSIONS[ch]
    return "".join(chars)

def normalize_ru(text: str) -> str:
    if not text:
        return text
    t = text.translate(LATIN_TO_CYR)
    t = _smart_digit_fix(t)
    # нормализация пробелов и тире
    t = t.replace("–", "-").replace("—", "-")
    t = " ".join(t.split())
    return t

@dataclass
class OCRItem:
    text: str
    bbox: List[int] | None
    conf: float

class PaddleEngine:
    def __init__(self, lang: str = "ru"):
        # Безопасные параметры, совместимые с 2.7.x
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            det=True, rec=True,
            rec_image_shape="3,48,640",  # было 320 → 640 (лучше держит длинные строки)
            use_space_char=True,
            show_log=False,
            det_db_thresh=0.25,  # было 0.3 → ловим слабые боксы
            det_db_box_thresh=0.45,  # было 0.5
            det_db_unclip_ratio=1.8,  # было 1.6 → чуток «распухаем» боксы
            drop_score=0.10,  # не выкидывать слабые символы слишком агрессивно
            # rec_batch_num=8,            # можно поднять, если хватает RAM/VRAM
        )

    @staticmethod
    def _poly_to_ltrb(poly: List[List[float]]) -> List[int]:
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    @staticmethod
    def _sort_boxes(items: List[OCRItem]) -> List[OCRItem]:
        # Стабильная сортировка: сначала по y, потом по x
        return sorted(items, key=lambda it: (999999 if it.bbox is None else it.bbox[1],
                                             999999 if it.bbox is None else it.bbox[0]))

    @staticmethod
    def _merge_tokens_to_lines(items: List[OCRItem], y_tol: int = 8) -> List[OCRItem]:
        """
        Группируем токены в строки по близости Y; склеиваем текст через пробел.
        bbox строки — минимальный охватывающий прямоугольник.
        """
        if not items:
            return []
        lines: List[List[OCRItem]] = []
        curr: List[OCRItem] = []

        def y_center(it: OCRItem) -> int:
            if not it.bbox: return 0
            return (it.bbox[1] + it.bbox[3]) // 2

        for it in items:
            if not curr:
                curr = [it]
                continue
            if abs(y_center(it) - y_center(curr[-1])) <= y_tol:
                curr.append(it)
            else:
                lines.append(curr)
                curr = [it]
        if curr:
            lines.append(curr)

        merged: List[OCRItem] = []
        for line in lines:
            texts = [normalize_ru(t.text) for t in line if t.text]
            text = " ".join([t for t in texts if t])
            # объединённый bbox
            xs1, ys1, xs2, ys2 = [], [], [], []
            confs = []
            for t in line:
                if t.bbox:
                    xs1.append(t.bbox[0]);
                    ys1.append(t.bbox[1])
                    xs2.append(t.bbox[2]);
                    ys2.append(t.bbox[3])
                confs.append(t.conf)
            bbox = None
            if xs1 and ys1 and xs2 and ys2:
                bbox = [min(xs1), min(ys1), max(xs2), max(ys2)]
            conf = float(np.mean(confs)) if confs else 0.0
            merged.append(OCRItem(text=text, bbox=bbox, conf=conf))
        return merged

    def run(self, img: Image.Image) -> List[Dict[str, Any]]:
        """
        Возвращает список элементов: {"text": "...", "bbox": [x1,y1,x2,y2], "conf": 0.xx}
        где каждый элемент — уже строка (а не «слово»)
        """
        arr = np.array(img)
        # result формата: [ [ [poly, text, conf], ... ] ] на страницу
        res = self.ocr.ocr(arr, cls=True)
        raw_items: List[OCRItem] = []

        if res and res[0]:
            for poly, (txt, sc) in res[0]:
                bbox = self._poly_to_ltrb(poly) if poly else None
                txt = normalize_ru(txt)
                raw_items.append(OCRItem(text=txt, bbox=bbox, conf=float(sc or 0.0)))

        # Сортировка + склейка в строки
        raw_items = self._sort_boxes(raw_items)
        line_items = self._merge_tokens_to_lines(raw_items, y_tol=10)

        # Фильтрация мусора: выкидываем очень короткие строки с низкой уверенностью
        cleaned = []
        for it in line_items:
            if (len(it.text) <= 2 and it.conf < 0.6):
                continue
            cleaned.append({"text": it.text, "bbox": it.bbox, "conf": it.conf})

        return cleaned
