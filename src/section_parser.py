# src/section_parser.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
import statistics as stats

# --- Регулярки ---
RE_NUM = re.compile(
    r"^\s*(?:(?P<roman>[IVXLCDM]+)\.|(?P<num>\d+(?:\.\d+){0,3})\.?)\s+(.{2,})$",
    re.IGNORECASE
)
RE_SUBPOINT = re.compile(r"^\s*((\d+\.\d+(\.\d+)*)|[-•*]|[а-я]\))\s+.+", re.IGNORECASE)
RE_ALLCAPS_LINE = re.compile(r"^[A-ZА-ЯЁ0-9 ,.;:()\"'«»\-–—/]{3,}$")

def _upper_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    upp = [c for c in letters if c == c.upper()]
    return len(upp) / max(1, len(letters))

def _heading_level_from_num(num: str | None, roman: str | None) -> int:
    if roman:
        return 1
    if num:
        return min(1 + num.count("."), 4)
    return 1

def _looks_like_heading(text: str, line_h: float, median_h: float, gap_above: float) -> Tuple[bool, int, str | None]:
    t = text.strip()

    # 1) Явная нумерация
    m = RE_NUM.match(t)
    if m:
        roman = m.group("roman")
        num = m.group("num")
        level = _heading_level_from_num(num, roman)
        numbering = roman + "." if roman else num
        return True, level, numbering

    # 2) Визуальный заголовок (ALLCAPS, разрыв, высота)
    if len(t) <= 120 and _upper_ratio(t) >= 0.6 and line_h >= median_h * 1.12 and gap_above >= median_h * 0.8:
        return True, 1, None

    return False, 0, None

def _group_tokens_to_lines(ocr_page: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Принимает OCR-слова и склеивает в строки.
    """
    items = []
    for o in ocr_page:
        txt = (o.get("text") or "").strip()
        bb = o.get("bbox")
        if not txt or not bb or len(bb) != 4:
            continue
        x1, y1, x2, y2 = bb
        h = max(1, y2 - y1)
        items.append({"text": txt, "bbox": (x1, y1, x2, y2), "h": h})

    if not items:
        return []

    items.sort(key=lambda r: ((r["bbox"][1] + r["bbox"][3]) / 2.0, r["bbox"][0]))
    heights = [it["h"] for it in items]
    med_h = stats.median(heights) if heights else 12.0
    tol = max(3.0, med_h * 0.6)

    lines = []
    cur = []

    def y_center(bb): return (bb[1] + bb[3]) / 2.0

    for it in items:
        if not cur:
            cur = [it]
            continue
        y_prev = y_center(cur[-1]["bbox"])
        y_now = y_center(it["bbox"])
        if abs(y_now - y_prev) <= tol:
            cur.append(it)
        else:
            lines.append(cur)
            cur = [it]
    if cur:
        lines.append(cur)

    out = []
    for ln in lines:
        ln.sort(key=lambda r: r["bbox"][0])
        text = " ".join(x["text"] for x in ln)
        x1 = min(x["bbox"][0] for x in ln)
        y1 = min(x["bbox"][1] for x in ln)
        x2 = max(x["bbox"][2] for x in ln)
        y2 = max(x["bbox"][3] for x in ln)
        out.append({
            "text": text.strip(),
            "bbox_line": (x1, y1, x2, y2),
            "items": ln,
            "line_h": y2 - y1,
            "y_top": y1,
        })
    return out

def _find_gap_above(i: int, lines: List[Dict[str, Any]]) -> float:
    if i == 0:
        return 9999.0
    prev = lines[i - 1]["bbox_line"]
    cur = lines[i]["bbox_line"]
    return max(0.0, cur[1] - prev[3])

def _section_id(num_str: str | None, fallback_counter: int) -> str:
    if num_str:
        return num_str
    return str(fallback_counter)

def _build_paragraphs(lines: List[Dict[str, Any]], median_h: float) -> List[str]:
    """
    Деление внутри раздела на абзацы по поднумерации или большим зазорам.
    """
    paragraphs = []
    cur = []
    last_y = None
    for ln in lines:
        text = ln["text"].strip()
        gap = (ln["y_top"] - last_y) if last_y else 0
        is_subpoint = RE_SUBPOINT.match(text) is not None
        if is_subpoint or (gap > median_h * 1.5):
            if cur:
                paragraphs.append("\n".join(cur))
                cur = []
        cur.append(text)
        last_y = ln["bbox_line"][3]
    if cur:
        paragraphs.append("\n".join(cur))
    return paragraphs

def build_sections(ocr_pages: List[List[Dict[str, Any]]], *, min_content_len: int = 0) -> List[Dict[str, Any]]:
    """
    Разбивает OCR-результат на разделы и абзацы.
    """
    sections: List[Dict[str, Any]] = []
    fallback_id = 1

    page_lines: List[List[Dict[str, Any]]] = []
    all_line_heights = []
    for p in ocr_pages:
        L = _group_tokens_to_lines(p)
        page_lines.append(L)
        all_line_heights.extend([x["line_h"] for x in L if x.get("line_h")])

    med_h_global = stats.median(all_line_heights) if all_line_heights else 12.0
    current_stack: List[Dict[str, Any]] = []

    def close_to_level(level: int, page_idx: int):
        while current_stack and current_stack[-1]["level"] >= level:
            sec = current_stack.pop()
            sec["page_to"] = page_idx + 1
            if len(sec["content"].strip()) >= min_content_len or sec.get("title"):
                sections.append(sec)

    for pi, lines in enumerate(page_lines):
        for i, ln in enumerate(lines):
            text = ln["text"]
            y_gap = _find_gap_above(i, lines)
            is_head, level, numbering = _looks_like_heading(
                text=text,
                line_h=ln["line_h"],
                median_h=med_h_global,
                gap_above=y_gap,
            )
            if is_head:
                close_to_level(level, pi)
                sec = {
                    "id": _section_id(numbering, fallback_id),
                    "title": text,
                    "level": int(level),
                    "num": numbering,
                    "content": "",
                    "paragraphs": [],  # 👈 новые абзацы
                    "page_from": pi + 1,
                    "page_to": pi + 1,
                }
                current_stack.append(sec)
                fallback_id += 1
            else:
                if current_stack:
                    current_stack[-1]["content"] += (text + "\n")

        # обновляем page_to
        for s in current_stack:
            s["page_to"] = pi + 1

    # закрываем хвосты
    while current_stack:
        sec = current_stack.pop()
        if len(sec["content"].strip()) >= min_content_len or sec.get("title"):
            sections.append(sec)

    # абзацы
    for sec in sections:
        # выделяем только те строки, что попали в контент
        lines_for_sec = []
        for pi in range(sec["page_from"] - 1, sec["page_to"]):
            lines_for_sec.extend(page_lines[pi])
        sec["paragraphs"] = _build_paragraphs(lines_for_sec, med_h_global)
        sec["content"] = sec["content"].rstrip()

    return sections
