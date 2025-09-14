from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import re
import difflib

# --- 1) Базовые маппинги похожих символов ---
LATIN_TO_CYR = str.maketrans({
    "A":"А","B":"В","C":"С","E":"Е","H":"Н","K":"К","M":"М","O":"О","P":"Р","T":"Т","X":"Х","Y":"У",
    "a":"а","c":"с","e":"е","o":"о","p":"р","x":"х","y":"у"
})
DIGIT_TO_CYR_IN_WORD = {"0":"О","1":"І","3":"З"}  # минимально инвазивно

RE_SPACE = re.compile(r"\s+")
RE_DASH = re.compile(r"[–—−]+")  # разные тире
RE_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9\-]+", re.UNICODE)

# --- 2) Словарь доменных фраз/терминов ---
PHRASES = [
    "ДОГОВОР",
    "I. ПРЕДМЕТ ДОГОВОРА",
    "II. ЦЕНА ТОВАРА, ОБЩАЯ СТОИМОСТЬ ДОГОВОРА",
    "III. КАЧЕСТВО, УПАКОВКА И МАРКИРОВКА",
    "IV. ПРИЕМКА ТОВАРА",
    "V. ПРЕТЕНЗИИ",
    "VI. СРОКИ И ПОРЯДОК ПОСТАВКИ",
    "VII. ПОРЯДОК РАСЧЕТОВ",
    "VIII. ОТВЕТСТВЕННОСТЬ СТОРОН",
    "IX. ФОРС-МАЖОР",
    "X. АРБИТРАЖ",
    "XI. ПРОЧИЕ УСЛОВИЯ",
    "Инкотермс 2010",
    "Продавец",
    "Покупатель",
    "Республика Беларусь",
    "Республика Казахстан",
    "товарная накладная",
    "сертификат качества",
    "сертификат происхождения",
    "железнодорожная накладная",
    "корректировочный акт",
    "счет-фактура",
]

TERMS = [
    "Договор","Предмет","Цена","Товара","Стоимость","Качество","Упаковка","Маркировка",
    "Приемка","Претензии","Сроки","Порядок","Поставки","Расчетов","Ответственность",
    "Форс-мажор","Арбитраж","Прочие","Условия","Продавец","Покупатель","Инкотермс",
    "Республика","Беларусь","Казахстан","накладная","паспорт","декларация","счет-фактура",
    "происхождения","качествa","железнодорожная","корректировочный","акт"
]

# --- 3) Типовые OCR-ошибки → канонические формы (регистронезависимо, по словам) ---
CANON_REPLACEMENTS = {
    r"\bродавец\b": "Продавец",
    r"\bродавцом\b": "Продавцом",
    r"\bродавцу\b": "Продавцу",
    r"\bродавец[а-я]*\b": "Продавец",

    r"\bоку[пп]ател[ьяею]?\b": "Покупатель",
    r"\bпокупател[ьяею]?\b": "Покупатель",

    r"\bР[еэ]спублика\b": "Республика",
    r"\bБеларус[ьъ]\b": "Беларусь",
    r"\bКазахста[нпm]\b": "Казахстан",

    r"\bИнкотерм[сc]\s*2010\b": "Инкотермс 2010",
    r"\bИнкотерм[сc]\b": "Инкотермс",

    r"\bтовар[оа]сопров[оа]дител[ьн]\w*\b": "товаросопроводительный",
    r"\bжелезнодоро\w+\b": "железнодорожная",

    r"\bдеклар[ао]ц[иi]я\b": "декларация",
    r"\bпроисхожден[иie]я\b": "происхождения",

    r"\bсчет[- ]?факт[уy]р[ао]\b": "счет-фактура",
    r"\bкорректировочн\w*\b": "корректировочный",

    r"\bкачес?тв[ао]\b": "качества",
}

def _fix_latin_and_digits(s: str) -> str:
    s = s.translate(LATIN_TO_CYR)
    s = RE_DASH.sub("-", s)
    s = RE_SPACE.sub(" ", s).strip()
    # цифры внутри слова → похожие кириллические
    def _fix_token(token: str) -> str:
        if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token):
            chars = []
            for ch in token:
                if ch in DIGIT_TO_CYR_IN_WORD:
                    chars.append(DIGIT_TO_CYR_IN_WORD[ch])
                else:
                    chars.append(ch)
            return "".join(chars)
        return token

    out_tokens = []
    for token in RE_TOKEN.findall(s):
        out_tokens.append(_fix_token(token))

    rebuilt = s
    for t in RE_TOKEN.findall(s):
        rebuilt = rebuilt.replace(t, out_tokens.pop(0), 1)
    return rebuilt

def _apply_canon_rules(s: str) -> str:
    t = s
    for pat, repl in CANON_REPLACEMENTS.items():
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    return t

def _closer(a: str, candidates: list[str], cutoff: float = 0.88) -> str | None:
    best = None
    best_ratio = 0.0
    ua = a.upper()
    for c in candidates:
        r = difflib.SequenceMatcher(None, ua, c.upper()).ratio()
        if r > best_ratio:
            best_ratio = r; best = c
    return best if best_ratio >= cutoff else None

def _fix_heading_like(line: str) -> str:
    cand = _closer(line, PHRASES, cutoff=0.82)
    return cand if cand else line

def _fix_terms_in_line(line: str) -> str:
    tokens = RE_TOKEN.findall(line)
    if not tokens:
        return line
    replaced = {}
    for tok in tokens:
        has_lat = re.search(r"[A-Za-z]", tok) is not None
        mixed = has_lat and re.search(r"[А-Яа-яЁё]", tok) is not None
        has_digit_in_word = any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok)
        if mixed or has_digit_in_word or tok.isupper():
            near = _closer(tok, TERMS, cutoff=0.9)
            if near:
                replaced[tok] = near
    out = line
    for src, dst in replaced.items():
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out)
    return out

@dataclass
class PostCorrector:
    enable_headings: bool = True
    enable_terms: bool = True

    def fix_text(self, s: str) -> str:
        if not s:
            return s
        t = _fix_latin_and_digits(s)
        t = _apply_canon_rules(t)
        if self.enable_headings:
            t = _fix_heading_like(t)
        if self.enable_terms:
            t = _fix_terms_in_line(t)
        return t

    def correct_items(self, ocr_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for it in ocr_items:
            txt = it.get("text", "")
            out.append({**it, "text": self.fix_text(txt)})
        return out
