import os
import re
import time
import json
import random
from typing import Optional, Dict, Any

import google.generativeai as genai

try:
    from google.api_core import exceptions as gexc
    GoogleAPIError = getattr(gexc, "GoogleAPIError", Exception)
    ResourceExhausted = getattr(gexc, "ResourceExhausted", GoogleAPIError)
except Exception:
    class GoogleAPIError(Exception): ...
    class ResourceExhausted(GoogleAPIError): ...

MAX_CTX = 20000

SYSTEM_RU = (
    "Ты — система извлечения полей из финансовых документов на русском и казахском языках. "
    "Тебе дают шумный текст (после OCR). Нужно вернуть ТОЛЬКО JSON по заданной схеме "
    'с ключом "fields". Ключи всегда на английском: amount, currency, date, iban, bic, '
    "iin_bin, invoice_no, payer, receiver. Если поле неизвестно, просто не включай его. "
    "Нормализуй: amount в формате 12345.67, currency как ISO (KZT/USD/EUR/RUB), "
    "date в формате YYYY-MM-DD. Учитывай русские синонимы. Если встречаешь «₸» или слово «тенге» → currency=KZT."
)

def _response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "amount": {"type": "string"},
                    "currency": {"type": "string"},
                    "date": {"type": "string"},
                    "iban": {"type": "string"},
                    "bic": {"type": "string"},
                    "iin_bin": {"type": "string"},
                    "invoice_no": {"type": "string"},
                    "payer": {"type": "string"},
                    "receiver": {"type": "string"},
                },
            }
        },
        "required": ["fields"],
    }

FEW_SHOTS_RU = [
    {
        "text": "СЧЁТ № 123 от 10.09.2025\nПлательщик: ТОО «Ромашка»\nПолучатель: АО «Банк Демонстрации»\nIBAN: KZ12345678901234567890  BIC: HSBKKZKX\nК оплате: 12 345,67 ₸",
        "out": {
            "fields": {
                "invoice_no": "123",
                "date": "2025-09-10",
                "iban": "KZ12345678901234567890",
                "bic": "HSBKKZKX",
                "amount": "12345.67",
                "currency": "KZT",
                "payer": "ТОО «Ромашка»",
                "receiver": "АО «Банк Демонстрации»",
            }
        },
    },
    {
        "text": "Счет: INV-77  Дата: 3 декабря 2024 г.\nИИК: KZ11223344556677889900  SWIFT/BIC: CASPKZKA\nИтого к оплате: 78 900,00 тенге",
        "out": {
            "fields": {
                "invoice_no": "INV-77",
                "date": "2024-12-03",
                "iban": "KZ11223344556677889900",
                "bic": "CASPKZKA",
                "amount": "78900.00",
                "currency": "KZT",
            }
        },
    },
]

_RU_MONTHS = {
    "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
    "мая": "05", "июня": "06", "июля": "07", "августа": "08",
    "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12",
}

def _norm_num(s: str) -> str:
    return s.replace(" ", "").replace("\u00A0", "").replace(",", ".")

def _try_parse_ru_date(t: str) -> Optional[str]:
    m = re.search(r"(?i)\b(0?[1-9]|[12]\d|3[01])\s+(" + "|".join(_RU_MONTHS.keys()) + r")\s+20(\d{2})\b", t)
    if m:
        dd = int(m.group(1))
        mm = _RU_MONTHS[m.group(2).lower()]
        yyyy = "20" + m.group(3)
        return f"{int(yyyy):04d}-{int(mm):02d}-{dd:02d}"
    return None

class LLMClient:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        disabled = os.getenv("LLM_DISABLED", "").lower() in ("1", "true", "yes")
        self.enabled = bool(api_key) and not disabled
        if self.enabled:
            genai.configure(api_key=api_key)

        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.generation_config = {
            "response_mime_type": "application/json",
            "response_schema": _response_schema(),
            "temperature": 0.0,
            "max_output_tokens": 512,
        }

    def map_to_fields(self, doc_type: str, text: str, hints: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {"fields": self._fallback(doc_type, text, hints)}

        hints_str = json.dumps(hints or {}, ensure_ascii=False)
        model_obj = genai.GenerativeModel(model or self.model_name, system_instruction=SYSTEM_RU)

        shot_blocks = []
        for ex in FEW_SHOTS_RU:
            shot_blocks += [
                "Пример входа:\n" + ex["text"],
                "Пример правильного JSON-выхода:\n" + json.dumps(ex["out"], ensure_ascii=False),
            ]

        user = f"Тип документа: {doc_type}\nПодсказки (кандидаты): {hints_str}\n\nИзвлеки поля: amount, currency, date, iban, bic, iin_bin, invoice_no, payer, receiver.\n\nТекст:\n{text[:MAX_CTX]}"

        last_err = None
        for attempt in range(3):
            try:
                resp = model_obj.generate_content(shot_blocks + [user], generation_config=self.generation_config)
                content = (getattr(resp, "text", "") or "").strip()
                if not content:
                    raise ValueError("Empty response from Gemini")
                return json.loads(content)
            except (ResourceExhausted, GoogleAPIError) as e:
                last_err = e
                code = getattr(e, "code", None)
                msg = (str(e) or "").lower()
                if isinstance(e, ResourceExhausted) or code == 429 or "429" in msg or "quota" in msg:
                    time.sleep(min(20, 2 ** attempt + random.random()))
                    continue
                break
            except Exception as e:
                last_err = e
                break

        return {"fields": self._fallback(doc_type, text, hints), "llm_error": str(last_err)[:500]}

    def _fallback(self, doc_type: str, text: str, hints: Optional[Dict[str, Any]]) -> Dict[str, str]:
        t = text or ""
        fields: Dict[str, str] = {}

        m = re.search(r"(?i)(итог(?:овая)?\s*сумма|итого(?:\s*к\s*оплате)?|сумма\s*к\s*оплате|к\s*оплате|total|amount|sum)\D{0,40}([0-9]+(?:[ .][0-9]{3})*(?:[.,][0-9]{2})?)", t)
        if not m:
            m = re.search(r"(?i)\b([0-9]+(?:[ .][0-9]{3})*(?:[.,][0-9]{2})?)\s*(тенге|₸|kzt)\b", t)
        if m:
            fields["amount"] = _norm_num(m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1))

        if re.search(r"(?i)\bKZT\b|₸|тенге", t):
            fields["currency"] = "KZT"
        else:
            cur = re.search(r"(?i)\b(USD|EUR|RUB)\b", t)
            if cur:
                fields["currency"] = cur.group(1).upper()

        d = re.search(r"\b((?:20\d{2}[-./](?:0?[1-9]|1[0-2])[-./](?:0?[1-9]|[12]\d|3[01]))|(?:(?:0?[1-9]|[12]\d|3[01])[-./](?:0?[1-9]|1[0-2])[-./]20\d{2}))\b", t)
        if d:
            s = d.group(1)
            if re.match(r"^\d{1,2}[./-]\d{1,2}[./-]20\d{2}$", s):
                dd, mm, yyyy = re.split(r"[./-]", s)
                fields["date"] = f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
            else:
                fields["date"] = s
        else:
            ru = _try_parse_ru_date(t)
            if ru:
                fields["date"] = ru

        iban = re.search(r"(?i)\bKZ\d{20}\b", t)
        if iban:
            fields["iban"] = iban.group(0)

        bic = re.search(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b", t)
        if bic:
            fields["bic"] = bic.group(0)

        iin = re.search(r"\b\d{12}\b", t)
        if iin:
            fields.setdefault("iin_bin", iin.group(0))

        inv = re.search(r"(?i)(?:сч(?:е|ё)т|номер\s*сч[её]та|invoice|inv|doc\s*no\.?)\s*[:#№]*\s*([\w\-_/]{3,})", t)
        if inv:
            fields["invoice_no"] = inv.group(1)

        recv = re.search(r"(?i)(?:получатель|receiver|beneficiary)\s*[:\-]?\s*(.+)", t)
        if recv:
            fields["receiver"] = recv.group(1).splitlines()[0].strip()
        pay = re.search(r"(?i)(?:плательщик|payer|customer)\s*[:\-]?\s*(.+)", t)
        if pay:
            fields["payer"] = pay.group(1).splitlines()[0].strip()

        return fields

    def fix_text(self, full_text: str, fields_hint: Optional[Dict[str, Any]] = None,
                 language: str = "ru") -> Dict[str, Any]:
        """
        Редактура OCR-текста с самопроверкой. Возвращает:
        {
          "corrected_text": "....",
          "edits": [{"from":"...", "to":"...", "reason":"..."}],
          "notes": "краткий лог (опционально)"
        }
        """
        if not self.enabled:
            # без LLM просто эхо
            return {"corrected_text": full_text, "edits": [], "notes": "LLM disabled"}

        schema = {
            "type": "object",
            "properties": {
                "corrected_text": {"type": "string"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                            "reason": {"type": "string"}
                        },
                        "required": ["from","to"]
                    }
                },
                "notes": {"type": "string"}
            },
            "required": ["corrected_text"]
        }

        system = (
            "Ты — редактор финансовых документов (RU/KZ).\n"
            "Исправляй только OCR-опечатки и склейки/разрывы слов, сохраняя исходный смысл и числа.\n"
            "Не придумывай новые факты. Не меняй суммы/даты, если они выглядят корректно.\n"
            "Возвращай ТОЛЬКО JSON по схеме."
        )
        hints = fields_hint or {}
        user = (
            f"Язык: {language}\n"
            f"Подсказки полей: {json.dumps(hints, ensure_ascii=False)}\n\n"
            f"Текст для правки:\n{full_text[:MAX_CTX]}"
        )

        model_obj = genai.GenerativeModel(self.model_name, system_instruction=system)
        cfg = {**self.generation_config, "response_schema": schema}

        last_err = None
        for attempt in range(3):
            try:
                resp = model_obj.generate_content([user], generation_config=cfg)
                txt = (getattr(resp, "text", "") or "").strip()
                if not txt:
                    raise ValueError("Empty response")
                return json.loads(txt)
            except (ResourceExhausted, GoogleAPIError) as e:
                last_err = e
                time.sleep(min(20, 2 ** attempt + random.random()))
            except Exception as e:
                last_err = e
                break

        return {"corrected_text": full_text, "edits": [], "notes": f"error: {str(last_err)[:300]}"}
