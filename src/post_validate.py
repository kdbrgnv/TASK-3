# src/post_validate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import re
from datetime import datetime

RE_IBAN_KZ = re.compile(r"\bKZ\d{20}\b", flags=re.I)
RE_BIC     = re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b")
RE_DATE    = re.compile(r"\b(20\d{2})[-./](0?[1-9]|1[0-2])[-./](0?[1-9]|[12]\d|3[01])\b")

def _ok_date_iso(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except Exception:
        return False

@dataclass
class Check:
    rule: str
    ok: bool
    details: str = ""

def validate_fields(fields: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    out: List[Check] = []

    # 1) IBAN
    ib = fields.get("iban")
    if ib:
        out.append(Check("iban_format", bool(RE_IBAN_KZ.fullmatch(ib or "")),
                         f"iban={ib}"))
    else:
        # если IBAN не извлекли, но в тексте есть — подсказка
        hint = RE_IBAN_KZ.search(text or "")
        out.append(Check("iban_present_in_text", hint is not None,
                         f"candidate={hint.group(0) if hint else ''}"))

    # 2) BIC
    bic = fields.get("bic")
    if bic:
        out.append(Check("bic_format", bool(RE_BIC.fullmatch(bic or "")),
                         f"bic={bic}"))

    # 3) Дата ISO
    dt = fields.get("date")
    out.append(Check("date_iso_yyyy_mm_dd", bool(dt and _ok_date_iso(dt)),
                     f"date={dt}"))

    # 4) Валюта
    cur = (fields.get("currency") or "").upper()
    out.append(Check("currency_iso", cur in {"KZT","USD","EUR","RUB"}, f"currency={cur}"))

    # 5) Сумма: должна быть числом
    amt = fields.get("amount")
    ok_amt = False
    if amt:
        try:
            float(amt)
            ok_amt = True
        except Exception:
            pass
    out.append(Check("amount_numeric", ok_amt, f"amount={amt}"))

    # Пример согласованности: если в тексте встречается знак ₸ — ожидаем KZT
    if "₸" in (text or "") and cur and cur != "KZT":
        out.append(Check("currency_symbol_consistency", False, "Text contains '₸' but currency!=KZT"))
    else:
        out.append(Check("currency_symbol_consistency", True, ""))

    return [c.__dict__ for c in out]
