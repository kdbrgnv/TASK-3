import re
from datetime import datetime
from typing import Optional, Dict

KZ_IBAN_RE = re.compile(r"\bKZ\d{20}\b")
BIC_RE = re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b")

def norm_amount(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    s = x.replace(" ", "").replace("\u00A0", "").replace(",", ".")
    m = re.findall(r"\d+\.?\d*", s)
    return m[-1] if m else None

def norm_currency(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    t = t.upper()
    if "₸" in t or "KZT" in t:
        return "KZT"
    for c in ("USD", "EUR", "RUB", "KZT"):
        if c in t:
            return c
    return None

def norm_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y.%m.%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    m = re.search(r"(20\d{2})[-./](\d{1,2})[-./](\d{1,2})", s)
    if m:
        y, mo, da = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(da):02d}"
    return None

def fix_fields(raw: Optional[Dict]) -> Dict:
    out = dict(raw or {})
    if iban := out.get("iban"):
        m = KZ_IBAN_RE.search(iban.replace(" ", ""))
        out["iban"] = m.group(0) if m else None
    if bic := out.get("bic"):
        m = BIC_RE.search(bic)
        out["bic"] = m.group(0) if m else None
    out["amount"] = norm_amount(out.get("amount"))
    out["currency"] = norm_currency(out.get("currency") or "")
    out["date"] = norm_date(out.get("date"))
    if iin := out.get("iin_bin"):
        m = re.search(r"\b\d{12}\b", iin)
        out["iin_bin"] = m.group(0) if m else None
    return {k: v for k, v in out.items() if v}
