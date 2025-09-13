import os, json, re
from pydantic import BaseModel

class Fields(BaseModel):
    date: str | None = None
    totalAmount: str | None = None
    currency: str | None = None
    iban: str | None = None
    bin: str | None = None
    merchant: str | None = None

SYSTEM_PROMPT = (
    "You are a strict data normalizer for bank documents. "
    "Return ONLY a JSON object with 'fields'. Dates -> YYYY-MM-DD; amounts -> 1234.56; currencies -> ISO alpha."
    )

class LLMClient:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def map_to_fields(self, doc_type: str, raw_text: str, hints: dict) -> dict:
        schema = Fields.model_json_schema()["properties"]
        prompt = (
        f"Document type: {doc_type}"
        f"Raw text (truncated):{raw_text[:6000]}"
        f"Hints: {json.dumps(hints, ensure_ascii=False)}"
        f"Schema fields: {json.dumps(schema, ensure_ascii=False)}"
        "Return JSON: {\"fields\":{...}} only." )
        resp = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
        temperature=0
        )
        txt = resp.choices[0].message.content
        try:
            return json.loads(txt)
        except Exception:
            return {"fields": {}}