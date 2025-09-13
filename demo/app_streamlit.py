# demo/app_streamlit.py
# --- делает импорт из src стабильным вне зависимости от рабочей директории ---
import sys, os
from pathlib import Path

CURR_DIR = Path(__file__).resolve().parent         # .../demo
PROJECT_ROOT = CURR_DIR.parent                     # корень проекта
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- стандартные импорты ---
import json
from io import BytesIO
from PIL import Image
import streamlit as st

# --- наш пайплайн ---
from src.pipeline import run_pipeline

st.set_page_config(page_title="OCR2-Banking (Strict)", layout="wide")
st.title("OCR2-Banking — Demo (Donut • PaddleOCR • LLM)")

# инициализация session_state
if "result" not in st.session_state:
    st.session_state.result = None
if "pages" not in st.session_state:
    st.session_state.pages = None

# загрузка и тип
f = st.file_uploader("Загрузите PDF/JPG/PNG", type=["pdf", "jpg", "jpeg", "png"])
doc_type = st.selectbox(
    "Тип документа (подсказка)",
    ["auto", "receipt", "contract", "statement"],
    index=0
)

# кнопка обработать
if f and st.button("Обработать"):
    tmp = PROJECT_ROOT / "tmp_upload" / f.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(f.read())
    hint = None if doc_type == "auto" else doc_type

    with st.spinner("Обработка…"):
        result, pages = run_pipeline(tmp, hint)

    st.session_state.result = result
    st.session_state.pages = pages
    st.success("Готово! Результат ниже.")

# если результата ещё нет — мягкая подсказка
if not st.session_state.result:
    st.info("Загрузите файл и нажмите «Обработать».")
    st.stop()

# ======= Preview & JSON =======
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    st.subheader("Предпросмотр")
    for i, p in enumerate(st.session_state.pages or []):
        st.image(p, caption=f"Страница {i+1}")

with c2:
    st.subheader("JSON")
    st.code(
        json.dumps(st.session_state.result, ensure_ascii=False, indent=2),
        language="json"
    )

# ======= Spy / Edit / Download panel =======
with st.expander("🔍 Spy • ✏️ Edit • ⬇️ Download", expanded=True):
    tab_spy, tab_edit, tab_dl = st.tabs(["🔍 Spy", "✏️ Edit", "⬇️ Download"])

    # --- 🔍 Spy ---
    with tab_spy:
        pages = st.session_state.pages or []
        result = st.session_state.result or {}
        debug = (result.get("debug") or {})
        ocr_pages = debug.get("ocr") or [[]]

        if not pages or not ocr_pages:
            st.info("Нет данных для инспекции.")
        else:
            page_idx = st.number_input(
                "Страница", min_value=1, max_value=len(pages), value=1
            )
            ocr_items = ocr_pages[min(page_idx - 1, len(ocr_pages) - 1)] or []
            if ocr_items:
                options = [
                    f"{i}: {o['text']} (conf={o.get('conf', 0):.2f})"
                    for i, o in enumerate(ocr_items)
                ]
                sel = st.selectbox("OCR элемент", options, index=0)
                i = int(sel.split(":", 1)[0])
                x1, y1, x2, y2 = ocr_items[i]["bbox"]
                crop = pages[page_idx - 1].crop((x1, y1, x2, y2))
                st.image(crop, caption=f"Crop: [{x1},{y1},{x2},{y2}]")
            else:
                st.info("OCR элементы не найдены для этой страницы.")

    # --- ✏️ Edit ---
    with tab_edit:
        st.write("Отредактируйте ключевые поля перед выгрузкой.")
        fields = (st.session_state.result or {}).get("fields", {}) or {}

        data = [
            {"key": k, "value": "" if v is None else v}
            for k, v in fields.items()
        ]
        edited = st.data_editor(
            data, num_rows="dynamic", use_container_width=True, key="fields_editor"
        )

        if st.button("Применить изменения"):
            new_fields = {}
            for row in edited:
                k = str(row.get("key", "")).strip()
                if k:
                    new_fields[k] = row.get("value", "")
            st.session_state.result["fields"] = new_fields
            st.success("Поля обновлены. См. JSON справа.")

    # --- ⬇️ Download ---
    with tab_dl:
        js = json.dumps(st.session_state.result, ensure_ascii=False, indent=2)
        st.download_button(
            "Скачать JSON",
            data=js.encode("utf-8"),
            file_name="ocr2_result.json",
            mime="application/json"
        )

        items = (st.session_state.result or {}).get("lineItems")
        if items:
            import io, csv
            s = io.StringIO()
            fieldnames = sorted({k for it in items for k in it.keys()})
            w = csv.DictWriter(s, fieldnames=fieldnames)
            w.writeheader()
            for it in items:
                w.writerow(it)
            st.download_button(
                "Скачать lineItems.csv",
                data=s.getvalue().encode("utf-8"),
                file_name="lineItems.csv",
                mime="text/csv"
            )
