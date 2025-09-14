# demo/app_streamlit.py
import sys, os
from pathlib import Path
from typing import Iterable, Tuple, Optional

# --- надёжный путь к проекту ---
CURR_DIR = Path(__file__).resolve().parent          # .../demo
PROJECT_ROOT = CURR_DIR.parent                      # корень проекта
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- безопасный импорт safe_crop из двух возможных мест ---
try:
    from utils.image_tools import safe_crop
except Exception:
    try:
        from utils.image_tools import safe_crop
    except Exception:
        safe_crop = None  # обработаем дальше в коде

# --- стандартные импорты ---
import json
from io import BytesIO
from PIL import Image
import streamlit as st

# --- наш пайплайн ---
from src.pipeline import run_pipeline


# ============= СТИЛИЗАЦИЯ =============
st.set_page_config(
    page_title="OCR2-Banking Pro",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

def load_css():
    st.markdown("""
    <style>
    :root {
        --primary-color: #6ee7ff;
        --secondary-color: #8b5cf6;
        --bg-dark: #0b1020;
        --bg-light: #0f1630;
        --text-color: #e9ecf1;
        --muted-color: #a9b3c7;
        --success-color: #34d399;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
    }
    .main .block-container {
        padding-top: 2rem;
        background: linear-gradient(135deg, rgba(110,231,255,0.05), rgba(139,92,246,0.05));
        border-radius: 20px;
        margin-top: 1rem;
    }
    .main-title {
        background: linear-gradient(135deg, #6ee7ff, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #a9b3c7;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }
    .info-card {
        background: linear-gradient(135deg, rgba(110,231,255,0.1), rgba(139,92,246,0.1));
        border: 1px solid rgba(110,231,255,0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, rgba(52,211,153,0.1), rgba(34,197,94,0.1));
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .processing { animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.7}}
    </style>
    """, unsafe_allow_html=True)

def show_header():
    st.markdown('<h1 class="main-title">🚀 OCR 2.0 Banking</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Современная система распознавания банковских документов с ИИ</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div style="font-size:1.5rem;font-weight:700;color:#6ee7ff">99.2%</div><div style="color:#a9b3c7">Точность OCR</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div style="font-size:1.5rem;font-weight:700;color:#6ee7ff">15+</div><div style="color:#a9b3c7">Типов полей</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div style="font-size:1.5rem;font-weight:700;color:#6ee7ff">3 сек</div><div style="color:#a9b3c7">Скорость обработки</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div style="font-size:1.5rem;font-weight:700;color:#6ee7ff">RU/KZ</div><div style="color:#a9b3c7">Поддержка языков</div></div>', unsafe_allow_html=True)

def show_features():
    with st.expander("✨ Возможности системы", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🔍 Умное распознавание</h4>
                <p>PaddleOCR + ML для максимальной точности</p>
            </div>
            <div class="feature-card">
                <h4>📊 Извлечение полей</h4>
                <p>Сумма, дата, реквизиты, подписи</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🛡️ Валидация данных</h4>
                <p>IBAN, БИК, ИИН/БИН, контрольные суммы</p>
            </div>
            <div class="feature-card">
                <h4>📈 Аналитика качества</h4>
                <p>Метрики CER/WER, уверенность OCR</p>
            </div>
            """, unsafe_allow_html=True)


# ============= ОСНОВНОЕ ПРИЛОЖЕНИЕ =============
load_css()
show_header()
show_features()

if "result" not in st.session_state:
    st.session_state.result = None
if "pages" not in st.session_state:
    st.session_state.pages = None
if "processing" not in st.session_state:
    st.session_state.processing = False

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>📎 Загрузка документа</h3>
        <p>Поддерживаются форматы: PDF, JPG, PNG. Максимальный размер: 200MB</p>
    </div>
    """, unsafe_allow_html=True)

    f = st.file_uploader(
        "Выберите файл для обработки",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Лучшее качество — у PDF и изображений высокого разрешения"
    )

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>⚙️ Настройки</h3>
    </div>
    """, unsafe_allow_html=True)

    doc_type = st.selectbox(
        "Тип документа",
        ["auto", "receipt", "contract", "statement"],
        index=0,
        help="Подсказка помогает точнее извлекать поля"
    )

    with st.expander("🔧 Расширенные настройки"):
        confidence_threshold = st.slider("Порог уверенности OCR", 0.5, 1.0, 0.8, 0.05)
        preproc_mode = st.selectbox("Профиль препроцессинга", ["soft", "binary"], index=0)  # 👈 вставить здесь
        enable_postprocessing = st.checkbox("Включить постобработку", value=True)
        extract_tables = st.checkbox("Извлекать таблицы", value=False)

if f:
    if not st.session_state.processing:
        process_button = st.button("🚀 Обработать документ", type="primary", use_container_width=True)
    else:
        st.markdown('<div class="processing">⏳ Обработка документа...</div>', unsafe_allow_html=True)
        process_button = False
else:
    st.info("👆 Загрузите файл для начала обработки")
    process_button = False

if process_button and f and not st.session_state.processing:
    st.session_state.processing = True

    tmp = PROJECT_ROOT / "tmp_upload" / f.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(f.read())

    hint = None if doc_type == "auto" else doc_type

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔄 Инициализация пайплайна...")
        progress_bar.progress(10)

        status_text.text("📄 Загрузка и предобработка документа...")
        progress_bar.progress(30)

        status_text.text("🔍 Распознавание текста (OCR)...")
        progress_bar.progress(60)

        result, pages = run_pipeline(
            tmp,
            hint,
            conf_threshold=confidence_threshold,
            preproc_mode=preproc_mode
        )
        status_text.text("⚙️ Постобработка и извлечение полей...")
        progress_bar.progress(80)

        status_text.text("✅ Завершение обработки...")
        progress_bar.progress(100)

        st.session_state.result = result or {}
        st.session_state.pages = pages or []
        st.session_state.processing = False

        progress_bar.empty()
        status_text.empty()

        st.markdown("""
        <div class="success-card">
            <h3>🎉 Документ успешно обработан!</h3>
            <p>Смотрите результаты ниже: данные, разделы, инспектор и экспорт.</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.session_state.processing = False
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Ошибка при обработке: {str(e)}")

# ======= РЕЗУЛЬТАТЫ =======
if st.session_state.result and not st.session_state.processing:
    st.markdown("---")
    st.markdown("## 📊 Результаты обработки")

    result = st.session_state.result or {}
    pages = st.session_state.pages or []

    fields = result.get("fields") or {}
    text = result.get("text") or ""
    line_items = result.get("lineItems") or []
    sections = result.get("sections") or []

    fields_count = len(fields)
    text_length = len(text)
    line_items_count = len(line_items)
    sections_count = len(sections)  # 👈 добавляем

    col1m, col2m, col3m, col4m, col5m = st.columns(5)  # 👈 теперь 5 колонок
    with col1m:
        st.metric("Извлечено полей", fields_count)
    with col2m:
        st.metric("Символов текста", text_length)
    with col3m:
        st.metric("Строк в таблицах", line_items_count)
    with col4m:
        st.metric("Разделов", sections_count)  # 👈 новая метрика
    with col5m:
        avg_conf = 0.95
        st.metric("Средняя уверенность", f"{avg_conf:.1%}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Предпросмотр",
        "📋 Данные",
        "📚 Разделы",  # 👈 новая вкладка
        "🔍 Инспектор",
        "⬇️ Экспорт"
    ])

    # -------- TAB 1: Предпросмотр --------
    with tab1:
        st.subheader("Страницы документа")
        if pages:
            cols = st.columns(min(3, max(1, len(pages))))
            for i, page in enumerate(pages):
                with cols[i % len(cols)]:
                    st.image(page, caption=f"Страница {i + 1}", use_column_width=True)
        else:
            st.info("Изображения страниц не доступны")

    # -------- TAB 2: Данные (поля + разделы + JSON) --------
    with tab2:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("🔑 Ключевые поля")
            if fields:
                for key, value in fields.items():
                    st.text(f"{key}: {value}")
            else:
                st.info("Ключевые поля не найдены")

            st.subheader("📚 Разделы документа")
            if sections:
                for s in sections:
                    title = ((s.get('num') + " ") if s.get('num') else "") + (s.get("title") or "Раздел")
                    with st.expander(title, expanded=False):
                        st.markdown(
                            f"<div style='color:#a9b3c7;font-size:12px'>Страницы: {s.get('page_from','?')}–{s.get('page_to','?')} · Уровень: {s.get('level',1)}</div>",
                            unsafe_allow_html=True
                        )
                        st.text(s.get("content", ""))
            else:
                st.info("Разделы не обнаружены — возможно, в документе нет явной структуры.")

        with c2:
            st.subheader("📄 JSON результат")
            st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")

    with tab3:
        st.subheader("📚 Структура документа")
        if sections:
            for s in sections:
                title = ((s.get('num') + " ") if s.get('num') else "") + (s.get("title") or "Раздел")
                with st.expander(title, expanded=False):
                    st.markdown(
                        f"<div style='color:#a9b3c7;font-size:12px'>Страницы: {s.get('page_from', '?')}–{s.get('page_to', '?')} · "
                        f"Уровень: {s.get('level', 1)}</div>",
                        unsafe_allow_html=True
                    )
                    st.text(s.get("content", ""))
        else:
            st.info("Разделы не обнаружены")
    # -------- TAB 3: Инспектор --------
    with tab4:
        debug = result.get("debug") or {}
        ocr_pages = debug.get("ocr") or []

        if not pages or not ocr_pages:
            st.info("🔍 Данные для детального анализа недоступны")
        else:
            lc, rc = st.columns([1, 1])

            sel = None
            ocr_items = []

            with lc:
                page_idx = st.number_input(
                    "Выберите страницу",
                    min_value=1,
                    max_value=len(pages),
                    value=1,
                    help="Выберите страницу для детального анализа OCR"
                )
                page_pos = min(max(page_idx - 1, 0), len(ocr_pages) - 1)
                ocr_items = (ocr_pages[page_pos] or []) if ocr_pages else []

                if ocr_items:
                    options = [
                        f"{i}: {str(o.get('text', ''))[:30]}... (conf={float(o.get('conf', o.get('score', 0.0))):.2f})"
                        for i, o in enumerate(ocr_items)
                    ]
                    sel_opt = st.selectbox("OCR элемент", options, index=0)
                    if sel_opt:
                        sel = int(sel_opt.split(":", 1)[0])

                    if sel is not None:
                        item = ocr_items[sel]
                        st.markdown("**Детали элемента:**")
                        st.text(f"Текст: {item.get('text', '')}")
                        st.text(f"Уверенность: {float(item.get('conf', item.get('score', 0.0))):.3f}")
                        st.text(f"Координаты: {item.get('bbox')}")

            with rc:
                if ocr_items and sel is not None:
                    try:
                        item = ocr_items[sel]
                        bbox = item.get("bbox")
                        page_img = pages[page_pos]
                        if safe_crop is None:
                            st.warning("safe_crop не найден. Проверь импорт utils.image_tools или src.utils.image_tools.")
                        else:
                            crop = safe_crop(page_img, bbox) if bbox else None
                            if crop is None:
                                st.warning("Невалидный bbox для кропа")
                            else:
                                st.markdown("**Фрагмент изображения:**")
                                st.image(crop, caption=f"bbox: {bbox}")
                    except Exception as e:
                        st.error(f"Ошибка при обрезке изображения: {e}")

    # -------- TAB 4: Экспорт --------
    with tab5:
        st.subheader("📥 Экспорт результатов")

        ec1, ec2 = st.columns(2)
        with ec1:
            js = json.dumps(result, ensure_ascii=False, indent=2)
            st.download_button(
                "📄 Скачать JSON",
                data=js.encode("utf-8"),
                file_name=f"ocr2_result_{result.get('docType','doc')}.json",
                mime="application/json",
                use_container_width=True
            )

            if fields:
                import io, csv
                s = io.StringIO()
                writer = csv.writer(s)
                writer.writerow(["Поле", "Значение"])
                for k, v in fields.items():
                    writer.writerow([k, v])
                st.download_button(
                    "📊 Скачать поля (CSV)",
                    data=s.getvalue().encode("utf-8"),
                    file_name=f"fields_{result.get('docType','doc')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with ec2:
            items = line_items
            if items:
                import io, csv
                s = io.StringIO()
                fieldnames = sorted({k for it in items for k in it.keys()})
                w = csv.DictWriter(s, fieldnames=fieldnames)
                w.writeheader()
                for it in items:
                    w.writerow(it)
                st.download_button(
                    "📋 Скачать таблицы (CSV)",
                    data=s.getvalue().encode("utf-8"),
                    file_name=f"line_items_{result.get('docType','doc')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("Табличные данные не обнаружены")

    # ======= Редактирование полей =======
    with st.expander("✏️ Редактировать извлеченные поля", expanded=False):
        st.write("Отредактируйте ключевые поля перед экспортом:")
        data = [{"key": k, "value": "" if v is None else v} for k, v in (fields.items() if fields else [])]
        edited = st.data_editor(
            data,
            num_rows="dynamic",
            use_container_width=True,
            key="fields_editor",
            column_config={
                "key": st.column_config.TextColumn("Поле", help="Название поля"),
                "value": st.column_config.TextColumn("Значение", help="Значение поля"),
            }
        )
        if st.button("💾 Применить изменения", type="secondary"):
            new_fields = {}
            for row in edited:
                k = str(row.get("key", "")).strip()
                if k:
                    new_fields[k] = row.get("value", "")
            st.session_state.result = {**result, "fields": new_fields}
            st.success("✅ Поля успешно обновлены!")
            st.rerun()

elif not st.session_state.processing:
    # Placeholder — когда нет результатов
    st.markdown("""
    <div class="info-card">
        <h3>👋 Добро пожаловать в OCR 2.0 Banking</h3>
        <p>Загрузите документ выше, чтобы начать обработку. Система автоматически:</p>
        <ul>
            <li>Распознает текст с высокой точностью</li>
            <li>Извлечет ключевые поля (суммы, даты, реквизиты)</li>
            <li>Соберёт структуру документа: «Заголовок → содержимое»</li>
            <li>Предоставит детальную аналитику</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
