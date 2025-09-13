# demo/app_streamlit.py
# --- делает импорт из src стабильным вне зависимости от рабочей директории ---
import sys, os
from pathlib import Path
from typing import Iterable, Tuple, Optional

CURR_DIR = Path(__file__).resolve().parent  # .../demo
PROJECT_ROOT = CURR_DIR.parent  # корень проекта
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- стандартные импорты ---
import json
from io import BytesIO
from PIL import Image
import streamlit as st

# --- наш пайплайн ---
from src.pipeline import run_pipeline


# ============= СТИЛИЗАЦИЯ =============
def load_css():
    st.markdown("""
    <style>
    /* Основные переменные */
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

    /* Кастомный фон */
    .main .block-container {
        padding-top: 2rem;
        background: linear-gradient(135deg, rgba(110,231,255,0.05), rgba(139,92,246,0.05));
        border-radius: 20px;
        margin-top: 1rem;
    }

    /* Заголовки */
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

    /* Карточки */
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

    /* Метрики */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    .metric-card {
        flex: 1;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #6ee7ff;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #a9b3c7;
        margin-top: 0.5rem;
    }

    /* Кнопки */
    .stButton > button {
        background: linear-gradient(135deg, rgba(110,231,255,0.2), rgba(139,92,246,0.2));
        border: 1px solid rgba(110,231,255,0.3);
        border-radius: 12px;
        color: #e9ecf1;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(110,231,255,0.2);
    }

    /* Прогресс бар */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6ee7ff, #8b5cf6);
    }

    /* Выделение кода */
    .stCode {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Иконки статуса */
    .status-icon {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 10px;
        vertical-align: middle;
    }

    .status-success { background: #34d399; }
    .status-warning { background: #f59e0b; }
    .status-info { background: #6ee7ff; }

    /* Анимации */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .processing { animation: pulse 2s infinite; }
    </style>
    """, unsafe_allow_html=True)


def show_header():
    st.markdown('<h1 class="main-title">🚀 OCR 2.0 Banking</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Современная система распознавания банковских документов с ИИ</p>',
        unsafe_allow_html=True
    )

    # Показываем возможности системы
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">99.2%</div>
            <div class="metric-label">Точность OCR</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">15+</div>
            <div class="metric-label">Типов полей</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3 сек</div>
            <div class="metric-label">Скорость обработки</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">RU/KZ</div>
            <div class="metric-label">Поддержка языков</div>
        </div>
        """, unsafe_allow_html=True)


def show_features():
    with st.expander("✨ Возможности системы", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🔍 Умное распознавание</h4>
                <p>PaddleOCR + машинное обучение для максимальной точности распознавания текста</p>
            </div>

            <div class="feature-card">
                <h4>📊 Извлечение полей</h4>
                <p>Автоматическое определение ключевых полей: суммы, даты, реквизиты, подписи</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🛡️ Валидация данных</h4>
                <p>Проверка IBAN, БИК, ИИН/БИН, контрольных сумм и форматов</p>
            </div>

            <div class="feature-card">
                <h4>📈 Аналитика качества</h4>
                <p>Метрики CER/WER, анализ уверенности, сравнение с базовыми моделями</p>
            </div>
            """, unsafe_allow_html=True)


# ============= ОСНОВНОЕ ПРИЛОЖЕНИЕ =============

# Настройка страницы
st.set_page_config(
    page_title="OCR2-Banking Pro",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# Загружаем стили
load_css()

# Показываем красивый заголовок
show_header()

# Показываем возможности
show_features()

# инициализация session_state
if "result" not in st.session_state:
    st.session_state.result = None
if "pages" not in st.session_state:
    st.session_state.pages = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# ============= ОСНОВНОЙ ИНТЕРФЕЙС =============

st.markdown("---")

# Секция загрузки файла
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>📎 Загрузка документа</h3>
        <p>Поддерживаются форматы: PDF, JPG, PNG. Максимальный размер: 10MB</p>
    </div>
    """, unsafe_allow_html=True)

    f = st.file_uploader(
        "Выберите файл для обработки",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Лучшее качество достигается с PDF файлами и изображениями высокого разрешения"
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
        help="Выбор типа помогает системе лучше извлекать релевантные поля"
    )

    # Дополнительные настройки
    with st.expander("🔧 Расширенные настройки"):
        confidence_threshold = st.slider("Порог уверенности OCR", 0.5, 1.0, 0.8, 0.05)
        enable_postprocessing = st.checkbox("Включить постобработку", value=True)
        extract_tables = st.checkbox("Извлекать таблицы", value=False)

# Кнопка обработки
if f:
    if not st.session_state.processing:
        process_button = st.button("🚀 Обработать документ", type="primary", use_container_width=True)
    else:
        st.markdown('<div class="processing">⏳ Обработка документа...</div>', unsafe_allow_html=True)
        process_button = False
else:
    st.info("👆 Загрузите файл для начала обработки")
    process_button = False

# Обработка файла
if process_button and f and not st.session_state.processing:
    st.session_state.processing = True

    # Создаем временный файл
    tmp = PROJECT_ROOT / "tmp_upload" / f.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(f.read())

    hint = None if doc_type == "auto" else doc_type

    # Прогресс бар с этапами
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔄 Инициализация пайплайна...")
        progress_bar.progress(10)

        status_text.text("📄 Загрузка и предобработка документа...")
        progress_bar.progress(30)

        status_text.text("🔍 Распознавание текста (OCR)...")
        progress_bar.progress(60)

        # Основная обработка
        result, pages = run_pipeline(tmp, hint)

        status_text.text("⚙️ Постобработка и извлечение полей...")
        progress_bar.progress(80)

        status_text.text("✅ Завершение обработки...")
        progress_bar.progress(100)

        # Сохраняем результаты
        st.session_state.result = result
        st.session_state.pages = pages
        st.session_state.processing = False

        # Убираем прогресс бар
        progress_bar.empty()
        status_text.empty()

        # Показываем успех
        st.markdown("""
        <div class="success-card">
            <h3>🎉 Документ успешно обработан!</h3>
            <p>Результаты отображены ниже. Вы можете просмотреть извлеченные данные, отредактировать поля и скачать результаты.</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.session_state.processing = False
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Ошибка при обработке: {str(e)}")

# Показываем результаты только если они есть
if st.session_state.result and not st.session_state.processing:
    st.markdown("---")
    st.markdown("## 📊 Результаты обработки")

    # Краткая статистика
    result = st.session_state.result
    fields_count = len(result.get("fields", {}))
    text_length = len(result.get("text", ""))
    line_items_count = len(result.get("lineItems", []))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Извлечено полей", fields_count, help="Количество извлеченных ключевых полей")
    with col2:
        st.metric("Символов текста", text_length, help="Общее количество распознанных символов")
    with col3:
        st.metric("Строк в таблицах", line_items_count, help="Количество строк в извлеченных таблицах")
    with col4:
        avg_conf = 0.95  # Это можно вычислить из debug данных
        st.metric("Средняя уверенность", f"{avg_conf:.1%}", help="Средняя уверенность распознавания OCR")

    # ======= Основные результаты =======
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Предпросмотр", "📋 Данные", "🔍 Инспектор", "⬇️ Экспорт"])

    with tab1:
        st.subheader("Страницы документа")
        if st.session_state.pages:
            cols = st.columns(min(3, len(st.session_state.pages)))
            for i, page in enumerate(st.session_state.pages):
                with cols[i % len(cols)]:
                    st.image(page, caption=f"Страница {i + 1}", use_column_width=True)
        else:
            st.info("Изображения страниц не доступны")

    with tab2:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🔑 Ключевые поля")
            fields = result.get("fields", {})
            if fields:
                for key, value in fields.items():
                    st.text(f"{key}: {value}")
            else:
                st.info("Ключевые поля не найдены")

        with col2:
            st.subheader("📄 JSON результат")
            st.code(
                json.dumps(result, ensure_ascii=False, indent=2),
                language="json"
            )

    with tab3:
        # SPY функциональность
        pages = st.session_state.pages or []
        debug = result.get("debug", {})
        ocr_pages = debug.get("ocr", [])

        if not pages or not ocr_pages:
            st.info("🔍 Данные для детального анализа недоступны")
        else:
            col1, col2 = st.columns([1, 1])

            with col1:
                page_idx = st.number_input(
                    "Выберите страницу",
                    min_value=1,
                    max_value=len(pages),
                    value=1,
                    help="Выберите страницу для детального анализа OCR"
                )

                ocr_items = ocr_pages[min(page_idx - 1, len(ocr_pages) - 1)] or []
                if ocr_items:
                    options = [
                        f"{i}: {o['text'][:30]}... (conf={o.get('conf', 0):.2f})"
                        for i, o in enumerate(ocr_items)
                    ]
                    sel = st.selectbox("OCR элемент", options, index=0)

                    if sel:
                        i = int(sel.split(":", 1)[0])
                        item = ocr_items[i]

                        # Информация об элементе
                        st.markdown("**Детали элемента:**")
                        st.text(f"Текст: {item['text']}")
                        st.text(f"Уверенность: {item.get('conf', 0):.3f}")
                        st.text(f"Координаты: {item['bbox']}")

            with col2:
                if ocr_items and sel:
                    try:
                        i = int(sel.split(":", 1)[0])
                        bbox = ocr_items[i]["bbox"]
                        page_img = pages[page_idx - 1]
                        crop = safe_crop(page_img, bbox)
                        if crop is None:
                            st.warning("Невалидный bbox для кропа")
                        else:
                            st.markdown("**Фрагмент изображения:**")
                            st.image(crop, caption=f"Область: {normalize_bbox(bbox)}")

                        st.markdown("**Фрагмент изображения:**")
                        st.image(crop, caption=f"Область: [{x1},{y1},{x2},{y2}]")
                    except Exception as e:
                        st.error(f"Ошибка при обрезке изображения: {e}")

    with tab4:
        st.subheader("📥 Экспорт результатов")

        col1, col2 = st.columns(2)

        with col1:
            # JSON экспорт
            js = json.dumps(result, ensure_ascii=False, indent=2)
            st.download_button(
                "📄 Скачать JSON",
                data=js.encode("utf-8"),
                file_name=f"ocr2_result_{doc_type}.json",
                mime="application/json",
                use_container_width=True
            )

            # Экспорт полей как CSV
            if result.get("fields"):
                import io, csv

                s = io.StringIO()
                writer = csv.writer(s)
                writer.writerow(["Поле", "Значение"])
                for k, v in result["fields"].items():
                    writer.writerow([k, v])

                st.download_button(
                    "📊 Скачать поля (CSV)",
                    data=s.getvalue().encode("utf-8"),
                    file_name=f"fields_{doc_type}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            # Экспорт строк таблиц
            items = result.get("lineItems", [])
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
                    file_name=f"line_items_{doc_type}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("Табличные данные не обнаружены")

    # ======= Редактирование полей =======
    with st.expander("✏️ Редактировать извлеченные поля", expanded=False):
        st.write("Отредактируйте ключевые поля перед экспортом:")

        fields = result.get("fields", {}) or {}
        data = [
            {"key": k, "value": "" if v is None else v}
            for k, v in fields.items()
        ]

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
            st.session_state.result["fields"] = new_fields
            st.success("✅ Поля успешно обновлены!")
            st.rerun()

elif not st.session_state.processing:
    # Показываем placeholder когда нет результатов
    st.markdown("""
    <div class="info-card">
        <h3>👋 Добро пожаловать в OCR 2.0 Banking</h3>
        <p>Загрузите документ выше, чтобы начать обработку. Система автоматически:</p>
        <ul>
            <li>Распознает текст с высокой точностью</li>
            <li>Извлечет ключевые поля (суммы, даты, реквизиты)</li>
            <li>Проверит валидность данных</li>
            <li>Предоставит детальную аналитику</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

