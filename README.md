# OCR 2.0 для банковских документов — README

## 🚀 О проекте
Цель — разработать **интеллектуальную OCR‑систему** для банковских документов (чеки, договоры, выписки), которая **превосходит baseline (Tesseract)** по качеству извлечения текста и ключевых полей. Репозиторий содержит:
- **baseline** (Tesseract, rule‑based шаблоны);
- **улучшенную пайплайн‑систему** на базе **PaddleOCR** + предобработка изображений + пост‑обработка;
- **демо‑приложение** на Streamlit;
- **скрипты оценки** на открытом тестовом наборе.

> Кейс соответствует «ТЗ 2. OCR 2.0 для банковских документов». Итоговая метрика сравнивается с baseline на открытом + скрытом наборах.


## 🧩 Структура проекта
```
TASK-2/
├─ demo/
│  └─ app_streamlit.py             # демо UI
├─ src/
│  ├─ pipeline.py                  # высокоуровневый конвейер
│  ├─ ocr_paddle.py                # обёртка над PaddleOCR
│  ├─ ocr_tesseract.py             # baseline на pytesseract
│  ├─ preproc.py                   # предобработка (бинаризация, выравнивание, шумоподавление)
│  ├─ post_llm.py                  # пост-обработка (нормализация, проверки, LLM-подсказки при необходимости)
│  ├─ pdf_utils.py                 # (опц.) обработка PDF через PyMuPDF (fitz)
│  └─ schemas.py                   # типы данных/валидаторы для извлечённых полей
├─ requirements.txt
├─ config.yaml                     # настройки проекта (язык, модели, пороги, пути)
└─ README.md                       # этот файл
```

**Важное соглашение:** импорт модулей идёт из пакета `src`. Если запускаете скрипты из корня, это работает сразу. Для запуска из подпапок (например, `demo/`) используем _добавление корня в `PYTHONPATH`_ или правим `sys.path` (см. ниже).


## 🛠 Требования
- **Python 3.12+** (минимум 3.10; в 3.9 нет поддержки `X | Y` для типов → ошибка вида _“Python version 3.9 does not allow writing union types as X | Y”_);
- macOS / Linux / Windows;
- Виртуальное окружение `venv` или `conda`.

### Базовые зависимости
```
paddleocr>=2.7.0
paddlepaddle>=2.6.0          # CPU-версия; для GPU смотрите документацию Paddle
pytesseract>=0.3.10
opencv-python>=4.9.0.80
numpy, pillow, pydantic
python-dotenv, pyyaml, loguru
streamlit
# опционально, если нужны PDF:
pymupdf
```
> Для macOS / Apple Silicon иногда требуется ставить зависимости через `conda` (особенно `opencv`).


## 📦 Установка (чистая)
```bash
# 0) из корня проекта
python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -V                           # проверьте, что Python 3.12+
pip install --upgrade pip

# 1) установить зависимости
pip install -r requirements.txt

# 2) (опционально) Tesseract для baseline
#   macOS (brew):    brew install tesseract
#   Ubuntu/Debian:   sudo apt-get install tesseract-ocr tesseract-ocr-rus

# 3) (опционально) PaddlePaddle GPU — смотрите оф. инструкцию под вашу CUDA/cuDNN
```

### Быстрая проверка PaddleOCR
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='ru')    # русская модель; можно 'en', 'ru', 'en+ru'
```
Если **`ModuleNotFoundError: No module named 'paddleocr'`**, убедитесь, что вы активировали `.venv` и что установка прошла без ошибок.


## ⚙️ Конфигурация
Файл **`config.yaml`** (пример):
```yaml
language: "ru"            # 'ru' | 'en' | 'en+ru'
use_gpu: false
preproc:
  denoise: true
  deskew: true
  adaptive_threshold: true
ocr:
  engine: "paddle"        # 'paddle' | 'tesseract'
  cls: true               # классификатор ориентации
postproc:
  normalize_spaces: true
  fix_common_mistakes: true
  enable_field_parsing: true
eval:
  dataset_dir: "data/public_eval"
  metrics: ["cer", "wer", "field_f1"]
```

## 🧠 Архитектура пайплайна
1. **Загрузка** изображения / PDF‑страницы (`pdf_utils.extract_pages`).
2. **Предобработка** (`preproc.py`): выравнивание, бинаризация, шумоподавление, увеличение контраста.
3. **OCR**:
   - **baseline:** `pytesseract.image_to_data` (с языком `rus` при наличии);
   - **improved:** `PaddleOCR(lang='ru', cls=True)` с пост‑фильтрацией боксов и confidence.
4. **Пост‑обработка** (`post_llm.py`, `schemas.py`):
   - нормализация пробелов и символов;
   - извлечение ключевых полей (ФИО, ИИН/БИН, суммы, даты, IBAN/BIC, назначение платежа);
   - валидация по regex/чек‑суммам;
   - (опц.) LLM‑подсказки для восстановления фрагментов (ограничиваем контекстом, логами и таймаутами).

5. **Оценка**: CER/WER по тексту, F1 по полям (точность, полнота).

Диаграмма данных (упрощённо):
```
Image/PDF → Preproc → OCR → Postproc (fields) → JSON (text, boxes, fields, confidences)
```


## ▶️ Запуск демо (Streamlit)
Из **корня проекта** (чтобы `src` корректно резолвился):
```bash
streamlit run demo/app_streamlit.py
```
Если запускаете из `demo/` и видите `ModuleNotFoundError: No module named 'src'`:
- Либо запускайте **из корня**;
- Либо добавьте в начало `demo/app_streamlit.py`:
  ```python
  import sys, os
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
  ```

## 🧪 Сравнение с baseline
### 1) Запуск baseline (Tesseract)
```bash
python -m src.pipeline --engine tesseract --input data/samples --out out/baseline
```
### 2) Запуск улучшенного пайплайна (PaddleOCR)
```bash
python -m src.pipeline --engine paddle --lang ru --input data/samples --out out/paddle
```
### 3) Оценка качества
```bash
python -m src.eval --pred out/paddle --gt data/labels --metrics cer wer field_f1
```
Результаты сохраняются в `out/metrics.json` и логах.


## 🗣 Поддержка русского языка
- **PaddleOCR**: используйте `PaddleOCR(lang="ru", cls=True)` — это включает русскую разметку и классификатор ориентации страниц.
- **Tesseract**: установите языковой пакет `rus` (см. установку выше) и в коде задайте `lang='rus'`.

**Если «ничего не изменилось насчёт русского языка»:**
1. Проверьте, что модель/пакет для `ru` действительно установлен;
2. Убедитесь, что входные изображения **достаточно крупные** (минимум ~120 DPI) — слишком маленькие сканы ломают распознавание;
3. Включите предобработку: выравнивание, бинаризацию, повышение резкости;
4. Логику `postproc` держите **языко‑зависимой** (кириллица vs латиница, замены `О/0`, `В/8`, `С/С`, и т.д.).


## 📄 Где используется PyMuPDF (pymupdf)
Если в проекте есть PDF, используем `src/pdf_utils.py` (пример):
```python
import fitz  # pymupdf

def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=200)  # 200-300 DPI — хороший баланс
        yield Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
```
→ Затем каждую страницу гоняем по конвейеру (`preproc → ocr → postproc`).


## 🧯 Траблшутинг

### 1) **Python 3.9 union types** / сообщение «Python version 3.9 does not allow writing union types as X | Y»
- Вы уже **обновили Python**, но окружение всё ещё указывает на 3.9.
  - Деактивируйте и **пересоздайте** venv на новой версии:
    ```bash
    deactivate || true
    rm -rf .venv
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -V  # должен быть 3.12.x
    pip install -r requirements.txt
    ```
  - Убедитесь, что IDE/терминал запускает команды именно из **активированного** окружения.

### 2) `ModuleNotFoundError: No module named 'paddleocr'`
- Активируйте venv и переустановите:
  ```bash
  pip install --upgrade pip
  pip install paddlepaddle paddleocr
  ```
- Для Apple Silicon и Linux без AVX иногда помогает установка через `conda`/архивы (смотрите оф. документацию Paddle).

### 3) `ModuleNotFoundError: No module named 'src'`
- Запускайте команды **из корня** или добавляйте корень в `PYTHONPATH`/`sys.path` (см. выше).

### 4) Ошибки при установке `opencv`
- macOS: попробуйте `pip install opencv-python` в чистом окружении; при проблемах — `conda install -c conda-forge opencv`.

### 5) Медленная/нестабильная работа PaddleOCR
- Выключите GPU (если нет корректной CUDA): `use_gpu: false`;
- Снизьте DPI рендера для PDF (например, 200 DPI);
- Ограничьте максимальную ширину изображения в предобработке.


## 🧱 Пример кода: `src/ocr_paddle.py`
```python
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

class PaddleEngine:
    def __init__(self, lang: str = "ru", use_gpu: bool = False):
        self.ocr = PaddleOCR(lang=lang, use_gpu=use_gpu, cls=True)

    def run(self, img: Image.Image):
        arr = np.array(img)
        res = self.ocr.ocr(arr, cls=True)
        out = []
        if res and res[0]:
            for det, (text, conf) in res[0]:
                x1,y1,x2,y2,x3,y3,x4,y4 = [int(v) for p in det for v in p]
                out.append({
                    "bbox": [x1,y1,x2,y2,x3,y3,x4,y4],
                    "text": text,
                    "conf": float(conf)
                })
        return out
```

## 🧪 Пример запуска пайплайна
```bash
python -m src.pipeline \
  --engine paddle \
  --lang ru \
  --input data/samples \
  --out out/run-$(date +%Y%m%d-%H%M)
```

## 📊 Метрики
- **CER/WER** — символьная/словесная ошибка распознавания;
- **Field F1** — F1‑мера по извлечённым ключевым полям (exact match/normalized match);
- Отчёты: `out/metrics.json`, `out/report.html` (опц.).


## 🧰 Полезные советы
- Сохраняйте **промежуточные картинки** предобработки (до/после) — это сильно помогает дебагу;
- Логируйте **confidence** и фильтруйте слишком низкие значения;
- Для таблиц (выписки) используйте морфологию/линейное выделение в OpenCV;
- Для чеков — нормализация перспективы + усиление контраста;
- Для дат/сумм — регулярки + контрольные суммы (IBAN/BIC/ИИН/БИН).


## 📄 Лицензирование и данные
- Используйте только **публичные** или **разрешённые** наборы банковских форм/чеков;
- Удаляйте персональные данные из репортов/примеров.


## 🧾 История типичных вопросов/ошибок
- «**Я поставил ру‑модель, но русский всё равно не читается**» → проверьте DPI/размер, предобработку, наличие `lang='ru'`, отсутствие перевёрнутых страниц (включите `cls=True`).
- «**Обновил Python, но проект всё ещё ругается на 3.9**» → почти всегда активировано старое окружение; пересоздайте `.venv` и посмотрите `python -V`.
- «**Где используется `pymupdf`?**» → в `src/pdf_utils.py` для рендера страниц PDF в изображения с нужным DPI перед OCR.
- «**`ModuleNotFoundError: No module named 'src'`**» → запускайте из корня или добавьте путь, как показано в разделе «Запуск демо».


## ✅ Чек‑лист перед сдачей
- [ ] Скрипт запуска пайплайна и демо работает на чистой машине (инструкции повторяемы);
- [ ] Оценка на открытом наборе ≥ baseline по CER/WER и Field‑F1;
- [ ] В отчёте описаны ошибки/уголки кейсов и как вы их улучшали;
- [ ] Есть примеры вход → предобработка → боксы → текст → финальные поля → метрики.

