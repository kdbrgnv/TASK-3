# OCR 2.0 для банковских документов — README

## 🚀 О проекте
Интеллектуальная OCR-система для банковских документов (чеки, договоры, выписки) с пайплайном предобработки, распознавания (PaddleOCR) и пост-обработки (извлечение ключевых полей, валидации, LLM-подсказки). Демо — на Streamlit.

## 🧩 Структура
```
TASK-2/
├─ demo/                 # Streamlit UI
├─ src/                  # пайплайн, OCR, пост-обработка, утилиты
├─ utils/                # вспомогательные функции (crop и т.п.)
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml    # (см. ниже — пример)
└─ README.md
```

## 📋 Требования
- Python 3.10+ (реком. 3.10)
- macOS / Linux / Windows
- (Опц.) Tesseract для baseline
- Для PDF — `pymupdf`

## 🔑 Переменные окружения (LLM)
Проект поддерживает OpenAI и Google Gemini (через `google-generativeai`).  
Задаём **провайдера** и **модель** через env-переменные:

- Общие:
  - `LLM_PROVIDER` — `openai` или `gemini`
- OpenAI:
  - `OPENAI_API_KEY` — токен OpenAI
  - `OPENAI_MODEL` — напр. `gpt-4o-mini`
- Gemini:
  - `GOOGLE_API_KEY` — токен Google AI Studio
  - `GEMINI_MODEL` — напр. `gemini-1.5-pro` или `gemini-1.5-flash`

### `.env` (пример)
```
# выбор провайдера
LLM_PROVIDER=gemini

# OpenAI
OPENAI_API_KEY=sk-********************************
OPENAI_MODEL=gpt-4o-mini

# Google Gemini
GOOGLE_API_KEY=AIza********************************
GEMINI_MODEL=gemini-1.5-pro
```

### Экспорт токенов (bash/zsh)
```bash
# если используешь OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-...your_key..."
export OPENAI_MODEL="gpt-4o-mini"

# если используешь Gemini
export LLM_PROVIDER=gemini
export GOOGLE_API_KEY="AIza...your_key..."
export GEMINI_MODEL="gemini-1.5-pro"
```

---

## ▶️ Локальный запуск
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run demo/app_streamlit.py
```
По умолчанию UI будет на `http://localhost:8501`.

---

## 🐳 Docker

### Сборка
```bash
docker build -t ocr-app .
```

### Запуск
```bash
docker run --rm -p 8501:8501   -e LLM_PROVIDER=gemini   -e GOOGLE_API_KEY=AIza...   -e GEMINI_MODEL=gemini-1.5-pro   -e OPENAI_API_KEY=sk-...   -e OPENAI_MODEL=gpt-4o-mini   ocr-app
```

---

## 🧩 docker-compose
```yaml
version: "3.9"
services:
  ocr-app:
    build:
      context: .
      dockerfile: Dockerfile
    image: ocr-app:latest
    ports:
      - "8501:8501"
    env_file:
      - .env
    restart: unless-stopped
```

Запуск:
```bash
docker compose up --build
```

---

## ⚙️ Конфигурация пайплайна
- Язык OCR: `ru`
- Для PDF используй `pymupdf` (рендер 200–300 DPI)

---

## 🧯 Траблшутинг
- **`requirements.txt: not found`** — запускай `docker build` из корня
- **`ModuleNotFoundError: No module named 'src'`** — запускай `streamlit run` из корня
- **`paddleocr` не виден** — проверь, что активировал `.venv`

---

## ✅ Чек-лист
- [ ] docker build и run работают
- [ ] .env заполнен
- [ ] UI открывается на 8504
