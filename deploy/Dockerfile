FROM python:3.10-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir torch==2.2.* transformers paddleocr pymupdf pillow numpy streamlit openai
ENV OPENAI_MODEL=gpt-4o-mini
EXPOSE 8501
CMD ["streamlit","run","demo/app_streamlit.py","--server.port","8501","--server.address","0.0.0.0"]