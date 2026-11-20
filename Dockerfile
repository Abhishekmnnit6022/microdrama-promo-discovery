# Container for Hugging Face Spaces (Docker SDK) — also works with `docker build . && docker run -p 7860:7860`.
FROM python:3.12-slim

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download both pretrained models at build time so "Process all episodes" works without a cold download.
RUN python -c "from transformers import pipeline; \
pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base'); \
pipeline('zero-shot-classification', model='MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33')"

COPY --chown=user . .

EXPOSE 7860
# XSRF/CORS off: required for st.file_uploader inside the Hugging Face page iframe.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--browser.gatherUsageStats=false"]
