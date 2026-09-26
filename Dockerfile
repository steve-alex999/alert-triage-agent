FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/models

COPY pyproject.toml README.md ./
COPY triage ./triage
RUN pip install --no-cache-dir .

# Bake the embedding model into the image so containers start without downloading it.
RUN python -c "from triage.embeddings import FastEmbedder; FastEmbedder()"

COPY data ./data
COPY eval.py ./

EXPOSE 8000
CMD ["uvicorn", "triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
