FROM python:3.13-slim
# System deps (needed by some python wheels / SSL)
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1


# Hugging Face cache location inside container (so we can mount it)
ENV HF_HOME=/data/hf \
  TRANSFORMERS_CACHE=/data/hf \
  SENTENCE_TRANSFORMERS_HOME=/data/hf

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
