FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

RUN python train.py

# Automatically use Render's assigned port (defaults to 10000 if not set)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
