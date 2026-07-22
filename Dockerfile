# Hugging Face Spaces — Hybrid Face Recognition System
FROM python:3.10-slim

WORKDIR /app

# OpenCV / image system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# HF Spaces expects port 7860
ENV PORT=7860
ENV TF_CPP_MIN_LOG_LEVEL=2
EXPOSE 7860

# Single worker — TensorFlow is memory-heavy
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--threads", "2", "--timeout", "180", "app:app"]
