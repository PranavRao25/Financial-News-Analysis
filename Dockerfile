FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# no pycache files
ENV PYTHONDONTWRITEBYTECODE=1

# logs when crash
ENV PYTHONUNBUFFERED=1

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN adduser --disabled-password --gecos '' appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
   
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

ENV PIP_HTTP_VERSION=1.1

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install --default-timeout=1000 --retries=10 -r requirements.txt

COPY src ./src/
COPY run.sh .
RUN chmod +x run.sh && chown -R appuser:appuser /app

EXPOSE 8501 5000 5001 5002 5003 8080 8000 9090 9093 9100 3000 6379

USER appuser

HEALTHCHECK CMD curl --fail http://localhost:8501 || exit 1

CMD ["./run.sh"]
