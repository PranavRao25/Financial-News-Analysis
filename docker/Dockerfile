FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# no pycache files
ENV PYTHONDONTWRITEBYTECODE=1

# logs when crash
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN adduser --disabled-password --gecos '' appuser

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

COPY src ./src/
COPY run.sh .
RUN chmod +x run.sh && chown -R appuser:appuser /app

EXPOSE 8501 5000 5001 5002 5003 8080 8000 9090 9093 9100 3000 6379

USER appuser

HEALTHCHECK CMD curl --fail http://localhost:8501 || exit 1

CMD ["./run.sh"]
