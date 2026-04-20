FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# no pycache files
ENV PYTHONDONTWRITEBYTECODE=1

# logs when crash
ENV PYTHONUNBUFFERED=1

WORKDIR /app

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

EXPOSE 6969 8000 5001 9090 3000

RUN run.sh

HEALTHCHECK CMD curl --fail http://localhost:6969 || exit 1
