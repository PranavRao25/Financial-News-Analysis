import uuid
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Info, Summary
from werkzeug.utils import secure_filename
import os
import requests
import json
from uuid import uuid4
import io
from io import StringIO
import psutil
import zipfile
import cv2
import pytesseract
from PIL import Image
import yaml
from pathlib import Path
import logging

print("Financial News Analysis app started")

app = Flask(__name__)
parent = Path(__file__).resolve().parent.parent

with open(parent / "config.yaml", 'r') as f:
    configs = yaml.full_load(f)

# title - Financial News Analysis
# Sentiment Analysis & Topic Modelling

def init_metrics_server(port=8000):
    print(f"Metrics server on port = {port}")
    start_http_server(port)
    return True

def get_metrics():
    return {
        "requests": Counter(
            "app_total_requests",
            "Total number of inference requests made",
            labelnames=["mode"]
        ),
        "errors": Counter(
            "app_total_errors",
            "Total number of exceptions",
            labelnames=["mode", "error_types"]
        ),
        "active_requests": Gauge(
            "app_active_requests_current",
            "Number of requests being currently processed",
            labelnames=["session_id"]
        ),
        "model_memory_usage": Gauge(
            "app_model_memory_bytes",
            "Current usage of RSS memory by the app"
        ),
        "inference_latency": Histogram(
            "app_inference_latency_seconds",
            "Latency of the app in seconds",
            labelnames=["mode", "model_type"]
        )
    }

def init_topic_model():
    print("Init topic model")
    serve_port = configs["deployment"]["topic_serve"]
    MLFLOW_URI = f"http://127.0.0.1:{serve_port}/invocations"
    return MLFLOW_URI

def init_sentiment_model():
    print("Init Sentiment model")
    # sentiment = SentimentModel(path)
    serve_port = configs["deployment"]["sent_serve"]
    MLFLOW_URI = f"http://127.0.0.1:{serve_port}/invocations"
    return MLFLOW_URI

def analyse(file, ext):
    if ext in {"jpg", "jpeg", "png"}:
        img = Image.open(file.stream).convert("RGB")
        txt = str(pytesseract.image_to_string(img))  # check for language also
    elif ext in {"pdf"}:
        txt = ""
    elif ext in {"zip"}:
        txt = ""
    elif ext in {"txt"}:
        txt = file.read().decode("utf-8")

    if not txt.strip():  # text empty
        raise ValueError("Text empty")
    
    sent_label, sent_conf = infer(txt, "sentiment", sent_model_name, sent_uri, sent_mapping)
    topic_label, topic_conf = infer(txt, "topic", topic_model_name, topic_uri, topic_mapping)

    return {
        "sentiment": {"label": sent_label, "confidence": sent_conf, "model": sent_model_name},
        "topic": {"label": topic_label, "confidence": topic_conf, "model": topic_model_name},
        "text": txt
    }

def infer(txt, mode, model_name, uri, mapping):
    metrics["requests"].labels(mode=mode).inc() # type: ignore

    label = "Unknown"
    confidence = 0.0

    try:
        with metrics["inference_latency"].labels(mode=mode, model_type=model_name).time(): # type: ignore
            payload = {"inputs": [txt]}
            headers = {"Content-Type": "application/json"}

            response = requests.post(uri, json=payload, headers=headers)

            if response.status_code != 200:
                raise Exception(f"MLFlow API Error {response.status_code}: {response.text}")

            results = response.json().get("predictions", response.json())

            for item in results:
                label = str(mapping[str(item["predicted_class"])])
                confidence = item["confidence"]
    except Exception as e:
        error_name = type(e).__name__
        # st.error(f"Error in processing : {error_name}")
        # st.write(metrics)
        # st.write(mode)
        logging.error(f"Error during {mode} inference: {e}")
        metrics["errors"].labels(mode=mode, error_types=error_name).inc() # type: ignore
    finally:
        metrics["active_requests"].labels(session_id=session_id).dec() # type: ignore

    return label, confidence

with open(parent / configs["topic"]["data"]["mapping"], "r") as f:
    topic_mapping = json.load(f)

with open(parent / configs["sentiment"]["data"]["mapping"], "r") as f:
    sent_mapping = json.load(f)

if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    init_metrics_server(configs["monitoring"]["port"]["model"])
    process = psutil.Process(os.getpid())

metrics = get_metrics()
metrics["model_memory_usage"].set(process.memory_info().rss) # type: ignore

topic_model_name = configs["topic"]["model"]["name"]
topic_model_path = configs["topic"]["model"]["path"]
topic_uri = init_topic_model()

sent_model_name = configs["sentiment"]["model"]["name"]
sent_model_path = configs["sentiment"]["model"]["path"]
sent_uri = init_sentiment_model()

port = configs["deployment"]["port"]
# st.write(f"Model used for Sentiment Analysis - {sent_model_name}")
# st.write(f"Model used for Topic Modelling - {topic_model_name}")

session_id = uuid.uuid4()
app.config["SESSION_ID"] = session_id
app.config["UPLOAD_DIR"] = parent / "output"
ALLOWED_EXTENSIONS = {"pdf", "txt", "jpeg", "jpg", "png"}
app.config["MAX_FILE_SIZE"] = 100 * 1000 * 1000
app.config["CORS_HEADER"] = "application/json"

print("App running")

@app.route('/', methods=["GET", "POST"]) # type: ignore
def root():
    """
        Homepage for model performance
    """

    if request.method == "POST":
        print(request.files)
        if "file" not in request.files:
            return jsonify({"message": "No file uploaded"}), 400
        
        file = request.files.get("file")
        filename = ""

        if file and file.filename:
            filename = secure_filename(file.filename)
            ext = filename.split('.')[-1]
            if ext not in ALLOWED_EXTENSIONS:
                return jsonify({"message": "File type not allowed"}), 400
            
            try:
                output = analyse(file, ext)
                return output, 200
            except Exception as e:
                logging.error(f"{e}")
                return jsonify({"message": f"Cannot analyse file ; {e}"}), 500
        else:
            return jsonify({"message": "No selected file"}), 400
    else:
        return render_template("index.html",
                               sent_model=sent_model_name,
                               topic_model=topic_model_name)

# @app.route("/health", methods=["GET", "POST"])
# def health():
#     pass

# @app.route("/ready", methods=["GET", "POST"])
# def ready():
#     pass

if __name__ == "__main__":
    print(f"Run app on port = {port}")
    app.run(debug=True, port=port, use_reloader=False)
