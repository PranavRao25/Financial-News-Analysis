from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, Response
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Info, Summary, generate_latest, CONTENT_TYPE_LATEST
from werkzeug.utils import secure_filename
import os
import requests
import json
from uuid import uuid4
import io
from io import StringIO
import psutil
import csv
import zipfile
import cv2
import pytesseract
from PIL import Image
import yaml
from pathlib import Path
import logging
import redis

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
        ),
        "topic_input_label": Counter(
            "topic_model_input_classes_total",
            "Distribution of classes seen by the Topic model",
            labelnames=["class_name"]
        ),
        "sent_input_label": Counter(
            "sent_model_input_classes_total",
            "Distribution of classes seen by the Sentiment model",
            labelnames=sent_mapping.values()
        ),
        "model_reliability": Counter(
            "model_inference_status_total",
            "Reliability of the models based on uptime",
            labelnames=["model_name", "status"]
        ),
        "topic_baseline_data_dist": Gauge(
            "topic_model_baseline_data_dist",
            "Baseline Topic class distribution from training data",
            labelnames=["classname"]
        ),
        "sent_baseline_data_dist": Gauge(
            "sent_model_baseline_data_dist",
            "Baseline Sentiment class distribution from training data",
            labelnames=["classname"]
        ),
        "feedback_received_total": Counter(
            "model_feedback_received_total",
            "Total ground truth labels received",
            labelnames=["model_name"]
        ),
        "prediction_correctness": Counter(
            "model_prediction_correctness_total",
            "Tracks true positives vs false positives for accuracy decay",
            labelnames=["model_name", "status"]
        ),
        "confusion_matrix": Counter(
            "model_confusion_matrix_total",
            "Real-time confusion matrix components",
            labelnames=["model_name", "true_class", "predicted_class"]
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
    prod_metrics["sent_input_label"].labels(class_name=sent_label).inc() # type: ignore
    topic_label, topic_conf = infer(txt, "topic", topic_model_name, topic_uri, topic_mapping)
    prod_metrics["topic_input_label"].labels(class_name=topic_label).inc() # type: ignore

    # cache save
    pred_id = str(uuid4())
    prediction_cache.hset(name=pred_id, mapping={"sentiment": sent_label, "topic": topic_label})

    return {
        "sentiment": {"label": sent_label, "confidence": sent_conf, "model": sent_model_name},
        "topic": {"label": topic_label, "confidence": topic_conf, "model": topic_model_name},
        "text": txt
    }

def infer(txt, mode, model_name, uri, mapping):
    prod_metrics["requests"].labels(mode=mode).inc() # type: ignore

    label = "Unknown"
    confidence = 0.0

    try:
        with prod_metrics["inference_latency"].labels(mode=mode, model_type=model_name).time(): # type: ignore
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
        # st.write(prod_metrics)
        # st.write(mode)
        logging.error(f"Error during {mode} inference: {e}")
        prod_metrics["errors"].labels(mode=mode, error_types=error_name).inc() # type: ignore
    finally:
        prod_metrics["active_requests"].labels(session_id=session_id).dec() # type: ignore

    return label, confidence

with open(parent / configs["topic"]["data"]["mapping"], "r") as f:
    topic_mapping = json.load(f)

with open(parent / configs["sentiment"]["data"]["mapping"], "r") as f:
    sent_mapping = json.load(f)

if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    init_metrics_server(configs["monitoring"]["port"]["model"])
    process = psutil.Process(os.getpid())

prediction_cache = redis.Redis(host="localhost", port=configs["monitoring"]["port"]["redis"], db=0, decode_responses=True)

prod_metrics = get_metrics()
prod_metrics["model_memory_usage"].set(process.memory_info().rss) # type: ignore

for label in topic_mapping.values():
    prod_metrics["topic_input_label"].labels(class_name=str(label)).inc(0) # type: ignore

for label in sent_mapping.values():
    prod_metrics["sent_input_label"].labels(class_name=str(label)).inc(0) # type: ignore

with open(parent / configs["topic"]["data"]["dist"], "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 2:
            class_name, dist_val = row[0], float(row[1])
            prod_metrics["topic_baseline_data_dist"].labels(class_name=class_name).set(dist_val) # type: ignore

with open(parent / configs["sentiment"]["data"]["dist"], "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 2:
            class_name, dist_val = row[0], float(row[1])
            prod_metrics["sent_baseline_data_dist"].labels(class_name=class_name).set(dist_val) # type: ignore

topic_model_name = configs["topic"]["model"]["name"]
topic_model_path = configs["topic"]["model"]["path"]
topic_uri = init_topic_model()

sent_model_name = configs["sentiment"]["model"]["name"]
sent_model_path = configs["sentiment"]["model"]["path"]
sent_uri = init_sentiment_model()

port = configs["deployment"]["port"]
# st.write(f"Model used for Sentiment Analysis - {sent_model_name}")
# st.write(f"Model used for Topic Modelling - {topic_model_name}")

session_id = uuid4()
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

@app.route("/metrics", methods=["GET", "POST"])
def display_metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/ingest", methods=["GET", "POST"])
def ingest():
    data = request.json

    if data is None:
        return jsonify({"error": "No data found"}), 404
    
    pred_id = data.get("pred_id", "")
    model_name = data.get("model_name", "")
    true_label = data.get("true_label", "")

    if pred_id not in prediction_cache:
        return jsonify({"error": "Prediction ID not found or expired"}), 404
    
    prediction = prediction_cache.hget(pred_id, model_name)

    if not prediction:
        return jsonify({"error": f"No prediction found for the model {model_name}"}), 404
    
    status = "correct" if (true_label == prediction) else "wrong"

    prod_metrics["feedback_received_total"].labels(model_name=model_name).inc() # type: ignore
    prod_metrics["prediction_correctness"].labels(model_name=model_name, status=status).inc() # type: ignore
    prod_metrics["confusion_matrix"].labels(model_name=model_name, true_class=true_label, predicted_class=prediction)
    
    return jsonify({"message": "Ground Truth logged", "status": status}), 200

if __name__ == "__main__":
    print(f"Run app on port = {port}")
    app.run(debug=True, port=port, use_reloader=False)
