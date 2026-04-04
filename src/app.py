from flask import Flask, request, render_template, jsonify
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Info, Summary
import os
from topic.topic_inference import TopicModel
from sentiment.inference import SentimentModel
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

app = Flask(__name__)
parent = Path(__file__).resolve().parent.parent

with open(parent / "configs.yaml", 'r') as f:
    configs = yaml.full_load(f)

# title - Financial News Analysis
# Sentiment Analysis & Topic Modelling

def init_metrics_server(port=8000):
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

def init_topic_model(path):
    topic = TopicModel(path)
    return topic

def init_sentiment_model(path):
    sentiment = SentimentModel(path)
    return sentiment

def infer(txt, mode, model_name, model, mapping):
    metrics["requests"].labels(mode=mode).inc() # type: ignore
    try:
        with metrics["inference_latency"].labels(mode=mode, model_type=model_name).time(): # type: ignore
            results = model.predict([txt])
            for item in results:
                label = str(mapping[str(item["predicted_class"])])
                # st.write(item["text"])
                # st.write(f"{str(mode).capitalize()} - {label} | \n Confidence - {item["confidence"]}")
    except Exception as e:
        error_name = type(e).__name__
        # st.error(f"Error in processing : {error_name}")
        # st.write(metrics)
        # st.write(mode)
        metrics["errors"].labels(mode=mode, error_types=error_name).inc() # type: ignore
    finally:
        metrics["active_requests"].labels(session_id=session_id).dec() # type: ignore

with open(parent / configs["topic"]["data"]["mapping"], "r") as f:
    topic_mapping = json.load(f)

with open(parent / configs["sentiment"]["data"]["mapping"], "r") as f:
    sent_mapping = json.load(f)

init_metrics_server(configs["monitoring"]["port"]["model"])
metrics = get_metrics()

process = psutil.Process(os.getpid())
metrics["model_memory_usage"].set(process.memory_info().rss) # type: ignore

topic_model_name = configs["topic"]["model"]["name"]
topic_model_path = configs["topic"]["model"]["path"]
topic = init_topic_model(parent / topic_model_path)

sent_model_name = configs["sentiment"]["model"]["name"]
sent_model_path = configs["sentiment"]["model"]["path"]
sentiment = init_sentiment_model(parent / sent_model_path)

port = configs["deployment"]["port"]
# st.write(f"Model used for Sentiment Analysis - {sent_model_name}")
# st.write(f"Model used for Topic Modelling - {topic_model_name}")

# TODO: Add a file upload button

@app.route('/', method=['GET', 'POST'])
def root():
    """
        Homepage for model performance
    """

    if request.method == "POST":
        try:
            data = request.get_json() if request.is_json else request.form
        except:
            return "Invalid input: expected data", 400
        
        value = []

        if request.is_json or request.headers.get("Accept") == "application/json":
            return jsonify(value)
        
@app.route("/health", method=["GET", "POST"])
def health():
    pass

@app.route("/ready", method=["GET", "POST"])
def ready():
    pass

if __name__ == "__main__":
    app.run(debug=True, port=port)