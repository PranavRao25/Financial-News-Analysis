import streamlit as st
import os
from topic.inference import TopicModel
from sentiment.inference import SentimentModel
import yaml
from pathlib import Path
import json
from uuid import uuid4
import io
import psutil
import zipfile
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Info, Summary
import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

st.title("Financial News Analysis")

st.write("Topic Modelling and Sentiment Analysis")

@st.cache_resource
def init_metrics_server(port=8000):
    start_http_server(port)
    return True

@st.cache_resource
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

@st.cache_resource
def init_topic_model(path):
    topic = TopicModel(path)
    return topic

@st.cache_resource
def init_sentiment_model(path):
    sentiment = SentimentModel(path)
    return sentiment

parent = Path(__file__).resolve().parent.parent

if "configs" not in st.session_state:
    with open(parent / "config.yaml", "r") as f:
        configs = yaml.full_load(f)
    st.session_state["configs"] = configs

if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid4()
session_id = st.session_state["session_id"]

if "topic_mapping" not in st.session_state:
    with open(parent / st.session_state["configs"]["topic"]["data"]["mapping"], "r") as f:
        mapping = json.load(f)
    st.session_state["topic_mapping"] = mapping
topic_mapping = st.session_state["topic_mapping"]

if "sent_mapping" not in st.session_state:
    with open(parent / st.session_state["configs"]["sentiment"]["data"]["mapping"], "r") as f:
        mapping = json.load(f)
    st.session_state["sent_mapping"] = mapping
sent_mapping = st.session_state["sent_mapping"]

init_metrics_server(st.session_state["configs"]["monitoring"]["port"]["model"])
metrics = get_metrics()

process = psutil.Process(os.getpid())
metrics["model_memory_usage"].set(process.memory_info().rss) # type: ignore

topic_model_name = st.session_state["configs"]["topic"]["model"]["name"]
topic_model_path = st.session_state["configs"]["topic"]["model"]["path"]
topic = init_topic_model(parent / topic_model_path)

sent_model_name = st.session_state["configs"]["sentiment"]["model"]["name"]
sent_model_path = st.session_state["configs"]["sentiment"]["model"]["path"]
sentiment = init_sentiment_model(parent / sent_model_path)

st.write(f"Model used for Sentiment Analysis - {sent_model_name}")
st.write(f"Model used for Topic Modelling - {topic_model_name}")

# TEMP : TEXT BOX TO TAKE INPUT
# LATER INCLUDE TESSERACT FOR OCR

def infer(txt, mode, model_name, model, mapping):
    metrics["requests"].labels(mode=mode).inc() # type: ignore
    try:
        with metrics["inference_latency"].labels(mode=mode, model_type=model_name).time(): # type: ignore
            results = model.predict([txt])
            for item in results:
                label = str(mapping[str(item["predicted_class"])])
                # st.write(item["text"])
                st.write(f"{str(mode).capitalize()} - {label} | \n Confidence - {item["confidence"]}")
    except Exception as e:
        error_name = type(e).__name__
        st.error(f"Error in processing : {error_name}")
        st.write(metrics)
        st.write(mode)
        metrics["errors"].labels(mode=mode, error_types=error_name).inc() # type: ignore
    finally:
        metrics["active_requests"].labels(session_id=session_id).dec() # type: ignore

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg", "text", "pdf", "zip"])
# txt = st.text_area("Text to analyse", "")

if uploaded_file is not None:
    file_size = uploaded_file.size
    metrics["active_requests"].labels(session_id=session_id).inc() # type: ignore
    
    if str(uploaded_file.type).split("/")[0] == "image":
        img = Image.open(uploaded_file).convert("RGB")
        txt = str(pytesseract.image_to_string(img))  # check for language also
    elif uploaded_file.type == "application/pdf":
        pass
    elif uploaded_file.type == "application/zip":
        pass
    elif uploaded_file.type == "text/plain":
        pass

    # Sentiment Analysis
    mode = "sentiment"
    infer(txt, mode, sent_model_name, sentiment, sent_mapping)

    # Topic Modelling
    mode = "topic"
    infer(txt, mode, topic_model_name, topic, topic_mapping)

# elif st.button("Classify", type="primary"):
#     metrics["active_requests"].labels(session_id=session_id).inc() # type: ignore

#     # Sentiment Analysis
#     mode = "sentiment"
#     infer(txt, mode, sent_model_name, sentiment, sent_mapping)

#     # Topic Modelling
#     mode = "topic"
#     metrics["requests"].labels(mode=mode).inc() # type: ignore
