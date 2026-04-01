import streamlit as st
import os
from topic.inference import TopicModel
import yaml
from pathlib import Path
import json
from uuid import uuid4
import io
import psutil
import zipfile
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Info, Summary

st.title("Financial News Analysis")

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
def init_model(path):
    topic = TopicModel(path)
    return topic

parent = Path(__file__).resolve().parent.parent

if "configs" not in st.session_state:
    with open(parent / "config.yaml", "r") as f:
        configs = yaml.full_load(f)
    st.session_state["configs"] = configs

if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid4()
session_id = st.session_state["session_id"]

if "mapping" not in st.session_state:
    with open(parent / st.session_state["configs"]["topic"]["data"]["mapping"], "r") as f:
        mapping = json.load(f)
    st.session_state["mapping"] = mapping
mapping = st.session_state["mapping"]

init_metrics_server(st.session_state["configs"]["monitoring"]["port"]["model"])
metrics = get_metrics()

process = psutil.Process(os.getpid())
metrics["model_memory_usage"].set(process.memory_info().rss) # type: ignore

model_name = st.session_state["configs"]["topic"]["model"]["name"]
model_path = st.session_state["configs"]["topic"]["model"]["path"]
topic = init_model(parent / model_path)

st.write(f"Model used - {model_name}")

# TEMP : TEXT BOX TO TAKE INPUT
# LATER INCLUDE TESSERACT FOR OCR

txt = st.text_area("Text to analyse", "")
if st.button("Classify", type="primary"):
    mode = "topic"

    metrics["active_requests"].labels(session_id=session_id).inc() # type: ignore
    metrics["requests"].labels(mode=mode).inc() # type: ignore

    try:
        with metrics["inference_latency"].labels(mode=mode, model_type=model_name).time(): # type: ignore
            results = topic.predict([txt])
            for item in results:
                label = str(mapping[str(item["predicted_class"])])
                st.write(item["text"])
                st.write(f"Topic - {label} \t\t Confidence - {item["confidence"]}")
    except Exception as e:
        error_name = type(e).__name__
        metrics["error"].labels(mode=mode, error_type=error_name).inc() # type: ignore
        st.error(f"Error in processing : {error_name}")
    finally:
        metrics["active_requests"].labels(session_id=session_id).dec() # type: ignore
