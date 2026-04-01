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

txt = st.text_area("Text to analyse", "")
if st.button("Classify", type="primary"):
    metrics["active_requests"].labels(session_id=session_id).inc() # type: ignore
    
    # Sentiment Analysis
    mode = "sentiment"
    metrics["requests"].labels(mode=mode).inc() # type: ignore
    try:
        with metrics["inference_latency"].labels(mode=mode, model_type=sent_model_name).time(): # type: ignore
            results = sentiment.predict([txt])
            for item in results:
                label = str(sent_mapping[str(item["predicted_class"])])
                st.write(item["text"])
                st.write(f"Sentiment - {label} \n Confidence - {item["confidence"]}")
    except Exception as e:
        error_name = type(e).__name__
        metrics["error"].labels(mode=mode, error_type=error_name).inc() # type: ignore
        st.error(f"Error in processing : {error_name}")
    finally:
        metrics["active_requests"].labels(session_id=session_id).dec() # type: ignore

    # Topic Modelling
    mode = "topic"
    metrics["requests"].labels(mode=mode).inc() # type: ignore
    try:
        with metrics["inference_latency"].labels(mode=mode, model_type=topic_model_name).time(): # type: ignore
            results = topic.predict([txt])
            for item in results:
                label = str(topic_mapping[str(item["predicted_class"])])
                st.write(item["text"])
                st.write(f"Topic - {label} \n Confidence - {item["confidence"]}")
    except Exception as e:
        error_name = type(e).__name__
        metrics["error"].labels(mode=mode, error_type=error_name).inc() # type: ignore
        st.error(f"Error in processing : {error_name}")
    finally:
        metrics["active_requests"].labels(session_id=session_id).dec() # type: ignore
