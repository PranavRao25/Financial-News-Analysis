from datetime import datetime, timedelta
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.utils.email import send_email
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.operators.python import PythonOperator
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re
import yaml
from pathlib import Path

parent = Path(__file__).resolve().parent.parent

with open(parent / "config/config.yaml", "r") as f:
    configs = yaml.full_load(f)
default_args = configs["default_args"]
default_args["retry_delay"] = timedelta(configs["default_args"]["retry_delay"])

def alert_dry_pipeline(context):
    """
        Send an alert email of no data
    """

    task_instance = context.get("task_instance")
    execution_date = context.get("execution_date")
    dag_id = task_instance.dag_id
    task_id = task_instance.task_id

    subject = f"AIRFLOW ALERT: Dry Pipeline {dag_id}"

    html_content = f"""
        <h3>Dry Pipeline Alert</h3>
        <p>The sensor task <b>{task_id}</b> timed out waiting for a new target list.</p>
        <ul>
            <li><b>DAG:</b> {dag_id}</li>
            <li><b>Execution Date:</b> {execution_date}</li>
            <li><b>Log URL:</b> <a href="{task_instance.log_url}">View Airflow Logs</a></li>
        </ul>
        <p>Please verify the upstream systems responsible for delivering the CSV.</p>
    """

    receipent_mail = os.getenv("EMAIL", "pranavrao2500@gmail.com")

    try:
        send_email(to=receipent_mail, subject=subject, html_content=html_content)
        print("Dry pipeline mail sent")
    except Exception as e:
        print(f"Failed to send mail {e}")

with DAG(
    dag_id="news_analysis_pipeline",
    default_args=default_args,
    description="Automated News Analysis Pipeline",
    schedule=None,
    schedule_interval=None,
    start_date=datetime(2026, 4, 24),
    tags=["retrain"],
    catchup=False) as dag:

    def extract_alert_context(**kwargs):
        dag_run = kwargs["dag_run"]
        model_name = kwargs["model_name"]
        payload = dag_run.conf

        if not payload or "alerts" not in payload:
            return {"model": "model_name", "method": "manual"}
        
        alerts = payload.get("alerts")
        for alert in alerts:
            if alert["status"] == "firing":
                return {"model": model_name, "drift": alert.get("annotations", {}).get("description")}
    
    # @task(pool="scraper_pool", retries=3, retry_exponential_backoff=True)
    
    def fetch_new_ground_truth(**kwargs):
        pass
    
    def train_model(**kwargs):
        pass

    def evaluate(**kwargs):
        pass
    
    parse_alerts = PythonOperator(
        task_id="parse_alert_context",
        python_callable=extract_alert_context,
        provide_context=True
    )
    
    fetch_data = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_new_ground_truth
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    eval = PythonOperator(
        task_id="evaluate",
        python_callable=evaluate
    )

    parse_alerts >> fetch_data >> train >> eval
