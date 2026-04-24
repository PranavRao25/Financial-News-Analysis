from datetime import datetime, timedelta
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.utils.email import send_email
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.operators.python import PythonOperator
import pandas as pd
from datasets import Dataset
import json
import subprocess
import logging
import requests
from bs4 import BeautifulSoup
import os
import re
import mlflow
from mlflow.tracking import MlflowClient
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
                drift_value = alert.get("annotations", {}).get("description")
                kwargs['ti'].xcom_push(key='retrain_reason', value=drift_value)
                return {"model": "topic_model", "reason": "data_drift", "details": drift_value}
    
    def fetch_new_ground_truth(**kwargs):
        ti = kwargs["ti"]
        alert_context = ti.xcom_pull(task_ids="parse_alert_context")
        model_name = alert_context.get("model", "topic") if alert_context else "topic"

        pg_hook = PostgresHook(postgres_conn_id="news_db")

        select_sql = """
            SELECT id, url, true_label, dataset_split
            FROM ground_truth_queue
            WHERE processed = FALSE AND target_model = %s;
        """
        records = pg_hook.get_records(select_sql, parameters=(model_name,))
        if not records:
            print("No new ground truths")
            return
        
        data = []
        processed_ids = []

        for record_id, url, label, split in records:
            try:
                # Enforce strict timeouts to prevent Airflow worker locking
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                
                soup = BeautifulSoup(resp.content, "html.parser") # type: ignore
                paragraphs = soup.find_all('p')
                raw_text = " ".join([p.get_text() for p in paragraphs])
                clean_text = re.sub(r'\s+', ' ', raw_text).strip()
                char_limit = configs["deployment"]["text_len_limit"] 
                if len(clean_text) > char_limit:
                    clean_text = clean_text[:char_limit]
                    
                if clean_text:
                    data.append({"text": clean_text, "label": label, "split": split})
                    processed_ids.append(record_id)
                    
            except Exception as e:
                print(f"Extraction failed for {url}: {e}")
        
        if not data:
            raise ValueError("Extraction yielded empty datasets. Check source structures.")

        df = pd.DataFrame(data)
        mapping_path = parent / configs[model_name]["data"]["mapping"]
        with open(mapping_path, "r") as f:
            label_map = {v: int(k) for k, v in json.load(f).items()}
            
        df['label'] = df['label'].map(label_map)
        
        train_df = df[df['split'] == 'train']
        valid_df = df[df['split'] == 'valid']
        train_ds = Dataset.from_pandas(train_df, preserve_index=False)
        valid_ds = Dataset.from_pandas(valid_df, preserve_index=False)
        
        root_path = Path(configs[model_name]["data"]["processed"])
        
        train_ds.save_to_disk(str(root_path / "train"))
        valid_ds.save_to_disk(str(root_path / "valid"))
        
        update_sql = """
            UPDATE ground_truth_queue 
            SET processed = TRUE, processed_at = NOW() 
            WHERE id = ANY(%s);
        """
        pg_hook.run(update_sql, parameters=(processed_ids,))
        
        print(f"Successfully processed and serialized {len(processed_ids)} records to disk.")
    
    def train_model(**kwargs):
        ti = kwargs['ti']
        alert_context = ti.xcom_pull(task_ids='parse_alert_context')
        model_name = alert_context.get('model', 'topic') if alert_context else 'topic'
        hyperparams = configs[model_name]["model"]["hyperparams"]
        script_path = parent / "src" / model_name / f"{model_name}_train.py"

        cmd = [
            "mlflow", "run", str(parent),
            "--experiment-name", f"Automated_Retrain_{model_name}",
            "--lr", f"{hyperparams.get('lr', 2e-5)}",
            "--epochs", f"{hyperparams.get('epochs', 5)}",
            "-wgt_decay", f"{hyperparams.get('weight_decay', 0.01)}"
        ]
        
        logging.info(f"Spawning isolated training process: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in process.stdout: # type: ignore
            logging.info(line.strip())
            
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"Training process aborted. Exit code: {process.returncode}")
            
        return f"{model_name} retraining complete."

    def evaluate(**kwargs):
        ti = kwargs['ti']
        alert_context = ti.xcom_pull(task_ids='parse_alert_context')
        task_model = alert_context.get('model', 'topic') if alert_context else 'topic'

        model_id = configs[task_model]["model"]["name"]
        formatted_id = model_id.replace("/", "_")
        registered_model_name = f"{task_model.capitalize()}_Modelling_{formatted_id}"

        db = parent / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db}")
        client = MlflowClient()

        staging_versions = client.search_model_versions(f"name='{registered_model_name}' and stage='Staging'")
        prod_versions = client.search_model_versions(f"name='{registered_model_name}' and stage='Production'")

        if not staging_versions:
            raise ValueError(f"No Staging versions found for {registered_model_name}.")
        
        staging_version = sorted(staging_versions, key=lambda v: int(v.version), reverse=True)[0]
        staging_run = client.get_run(staging_version.run_id) # type: ignore

        staging_loss = staging_run.data.metrics.get("eval_loss", float('inf'))

        if not prod_versions:
            print(f"No Production baseline found. Promoting version {staging_version.version} to Production.")
            client.transition_model_version_stage(
                name=registered_model_name,
                version=staging_version.version,
                stage="Production",
                archive_existing_versions=False
            )
            return f"Promoted first version {staging_version.version} to Production."

        prod_version = prod_versions[0] # Assuming single production model at a time
        prod_run = client.get_run(prod_version.run_id) # type: ignore
        prod_loss = prod_run.data.metrics.get("eval_loss", float('inf'))

        print(f"Baseline (Production v{prod_version.version}) Loss: {prod_loss:.4f}")
        print(f"Candidate (Staging v{staging_version.version}) Loss: {staging_loss:.4f}")

        epsilon = 1e-4 
        if staging_loss < (prod_loss - epsilon):
            print(f"Candidate outperforms Baseline. Promoting version {staging_version.version}.")
            client.transition_model_version_stage(
                name=registered_model_name,
                version=staging_version.version,
                stage="Production",
                archive_existing_versions=True # Automatically moves old prod to 'Archived'
            )
            return "Promotion Successful."
        else:
            print(f"Candidate did not outperform Baseline. Archiving version {staging_version.version}.")
            client.transition_model_version_stage(
                name=registered_model_name,
                version=staging_version.version,
                stage="Archived"
            )
            return "Promotion Rejected. Candidate Archived."

    
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
