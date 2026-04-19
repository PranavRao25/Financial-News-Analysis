from datetime import datetime, timedelta
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.utils.email import send_email
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re
import yaml
from pathlib import Path

parent = Path(__file__).resolve().parent.parent

with open(parent / "config.yaml", "r") as f:
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
    dag_id="web_scraper_pipeline",
    default_args=default_args,
    description="Automated Web-to-DB Pipeline",
    schedule=timedelta(minutes=2),
    start_date=datetime(2026, 4, 18),
    catchup=False) as dag:

    @task
    def extract_urls_from_csv(folder_path: str) -> list[str]:
        """
            Read the csv files and extract and clean the urls
        """

        urls = []
        files = Path(folder_path).glob("*.csv")
        url_pattern = r'(https?://[^\s,"]+|www\.[^\s,"]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        
        for file in files:
            df = pd.read_csv(file)
            values = df.values.flatten()  # get the single column of urls
            for val in values:
                if pd.notna(val):
                    matches = re.findall(url_pattern, str(val))  # regex match
                    urls.extend(matches)
        for url in urls:
            if not url.startswith(("http://", "https://")):  # cleaning
                url = "https://" + url
        urls = list(set(urls))  # unique urls
        return urls
    
    @task(pool="scraper_pool", retries=3, retry_exponential_backoff=True)
    def scrape_target_urls(url: str):
        """
            Visit the urls to collect images
        """

        output = {}
        receipent_mail = os.getenv("SMTP_USER", "pranavrao2500@gmail.com")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            
            if response.status_code != 200:  # url visit failed
                subject = f"Scraping : Link Failure Alert {response.status_code}"
                html_content = f"<p>The following URL scraping failed:</p><p><b>{url}</b></p>"
                send_email(to=receipent_mail, subject=subject, html_content=html_content)
                return output
            
            soup = BeautifulSoup(response.text, "html.parser")
            images = [img.get("src") for img in soup.find_all("img") if img.get("src")]
            output = {"url": url, "links": images}
            return output
        except Exception as e:
            print(f"Exception {e}")
            subject = f"Scraping : Exception {e}"
            html_content = f"<p>The following URL scraping failed:</p><p><b>{url}</b></p>"
            send_email(to=receipent_mail, subject=subject, html_content=html_content)
            raise
    
    @task
    def create_database():
        """
            Create Postgres Database
        """

        create_query = """
            CREATE TABLE IF NOT EXISTS images (
            url VARCHAR UNIQUE,
            image TEXT
            )
        """

        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        with pg_hook.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_query)
            conn.commit()

    @task
    def persist_data(scraped_batch: list[dict]):
        """
            Add new urls into database
        """

        pg_hook = PostgresHook(postgres_conn_id="postgres_default")

        insert_query = """
            INSERT INTO images (url, image)
            VALUES (%s, %s)
            ON CONFLICT (url) DO NOTHING
        """

        new_insert_counts = 0
        total_records = len(scraped_batch)

        with pg_hook.get_conn() as conn:
            with conn.cursor() as cursor:
                for record in scraped_batch:
                    if not record:
                        continue

                    cursor.execute(
                        insert_query,
                        (record["url"], str(record["links"]))
                    )

                    new_insert_counts += cursor.rowcount
            conn.commit()
        
        select_query = """
        SELECT COUNT(image) FROM images GROUP BY url
        """

        with pg_hook.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query)
                dist = cursor.fetchall()
            conn.commit()
        
        threshold = 5  # send mail on every 5 new additions
        receipent_mail = os.getenv("SMTP_USER", "pranavrao2500@gmail.com")
        if new_insert_counts >= threshold:
            subject = f"{new_insert_counts} New Pages Processed"
            html_content = f"""
            <h3>Batch Collection Complete</h3>
            <p>Inserted <b>{new_insert_counts}</b> new pages into the database.</p>
            <p>Total URLs attempted in this batch: {total_records}</p>
            <p>Distribution of images: {dist}</p>
            """
            send_email(to=receipent_mail, subject=subject, html_content=html_content)
            print(f"Batch notification sent for {new_insert_counts} records.")

    # FILE SENSOR
    watch_for_target_list = FileSensor(
        task_id="watch_for_image_file",
        fs_conn_id="image_folder_conn",
        filepath="*.png",
        poke_interval=30,
        timeout=60,
        mode="reschedule",
        on_failure_callback=alert_dry_pipeline,
        soft_fail=False
    )
    
    create = create_database()
    url_list = extract_urls_from_csv("/opt/airflow/data/")
    scraped_data = scrape_target_urls.expand(url=url_list)
    done = persist_data(scraped_data)

    # DEFINE THE DEPENDENCIES
    watch_for_target_list >> create >> url_list >> scraped_data >> done
