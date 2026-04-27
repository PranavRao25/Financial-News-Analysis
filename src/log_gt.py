import pandas as pd
import requests
import concurrent.futures
from io import BytesIO
import pytesseract
from PIL import Image
from pathlib import Path
import time

INFERENCE_URL = "http://localhost:8501/"
INGEST_URL = "http://localhost:8501/ingest"
IMAGE_PARENT_DIR = "inputs/ground_truth_images"

def process_image(row, sent=False, topic=False):
    try:
        image = str(row.get('image', row.get('Image', '')))
        file = Path(IMAGE_PARENT_DIR) / image
        img = Image.open(file).convert("RGB")
        text = str(pytesseract.image_to_string(img))
        
        file_like_object = BytesIO(text.encode("utf-8"))
        files = {"file": ("data.txt", file_like_object, "text/plain")}

        infer_resp = requests.post(INFERENCE_URL, files=files)
        if infer_resp.status_code != 200:
            print(f"Inference failed: {infer_resp.text}")
            return
            
        pred_id = infer_resp.json().get("pred_id")
        if not pred_id:
            return
        
        if topic:
            true_topic = str(row.get('topic', row.get('Topic', row.get('label', ''))))
            requests.post(INGEST_URL, json={
                "pred_id": pred_id,
                "model_name": "topic",
                "true_label": true_topic
            })
        
        if sent:
            true_sentiment = str(row.get('sentiment', row.get('Sentiment', row.get('label', ''))))
            requests.post(INGEST_URL, json={
                "pred_id": pred_id,
                "model_name": "sentiment", # 3. STRICT API MATCH
                "true_label": true_sentiment
            })
        
        print(f"Successfully processed image and ingested pred_id: {pred_id}")
        time.sleep(1) 
        
    except Exception as e:
        print(f"Worker thread crashed on image: {e}")

def process_row(row, sent=False, topic=False):
    try:
        text = str(row.get('text', row.get('Text', '')))
        file_like_object = BytesIO(text.encode('utf-8'))
        files = {"file": ("data.txt", file_like_object, "text/plain")}
        
        infer_resp = requests.post(INFERENCE_URL, files=files)
        if infer_resp.status_code != 200:
            print(f"Inference failed: {infer_resp.text}")
            return
            
        pred_id = infer_resp.json().get("pred_id")
        if not pred_id:
            return
        
        if topic:
            true_topic = str(row.get('topic', row.get('Topic', row.get('label', ''))))
            requests.post(INGEST_URL, json={
                "pred_id": pred_id,
                "model_name": "topic",
                "true_label": true_topic
            })
        
        if sent:
            true_sentiment = str(row.get('sentiment', row.get('Sentiment', row.get('label', ''))))
            requests.post(INGEST_URL, json={
                "pred_id": pred_id,
                "model_name": "sentiment",
                "true_label": true_sentiment
            })
        
        print(f"Successfully processed and ingested pred_id: {pred_id}")
        time.sleep(1)
        
    except Exception as e:
        print(f"Worker thread crashed on row: {e}")

def log_ground_truth(row):
    try:
        process_row(row, sent=True, topic=True)
    except:
        print(f"Worker thread crashed on row: {row}")

def log_sentiment(row):
    try:
        process_row(row, sent=True)
    except:
        print(f"Worker thread crashed on row: {row}")
    
def log_topic(row):
    try:
        process_row(row, topic=True)
    except:
        print(f"Worker thread crashed on row: {row}")

if __name__ == "__main__":
    print("Loading Ground Truth Dataset...")
    df = pd.read_csv("inputs/ground_truth.csv")
    print(f"Spawning ingestion threads for {len(df)} records...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(log_ground_truth, row) for _, row in df.iterrows()]
        concurrent.futures.wait(futures)
   
    print(f"Ingestion Matrix complete in {time.time() - start_time:.2f} seconds.")

    # print("Loading Sentiment Dataset...")
    # df = pd.read_csv("inputs/sent_test.csv")

    # print(f"Spawning ingestion threads for {len(df)} records...")
    # start_time = time.time()
    
    # print(df)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #     futures = [executor.submit(log_sentiment, row) for _, row in df.iterrows()]
    #     concurrent.futures.wait(futures)
    
    # print(f"Ingestion Matrix complete in {time.time() - start_time:.2f} seconds.")
    
    # print("Loading Topic Dataset...")
    # df = pd.read_csv("inputs/topic_test.csv")

    # print(f"Spawning ingestion threads for {len(df)} records...")
    # start_time = time.time()
    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #     futures = [executor.submit(log_topic, row) for _, row in df.iterrows()]
    #     concurrent.futures.wait(futures)
        
    # print(f"Ingestion Matrix complete in {time.time() - start_time:.2f} seconds.")

    # print("Loading Image Dataset...")
    # df = pd.read_csv("inputs/image_test.csv")

    # print(f"Spawning ingestion threads for {len(df)} records...")
    # start_time = time.time()
    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #     futures = [executor.submit(process_image, row) for _, row in df.iterrows()]
    #     concurrent.futures.wait(futures)
        
    # print(f"Ingestion Matrix complete in {time.time() - start_time:.2f} seconds.")
