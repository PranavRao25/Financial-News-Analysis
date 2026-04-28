# API Reference: Financial News Analysis Application

## System Overview

This application uses a Flask-based REST API service for user interaction as well as health monitoring. Its workflow is:
1. It accepts and processes multimodal inputs (images and text). It uses Tesseract OCR for images.
2. It routes the payload to the isolated MLFlow model serving endpoints concurrently.
3. It displays the predictions to a static HTML page.

It uses Redis to persist history, observed inputs and ground-truths and Prometheus for observability.

---

## REST API Endpoints

### Inference Gateway

* Endpoint - `/`
* Methods - `GET` and `POST`

#### GET
* `GET`: Renders the UI with the model predictions via `index.html`

#### POST
* `POST`: Model inference
* Supported File Extensions: `.txt`, `.pdf`, `.jpeg`, `.jpg`, `.png`
* Request Form-Data: `file` (File) - The document or image payload. Max size: 100MB.
* Two IO threads - one for each task
* * Makes a post request at the model serving endpoint
* Hashs the input-output pair using Redis Cache
* Responses -
* * Success Response - ```200 OK```
    ```json
    {
      "sentiment": {
        "label": "",
        "confidence": ,
        "model": "",
        "status": "success",
        "error": ""
      },
      "topic": {
        "label": "",
        "confidence": ,
        "model": "",
        "status": "success",
        "error": ""
      },
      "text": "",
      "pred_id": ""
    }
    ```
* * Error Responses:
* * * ```400 Bad Request``` (Missing/Invalid File)
* * * ```500 Internal Server Error``` (Inference/Extraction fault)

### Ground Truth Ingestion
* Endpoint: `/ingest`
* Methods: `GET`, `POST`

#### GET
* `GET`: 

#### POST
* `POST`: Consumes user-provided ground truths or downstream feedback on previous predictions. It logs the drift metrics to Prometheus and stores the ground truths using Redis cache.
* Request (JSON):
    ```json
    {
      "pred_id": "",
      "model_name": "", 
      "true_label": ""
    }
    ```
* Responses:
* * Success Response - ```200 OK```
    ```json
    {
      "message": "Ground Truth logged",
      "status": "wrong" 
    }
    ```
* * Error Responses - `404 Not Found` (Cache miss, expired TTL, or missing payload).

### Observability Endpoints
* Endpoints: `/health`, `/ready`, `/metrics`
* Methods: `GET`
* Description: Exposes the Prometheus client registry. Returns serialized time-series data for scraping by a Prometheus server

---

## 2. Internal Core Functions

### `analyse(file, ext)`
* Purpose: Main inference function 
* Execution: Extracts data from the file, spins up two worker threads, performs data validation for the model inference and caches the results.
* Return type - 
    ```json
    {
        "sentiment": "",
        "topic": "",
        "text": "",
        "pred_id": ""
    }
    ```

### `infer(txt, mode, model_name, uri, mapping)`
* Purpose: Handles the HTTP layer integration with the downstream MLflow model registry.
* Mechanism: Formats inputs according to the standard MLflow HuggingFace/Transformers signature:
    `{"inputs": [txt], "params": {"truncation": True, "max_length": 512, "return_token_type_ids": False}}`
* Returns:
    ```json
    {
        "label": "",
        "score": ""
    }
    ```


