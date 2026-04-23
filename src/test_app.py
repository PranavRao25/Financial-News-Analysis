import pytest
import io
import json
from unittest.mock import patch, MagicMock
from werkzeug.datastructures import FileStorage
from prometheus_client import CONTENT_TYPE_LATEST

with patch("redis.Redis") as MockRedis, \
     patch("prometheus_client.start_http_server"):
    import app

@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield client

@pytest.fixture
def mock_redis():
    return app.prediction_cache

@pytest.fixture
def mock_infer():
    with patch("app.infer") as mock:
        def side_effect(txt, mode, model_name, uri, mapping):
            if mode == "sentiment":
                return {"0": {"label": "LABEL_0", "score": 0.95}}
            elif mode == "topic":
                return {"label": "LABEL_1", "score": 0.88}
        mock.side_effect = side_effect
        yield mock

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Financial News Analysis" in response.data # Assuming title is in template

def test_text(client, mock_infer, mock_redis):
    data = {
        "file": (io.BytesIO(b"The stock market experienced a significant surge today."), "market_news.txt")
    }
    response = client.post("/", data=data, content_type="multipart/form-data")
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert "sentiment" in json_data
    assert "topic" in json_data
    assert mock_infer.call_count == 2
    mock_redis.hset.assert_called_once()

@patch("app.pytesseract.image_to_string")
def test_image(mock_ocr, client, mock_infer):
    mock_ocr.return_value = "Extracted text from the news image."
    
    # We use a valid image extension so it passes the ALLOWED_EXTENSIONS check
    data = {
        "file": (io.BytesIO(b"fake_image_bytes"), "scanned_news.jpg")
    }
    
    # We must patch PIL.Image.open because fake_image_bytes isn't a valid image
    with patch("app.Image.open") as mock_image:
        response = client.post("/", data=data, content_type="multipart/form-data")
        
    assert response.status_code == 200
    mock_ocr.assert_called_once()
    assert mock_infer.call_count == 2

def test_empty_text(client):
    data = {
        "file": (io.BytesIO(b"   \n  "), "empty.txt")
    }
    response = client.post("/", data=data, content_type="multipart/form-data")
    
    # app.py raises ValueError("Text empty") which is caught by the broad Exception block
    assert response.status_code == 500
    assert b"Cannot analyse file" in response.data

def test_diff_lang(client, mock_infer):
    data = {
        "file": (io.BytesIO("Le marché boursier est en hausse.".encode('utf-8')), "french_news.txt")
    }
    response = client.post("/", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    
    # Verify the UTF-8 text was correctly passed to the inference function
    args, _ = mock_infer.call_args_list[0]
    assert args[0] == "Le marché boursier est en hausse."

def test_wrong_file_type(client):
    data = {
        "file": (io.BytesIO(b"some binary data"), "diff.exe")
    }
    response = client.post("/", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["message"] == "File type not allowed"

def test_missing_file_payload(client):
    response = client.post("/", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["message"] == "No file uploaded"

@patch("app.requests.post")
def test_infer_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"predictions": [{"label": "positive", "score": 0.99}]}
    mock_post.return_value = mock_response

    result = app.infer("Test string", "sentiment", "sent_model", "http://fake-uri", {})
    
    assert result == {"label": "positive", "score": 0.99}
    mock_post.assert_called_once_with(
        "http://fake-uri", 
        json={"inputs": ["Test string"]}, 
        headers={"Content-Type": "application/json"}
    )

@patch("app.requests.post")
def test_infer_mlflow_error(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response

    with pytest.raises(Exception) as exc_info:
        app.infer("Test string", "sentiment", "sent_model", "http://fake-uri", {})
    
    assert "MLFlow API Error 500" in str(exc_info.value)

def test_ingest_success(client, mock_redis):
    mock_redis.__contains__.return_value = True
    mock_redis.hget.return_value = "positive"
    
    payload = {
        "pred_id": "123e4567-e89b-12d3-a456-426614174000",
        "model_name": "sentiment",
        "true_label": "positive"
    }
    
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    assert response.get_json()["status"] == "correct"

def test_ingest_missing_id(client, mock_redis):
    mock_redis.__contains__.return_value = False
    
    payload = {
        "pred_id": "invalid-id",
        "model_name": "sentiment",
        "true_label": "negative"
    }
    
    response = client.post("/ingest", json=payload)
    assert response.status_code == 404
    assert "not found or expired" in response.get_json()["error"]

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    # Prometheus generate_latest() returns text/plain bytes
    # assert response.content_type == "text/plain; version=0.0.4; charset=utf-8"
    assert CONTENT_TYPE_LATEST in response.content_type

def test_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200

def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    # Ensure our custom metrics are exposed
    assert b"app_total_requests" in response.data
    assert b"app_active_requests_current" in response.data

def test_file_size():
    """
    App doesn't currently strictly enforce MAX_FILE_SIZE natively in code via request.content_length,
    but we leave this test stubbed based on the app.config['MAX_FILE_SIZE'] existence.
    A future implementation should use werkzeug.exceptions.RequestEntityTooLarge.
    """
    pass

def test_text_size():
    """
    App's comment mentions 'Need to ensure the size of the text is fixed'. 
    This should ideally be tested against the model's sequence length limits (e.g., 512 tokens).
    """
    pass