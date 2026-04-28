# Test Plan
REST-API testing using Pytest

## app.py

Testfile - `src/test_app.py`

| Test | Function | Desc | Status |
|------|----------|------|--------|
| test_root | / | Test GET | <span style="color:green">PASS</span> |
| test_text | / | Test POST | <span style="color:green">PASS</span> |
| test_image | / | Test POST Image | <span style="color:green">PASS</span> |
| test_empty_text | / | Test POST Empty file | <span style="color:green">PASS</span> |
| test_diff_lang | / | Test POST Diff language text | <span style="color:green">PASS</span> |
| test_wrong_file_type | / | Test POST Wrong file type (apart from .txt/.jpg/.png) | <span style="color:green">PASS</span> |
| test_missing_file_payload | / | Test POST No file | <span style="color:green">PASS</span> |
| test_infer_success | /invocations | Test Model Serving endpoint Success | <span style="color:green">PASS</span> |
| test_infer_mlflow_error | / | Test Model serving endpoint error | <span style="color:green">PASS</span> |
| test_ingest_success | /ingest | Test POST | <span style="color:green">PASS</span> |
| test_ingest_missing_id | /ingest | Test POST Wrong Pred ID | <span style="color:green">PASS</span> |
| test_health | /health | Test GET | <span style="color:green">PASS</span> |
| test_ready | /ready | Test GET | <span style="color:green">PASS</span> |
| test_metrics | /metrics | Test GET | <span style="color:green">PASS</span> |

## mail.py

Testfile = `src/test_mail.py`

| Test | Function | Desc | Status |
|------|----------|------|--------|
| test_send_success | TestSendMail | Test POST Success | <span style="color:green">PASS</span> |
| test_authentication_error | TestSendMail | Test Auth Error | <span style="color:green">PASS</span> |
| test_exception | TestSendMail | Test POST Error | <span style="color:green">PASS</span> |
| test_webhook_valid_payload | TestWebhookRoute | Test POST Success | <span style="color:green">PASS</span> |
| test_webhook_invalid_json | TestWebhookRoute | Test POST Invalid JSON | <span style="color:green">PASS</span> |
| test_webhook_missing_alerts_key | TestWebhookRoute | Test POST No Alert | <span style="color:green">PASS</span> |
| test_webhook_multiple_alerts | TestWebhookRoute | Test POST Multiple Alert Success | <span style="color:green">PASS</span> |