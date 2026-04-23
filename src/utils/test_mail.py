import pytest
import os
import json
from pathlib import Path
import yaml
from unittest.mock import patch, MagicMock
import importlib.util
from smtplib import SMTPAuthenticationError

with patch('builtins.open'), patch('yaml.full_load', return_value=
          {"monitoring": {"port": {"alert": 5000}}, 
        "mail_alerts": {
            "sender_mail": os.getenv("EMAIL"), 
            "receiver_mail": os.getenv("EMAIL"),
            "password": os.getenv("PASSWORD")
        }
    }), patch('dotenv.load_dotenv'), patch('os.path.expandvars'):
    import mail

@pytest.fixture
def client():
    mail.app.config["TESTING"] = True
    with mail.app.test_client() as client:
        yield client

@pytest.fixture
def valid_payload():
    return {
        "alerts": [
        {
            "status": "firing",
            "labels": {},
            "annotations": {},
            "alertname": "Test Alert",
            "severity": "Null",
            "summary": "None",
            "description": "None"
        }
    ]}

class TestSendMail:

    @patch("mail.smtplib.SMTP")
    def test_send_success(self, mock_smtp):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        subject = "Test Success"
        body = "Test Body"

        result = mail.send_mail(subject, body)

        assert result is True
        mock_smtp.assert_called_once_with("smtp.gmail.com", 587)
        mock_server.ehlo.assert_called()
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with(os.getenv("EMAIL"), os.getenv("PASSWORD"))
        mock_server.send_message.assert_called_once()
    
    @patch("mail.smtplib.SMTP")
    def test_authentication_error(self, mock_smtp):
        mock_server = MagicMock()

        mock_smtp.return_value.__enter__.return_value = mock_server
        mock_server.login.side_effect = SMTPAuthenticationError(535, b"Authentication Failed")

        result = mail.send_mail("Test Failure", "Test Body")

        assert result is False

    @patch("mail.smtplib.SMTP")
    def test_exception(self, mock_smtp):
        mock_smtp.side_effect = Exception("Error")

        result = mail.send_mail("Test Exception", "Test Body")

        assert result is False

class TestWebhookRoute:
    @patch('mail.send_mail')
    def test_webhook_valid_payload(self, mock_send_mail, client, valid_payload):
        mock_send_mail.return_value = True
        
        response = client.post('/', json=valid_payload)
        
        assert response.status_code == 200
        assert response.json == {"status": "success"}
        
        mock_send_mail.assert_called_once()
        args, _ = mock_send_mail.call_args
        subject = args[0]
        body = args[1]
        
        assert "[FIRING] UNKNOWN Alert : Unnamed Alert" in subject

    def test_webhook_invalid_json(self, client):
        response = client.post('/', data="Not a JSON payload", content_type='application/json')
        
        assert response.status_code == 400
        assert b"Invalid json" in response.data

    @patch('mail.send_mail')
    def test_webhook_missing_alerts_key(self, mock_send_mail, client):
        payload = {"status": "resolved"} # No 'alerts' key
        
        response = client.post('/', json=payload)
        
        assert response.status_code == 200
        mock_send_mail.assert_not_called()

    @patch('mail.send_mail')
    def test_webhook_multiple_alerts(self, mock_send_mail, client, valid_payload):
        valid_payload["alerts"].append(valid_payload["alerts"][0])
        
        response = client.post('/', json=valid_payload)
        
        assert response.status_code == 200
        assert mock_send_mail.call_count == 2
