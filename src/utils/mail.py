from flask import Flask, request, jsonify
import yaml
from pathlib import Path
import logging
import json
import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv
from werkzeug.exceptions import BadRequest

app = Flask(__name__)
parent = Path(__file__).resolve().parent.parent.parent

load_dotenv(parent / ".env")

with open(parent / "config/config.yaml", "r") as f:
    raw_path = f.read()

expanded_yaml = os.path.expandvars(raw_path)
configs = yaml.full_load(expanded_yaml)

port = configs["monitoring"]["port"]["alert"]

SENDER_EMAIL = configs["mail_alerts"]["sender_mail"]
RECEIVER_EMAIL = configs["mail_alerts"]["receiver_mail"]
MAIL_PASSWORD = configs["mail_alerts"]["password"]

def send_mail(subject, body):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = f"Alertmanager <{SENDER_EMAIL}>"
        msg['To'] = RECEIVER_EMAIL

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()  # Identify ourselves to the server
            server.starttls()  # Upgrade the connection to secure TLS
            server.ehlo()  # Re-identify over the secure connection
            server.login(SENDER_EMAIL, MAIL_PASSWORD)
            server.send_message(msg)
        logging.info(f"Successfully sent alert email: {subject}")
        return True
    except smtplib.SMTPAuthenticationError:
        logging.error("SMTP Authentication failed. Verify your App Password and Email.")
        return False
    except Exception as e:
        logging.error(f"Failed to send email with exception: {e}")
        print(f"Failed with exception : {e}")
        return False

@app.route('/', methods = ["POST"])
def webhook():
    if request.method == "POST":
        try:
            payload = request.get_json()
        except BadRequest:
            return "Invalid json", 400
        except json.JSONDecodeError:
            return "Invalid json", 400

        if payload and "alerts" in payload:
            for alert in payload["alerts"]:
                status = alert.get("status", "Unknown")
                labels = alert.get("labels", {})
                annotations = alert.get("annotations", {})

                alertname = labels.get("alertname", "Unnamed Alert")
                severity = labels.get("severity", "unknown")
                summary = annotations.get("summary", "no summary")
                description = annotations.get("description", "no description")

                subject = f"[{status.upper()}] {severity.upper()} Alert : {alertname}"
                body = (
                    f"Alert Status: {status}\n"
                    f"Alert Name: {alertname}\n"
                    f"Summary: {summary}\n"
                    f"Description: {description}\n"
                    f"Payload: {alert}\n"
                )
                _ = send_mail(subject, body)
        
    return jsonify({"status": "success"}), 200
    
if __name__ == "__main__":
    app.run(debug=True, port = port)
