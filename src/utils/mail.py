from flask import Flask, request, jsonify
import yaml
from pathlib import Path
import logging
import json
import mailtrap as mt
import os
from dotenv import load_dotenv

app = Flask(__name__)
parent = Path(__file__).resolve().parent.parent

load_dotenv(parent / ".env")

with open(parent / "config.yaml", "r") as f:
    raw_path = f.read()
expanded_yaml = os.path.expandvars(raw_path)
configs = yaml.full_load(expanded_yaml)

port = configs["monitoring"]["port"]["alert"]

SENDER_EMAIL = configs["mail_alerts"]["sender_mail"]
RECEIVER_EMAIL = configs["mail_alerts"]["receiver_mail"]
MAIL_API = configs["mail_alerts"]["api"]

def send_mail(subject, body):
    try:
        mail = mt.Mail(
            sender=mt.Address(email=SENDER_EMAIL, name="Alertmanager"),
            to=[mt.Address(email=RECEIVER_EMAIL)],
            subject=subject,
            text=body
        )
        print(mail)
        client = mt.MailtrapClient(token=MAIL_API)
        print(client)
        response = client.send(mail)
        print(response)
        return response
    except Exception as e:
        print(f"Failed with exception : {e}")

@app.route('/', methods = ["POST"])
def webhook():
    if request.method == "POST":
        try:
            payload = request.json
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
