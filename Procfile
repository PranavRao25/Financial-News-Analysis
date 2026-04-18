run: mlflow run . --experiment-name "Sentiment_Analysis_Model_Comparisons"
ui: mlflow server --backend-store-uri sqlite:///./mlflow.db --default-artifact-root ./models --host 0.0.0.0 --port 5000
sent_serve: mlflow models serve -m "runs:/c0a1a4fd8b154b49a97a1a55633bea50/model" -p 5001 --env-manager local
prometheus: ./prometheus-3.10.0.linux-amd64/prometheus --config.file=prometheus-3.10.0.linux-amd64/prometheus.yml
alertmanager: ./alertmanager-0.31.1.linux-amd64/alertmanager --config.file=alertmanager-0.31.1.linux-amd64/alertmanager.yml
node_exporter: ./node_exporter-1.10.2.linux-amd64/node_exporter
webhook: python3 src/mail.py
app: python3 ./src/app.py