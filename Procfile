ui: mlflow server --backend-store-uri sqlite:///./mlflow.db --default-artifact-root ./models --host 0.0.0.0 --port 5000
serve: mlflow models serve -m "runs:/<run_name>/model" -p 5001 --env-manager local
prometheus: ./prometheus-3.10.0.linux-amd64/prometheus --config.file=prometheus-3.10.0.linux-amd64/prometheus.yml
alertmanager: ./alertmanager-0.31.1.linux-amd64/alertmanager --config.file=alertmanager-0.31.1.linux-amd64/alertmanager.yml
node_exporter: ./node_exporter-1.10.2.linux-amd64/node_exporter
webhook: python3 src/mail.py
streamlit: PYTHONPATH="." streamlit run src/app.py