ui: mlflow server --backend-store-uri sqlite:///./mlflow.db --default-artifact-root ./models --host 0.0.0.0 --port 5000
topic_serve: mlflow models serve -m "runs:/f7c0ab5f75a84583a757f116fb990ba6/model" -p 5003
sent_serve: mlflow models serve -m "runs:/53e435dd28854b8f96ab215478792166/model" -p 5001 --env-manager local
airflow: ./scripts/run_airflow.sh
prometheus: ./prometheus-3.10.0.linux-amd64/prometheus --config.file=./config/prometheus.yml
alertmanager: ./alertmanager-0.31.1.linux-amd64/alertmanager --config.file=./config/alertmanager.yml
node_exporter: ./node_exporter-1.10.2.linux-amd64/node_exporter
webhook: python3 src/utils/mail.py
app: python3 ./src/app.py