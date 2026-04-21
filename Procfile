run: mlflow run . --experiment-name "Topic_Model_Comparisons"
ui: mlflow server --backend-store-uri sqlite:///./mlflow.db --default-artifact-root ./models --host 0.0.0.0 --port 5000
topic_serve: mlflow models serve -m "runs:/f0973911e13c4908b9453d6efd99c1d3/model" -p 5001
# run: mlflow run . --experiment-name "Sentiment_Analysis_Model_Comparisons" --env-manager=local
# ui: mlflow server --backend-store-uri sqlite:////home/pranav-rao/Documents/iit_madras/MLOps/Financial-News-Analysis/mlflow.db --default-artifact-root ./models --host 0.0.0.0 --port 5000
sent_serve: mlflow models serve -m "runs:/ec9cac28f3b94bf9975a3c74dac67f12/model" -p 5001 --env-manager local
airflow: ./scripts/run_airflow.sh
prometheus: ./prometheus-3.10.0.linux-amd64/prometheus --config.file=./config/prometheus.yml
alertmanager: ./alertmanager-0.31.1.linux-amd64/alertmanager --config.file=./config/alertmanager.yml
node_exporter: ./node_exporter-1.10.2.linux-amd64/node_exporter
webhook: python3 src/utils/mail.py
app: python3 ./src/app.py