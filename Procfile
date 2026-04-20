run: mlflow run . --experiment-name "Topic_Model_Comparisons"
ui: mlflow server --backend-store-uri sqlite:///./mlflow.db --default-artifact-root ./models --host 0.0.0.0 --port 5000
topic_serve: mlflow models serve -m "runs:/f0973911e13c4908b9453d6efd99c1d3/model" -p 5001
airflow: ./scripts/run_airflow.sh
prometheus: ./prometheus-3.10.0.linux-amd64/prometheus --config.file=./config/prometheus.yml
alertmanager: ./alertmanager-0.31.1.linux-amd64/alertmanager --config.file=./config/alertmanager.yml
node_exporter: ./node_exporter-1.10.2.linux-amd64/node_exporter
webhook: python3 src/mail.py
streamlit: PYTHONPATH="." streamlit run src/app.py