# Financial-News-Analysis

Author - Pranav Rao (DA25M022)

This application performs analysis of Financial News in two methods:
1. Market Topic Classification - Identifying which part of Financial sector the news pertains (example - Central Banks / Politics / Energy / Company)
2. Market Impact Classification (Sentiment Analysis) - Identifying the broad impact the topic of the news will have on the market (Bearish / Bullish / Neutral)
The application accepts a news segment as a file (txt or image) or as text input.

It uses distillBERT finetuned on the corresponding tasks.

### Ports to Monitor
1. 8501 - Main Application
2. 3000 - Grafana Dashboard
3. 9093 - Alert Managers
4. 8080 - Airflow
5. 5000 - MLFlow Dashboard
6. 5001 / 5003 - MLFlow Serving endpoints
7. 9090 - Prometheus

---

## User Manual

Spin up an instance by running:
```
./run.sh
```

Visit the following endpoints (GUI):
1. Standard inference - ```localhost:8501/```
2. Ingest ground truth - ```localhost:8501/ingest```
3. Retraining DAG - ```localhost:8080/dags```
4. Application Monitoring - ```localhost:3000/```

Curl Request:
1. Standard Inference:
```
curl -X POST -F "file=@/path/to/your/test_document.txt" http://localhost:8501/
```

2. Ingest ground truth -
```
curl -X POST http://localhost:8501/ingest -H "Content-Type: application/json" \
-d '{
    "pred_id": "",
    "model_name": "",
    "true_label": ""
}'
```

## Model Training
| Task | Dataset | Model | Metrics |
|------|---------|-------|---------|
|Sentiment Analysis | takala/financial_phrasebank | distillbert-base | F1 / Accuracy |
| Topic Classification | zeroshot/twitter-financial-news-topic | distillbert-base | F1 / Accuracy |

---

## Documentation for Reference

* [API Reference](docs/api_reference.md) : Description of all REST APIs and functions used (Low Level Document)
* [HLD](docs/architecture.jpeg) : High level Diagram
* [Report](docs/Project_Report.pdf): Comprehensive Project Report
* [Test Plan](docs/test_plan.md) : Test Plan
* [Output Dir](output/) : Output Directory
* [Input Dir](inputs/) : Inference input data

---

Directory Structure:
```
.
├── airflow.sh
├── alertmanager-0.31.1.linux-amd64
│   ├── alertmanager
│   ├── amtool
│   ├── LICENSE
│   └── NOTICE
├── conda.yaml
├── config
│   ├── airflow.cfg
│   ├── alertmanager.yml
│   ├── alerts.yaml
│   ├── config.yaml
│   └── prometheus.yml
├── dags
│   ├── dag.py
├── data
│   ├── sentiment
│   │   ├── dist.csv
│   │   ├── manifest.txt
│   │   ├── mapping.json
│   │   ├── processed
│   │   │   ├── data.csv
│   │   │   ├── test
│   │   │   │   ├── data-00000-of-00001.arrow
│   │   │   │   ├── dataset_info.json
│   │   │   │   └── state.json
│   │   │   ├── train
│   │   │   │   ├── data-00000-of-00001.arrow
│   │   │   │   ├── dataset_info.json
│   │   │   │   └── state.json
│   │   │   ├── train.csv
│   │   │   ├── valid
│   │   │   │   ├── data-00000-of-00001.arrow
│   │   │   │   ├── dataset_info.json
│   │   │   │   └── state.json
│   │   │   └── valid.csv
│   │   ├── production
│   │   ├── raw
│   │   │   ├── Sentences_50Agree.txt
│   │   │   ├── Sentences_66Agree.txt
│   │   │   └── Sentences_75Agree.txt
│   │   └── raw.dvc
│   ├── topic
│   │   ├── dist.csv
│   │   ├── manifest.txt
│   │   ├── mapping.json
│   │   ├── processed
│   │   │   ├── test
│   │   │   │   ├── data-00000-of-00001.arrow
│   │   │   │   ├── dataset_info.json
│   │   │   │   └── state.json
│   │   │   ├── train
│   │   │   │   ├── data-00000-of-00001.arrow
│   │   │   │   ├── dataset_info.json
│   │   │   │   └── state.json
│   │   │   ├── train.csv
│   │   │   ├── valid
│   │   │   │   ├── data-00000-of-00001.arrow
│   │   │   │   ├── dataset_info.json
│   │   │   │   └── state.json
│   │   │   └── valid.csv
│   │   ├── production
│   │   ├── raw
│   │   │   ├── test.csv
│   │   │   └── train.csv
│   │   └── raw.dvc
├── docker-compose.yaml
├── Dockerfile
├── Dockerfile.airflow
├── docs
│   ├── AI Application Evaluation Guideline.pdf
│   ├── api_reference.md
│   ├── architecture.jpeg
├── grafana
│   ├── dashboards
│   │   └── dashboard.json
│   └── provisioning
│       ├── dashboards
│       │   └── dashboard.yaml
│       └── datasources
│           └── prometheus.yaml
├── inputs
│   ├── ground_truth.csv
│   ├── images
│   │   ├── India
│   │   └── US
│   ├── papers
│   │   ├── India
│   │   └── US
│   ├── sent_test.csv
│   └── topic_test.csv
├── LICENSE
├── mlflow.db
├── MLProject
├── models
│   ├── sentiment
│   │   └── v4
│   └── topic
│       └── v4
├── node_exporter-1.10.2.linux-amd64
│   ├── LICENSE
│   ├── node_exporter
│   └── NOTICE
├── output
│   └── v4
│       ├── confusion_matrix.png
│       ├── epoch_metrics_distilbert_distilbert-base-uncased.csv
│       ├── epoch_metrics_huawei-noah_TinyBERT_General_4L_312D.csv
│       ├── sentiment
│       │   ├── comparison_runs.png
│       │   ├── confusion_matrix.png
│       │   ├── epoch_metrics_CardiffNLP_twitter-roberta-base-sentiment.csv
│       │   ├── epoch_metrics_distilbert_distilbert-base-uncased.csv
│       │   ├── epoch_metrics_huawei-noah_TinyBERT_General_4L_312D.csv
│       │   ├── eval_accuracy.png
│       │   ├── eval_brier.png
│       │   ├── eval_ece.png
│       │   ├── eval_macro_f1.png
│       │   ├── eval_macro_precision.png
│       │   ├── eval_macro_recall.png
│       │   ├── eval_mcc.png
│       │   ├── eval_micro_f1.png
│       │   ├── eval_micro_precision.png
│       │   ├── eval_micro_recall.png
│       │   ├── eval_pr_auc_macro.png
│       │   ├── eval_pr_auc_micro.png
│       │   ├── eval_pr_auc_weighted.png
│       │   ├── eval_runtime-eval_macro_f1.png
│       │   └── eval_weighted_f1.png
│       └── topic
│           ├── confusion_matrix.png
│           ├── epoch_metrics_distilbert_distilbert-base-uncased.csv
│           ├── epoch_metrics_huawei-noah_TinyBERT_General_4L_312D.csv
│           ├── eval_accuracy.png
│           ├── eval_brier.png
│           ├── eval_ece.png
│           ├── eval_macro_f1.png
│           ├── eval_macro_precision.png
│           ├── eval_macro_recall.png
│           ├── eval_mcc.png
│           ├── eval_micro_f1.png
│           ├── eval_micro_precision.png
│           ├── eval_micro_recall.png
│           ├── eval_pr_auc_macro.png
│           ├── eval_pr_auc_micro.png
│           ├── eval_pr_auc_weighted.png
│           └── eval_weighted_f1.png
├── Procfile
├── prog_log.log
├── prometheus-3.10.0.linux-amd64
│   ├── LICENSE
│   ├── NOTICE
│   ├── prometheus
│   └── promtool
├── README.md
├── requirements.txt
├── run_airflow.sh
├── run.sh
├── src
│   ├── app.py
│   ├── __init__.py
│   ├── log_gt.py
│   ├── sentiment
│   │   ├── data_prep.py
│   │   ├── dvc.yaml
│   │   ├── __init__.py
│   │   ├── sent_data_clean.py
│   │   ├── sent_data_split.py
│   │   ├── sent_data_tokenize.py
│   │   ├── sent_inference.py
│   │   ├── sent_models.json
│   │   └── sent_train.py
│   ├── templates
│   │   └── index.html
│   ├── test_app.py
│   ├── topic
│   │   ├── dvc.yaml
│   │   ├── __init__.py
│   │   ├── models.json
│   │   ├── topic_data_prep.py
│   │   ├── topic_data_tokenize.py
│   │   ├── topic_inference.py
│   │   └── topic_train.py
│   └── utils
│       ├── __init__.py
│       ├── mail.py
│       ├── metrics.py
│       └── test_mail.py

205 directories, 823 files
```