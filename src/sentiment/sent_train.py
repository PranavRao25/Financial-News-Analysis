import csv
import pandas as pd
import json
import yaml
import logging
from pathlib import Path
import evaluate
import numpy as np
import datetime
import os
import torch
import mlflow.transformers as mpt
import importlib.util
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

mpt.autolog(log_model_signatures=True)
mlflow.enable_system_metrics_logging()
assert torch.cuda.is_available(), "CUDA is not available. Check your PyTorch installation!"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)

path = "src/utils/mail.py"
mname = "mail"
spec = importlib.util.spec_from_file_location(mname, path)
assert spec is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod) # type: ignore

parent = Path(__file__).resolve().parent.parent.parent
db = parent  / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db}")

def successful_mail(run_name, exp_name, model_id, results):
    subject = f"MODEL_ID : {model_id} successfully trained"
    body = f"""
        Experiment {exp_name} Run {run_name} complete\n
        Validation results:\n
        {results}
    """
    mod.send_mail(subject, body)  # TODO: Not working

def failure_mail(run_name, exp_name, model_id, e):
    subject = f"MODEL_ID : {model_id} training failed"
    body = f"""
        Experiment {exp_name} Run {run_name} failed\n
        Exception: {e}
    """
    mod.send_mail(subject, body)

def train(train_dataset_path, valid_dataset_path, model_path, model_id, no_classes, exp, hyperparams):
    print(f"Model {model_id} training start")
    run = mlflow.start_run(experiment_id=exp.experiment_id, run_name=model_id, 
                           description=f"Test the performance of {model_id} for topic modelling")
    mlflow.set_tag("Model_id", model_id)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels,
                               average="macro")
        
        return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]} # type: ignore

    accuracy_metric = evaluate.load("accuracy")  # TODO: CHANGE THE METRIC
    f1_metric = evaluate.load("f1")
    
    # PREPARE THE DATASET
    train_dataset = load_from_disk(str(train_dataset_path))
    valid_dataset = load_from_disk(str(valid_dataset_path))

    # PREPARE THE MODEL
    lr = float(hyperparams.get("lr", 1e-5))
    train_batch_size = hyperparams.get("train_batch_size", 32)
    eval_batch_size = hyperparams.get("eval_batch_size", 64)
    epochs = hyperparams.get("epochs", 10)
    weight_decay = float(hyperparams.get("weight_decay", 0.01))
    gradient_accumulation_steps = int(hyperparams.get("gradient_accumulation_steps", 8))

    print(f"No of Epochs: {epochs}")
    mlflow.log_params(hyperparams)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, 
            num_labels=no_classes
        )

        training_args = TrainingArguments(
            output_dir=model_path,
            learning_rate=lr,
            optim="adamw_torch",
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=model_path,
            gradient_checkpointing=True,
            logging_steps=50,
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),   # Mixed precision training for speed
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset, # type: ignore
            eval_dataset=valid_dataset, # type: ignore
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics # type: ignore
        )

        trainer.train()

        results = trainer.evaluate()
        mlflow.log_metrics(results)

        trainer.save_model(model_path)

        # param_counts = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # mlflow.log_param("param_count", param_counts)

        model_info = mpt.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            registered_model_name=f"Financial_Sentiment_{model_id}"
        )
        client = MlflowClient()
        client.transition_model_version_stage(
            name=f"Financial_Sentiment_{model_id}",
            version=str(model_info.registered_model_version),
            stage="Staging"
        )
        print("Training Done")
        successful_mail(run.info.run_name, exp.name, model_id, results)
    except Exception as e:
        failure_mail(run.info.run_name, exp.name, model_id, e)
        raise e
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    logging.basicConfig(
        filename=parent / "prog_log.log",
        format='%(levelname)s: %(message)s',
        filemode='a'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.debug("MODEL TRAIN START")

    with open(parent / "config.yaml", "r") as f:
        configs = yaml.full_load(f)

    root = configs["sentiment"]["data"]
    no_classes = root["no_classes"]
    train_dataset_path = Path(root["processed"]) / "train"
    valid_dataset_path = Path(root["processed"]) / "valid"

    model_details = configs["sentiment"]["model"]
    model_path = model_details["path"]

    with open(parent / "src/sentiment/sent_models.json", "r") as f:
        models = json.load(f)

    experiment = mlflow.get_experiment_by_name("Sentiment_Analysis_Model_Comparisons")
    if experiment:
        print("Experiment exists")
        exp_id = experiment.experiment_id
    else:
        print("Creating new Experiment")
        exp_id = mlflow.create_experiment("Sentiment_Analysis_Model_Comparisons")
    try:
        exp = mlflow.set_experiment(experiment_id=exp_id)
        for model in models:
            model_id, hyperparams = model["name"], model["hyperparams"]
            train(train_dataset_path, valid_dataset_path, model_path, model_id, no_classes, exp, hyperparams)
    except Exception as e:
        print(e)
        raise e
    finally:
        logger.debug("MODEL TRAIN PASS")
