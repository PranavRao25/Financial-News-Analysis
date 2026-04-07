import csv
import pandas as pd
import json
import yaml
import logging
from pathlib import Path
import evaluate
import numpy as np
import torch
import wandb
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)

parent = Path(__file__).resolve().parent.parent.parent

def train(train_dataset_path, valid_dataset_path, model_path,
          model_id, no_classes, hyperparams, device):
    
    run = wandb.init(
        project="financial-news-sentiment",
        name=model_id,
        tags=["baseline", model_id],
        config={
            "architecture": model_id,
            **hyperparams
        }
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # argmax over the last dimension to get predicted class indices
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    metric = evaluate.load("f1")  # TODO: CHANGE THE METRIC
    
    # PREPARE THE DATASET
    train_dataset = load_from_disk(str(train_dataset_path))
    valid_dataset = load_from_disk(str(valid_dataset_path))

    # PREPARE THE MODEL
    lr = float(hyperparams.get("lr", 2e-5))
    train_batch_size = hyperparams.get("train_batch_size", 32)
    eval_batch_size = hyperparams.get("eval_batch_size", 64)
    epochs = hyperparams.get("epochs", 5)
    weight_decay = float(hyperparams.get("weight_decay", 0.01))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=no_classes
    ).to(device)

    wandb.watch(model, log="all", log_freq=100)

    training_args = TrainingArguments(
        output_dir=model_path,
        learning_rate=lr,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=model_path,
        logging_steps=50,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),   # Mixed precision training for speed
        report_to="wandb",
        run_name=model_id
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
    trainer.save_model(model_path)

    run.alert(
        title=f"Training Run Complete {model_id}",
        text = f"Model {model_id} Trained. ",
        level="INFO",
        wait_duration=0
    )

    param_counts = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"trainable_parameters": param_counts})

    wandb.finish()

if __name__ == "__main__":
    logging.basicConfig(
        filename=parent/"prog_log.log",
        format='%(levelname)s: %(message)s',
        filemode='a'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.debug("MODEL TRAIN START")

    with open(parent / "config.yaml", "r") as f:
        configs = yaml.full_load(f)

    root = configs["topic"]["data"]
    no_classes = root["no_classes"]
    train_dataset_path = Path(root["processed"]) / "train"
    valid_dataset_path = Path(root["processed"]) / "valid"

    model_details = configs["topic"]["model"]
    model_path = model_details["path"]
    # model_id = model_details["name"]
    # hyperparams = model_details["hyperparams"]
    device = model_details["device"]

    with open("models.json", "r") as f:
        models = json.load(f)
    
    for model in models:
        model_id, hyperparams = model["name"], model["hyperparams"]
        train(train_dataset_path, valid_dataset_path, model_path,
              model_id, no_classes, hyperparams, device)
    logger.debug("MODEL TRAIN PASS")
