import csv
import pandas as pd
import json
import yaml
import logging
from pathlib import Path
import evaluate
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)

parent = Path(__file__).resolve().parent.parent

def train(train_dataset_path, valid_dataset_path, model_path,
          model_id, no_classes, hyperparams):
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # argmax over the last dimension to get predicted class indices
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    metric = evaluate.load("accuracy")  # TODO: CHANGE THE METRIC
    
    # PREPARE THE DATASET
    train_dataset = load_from_disk(str(train_dataset_path))
    valid_dataset = load_from_disk(str(valid_dataset_path))

    # PREPARE THE MODEL
    lr = float(hyperparams.get("lr", 1e-5))
    train_batch_size = hyperparams.get("train_batch_size", 32)
    eval_batch_size = hyperparams.get("eval_batch_size", 64)
    epochs = hyperparams.get("epochs", 10)
    weight_decay = float(hyperparams.get("weight_decay", 0.01))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=no_classes
    )

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

    root = configs["topic"]["data"]
    no_classes = root["no_classes"]
    train_dataset_path = Path(root["processed"]) / "train"
    valid_dataset_path = Path(root["processed"]) / "valid"

    model_details = configs["topic"]["model"]
    model_path = model_details["path"]
    model_id = model_details["name"]
    hyperparams = model_details["hyperparams"]
    
    train(train_dataset_path, valid_dataset_path, model_path, model_id, no_classes, hyperparams)
    logger.debug("MODEL TRAIN PASS")
