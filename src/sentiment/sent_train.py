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

parent = Path(__file__).resolve().parent.parent.parent

def train(train_dataset_path, valid_dataset_path, model_path,
          model_id, no_classes, hyperparams):
    
    run = wandb.init(
        project="financial_news_sentiment",
        name=model_id,
        tags=["baseline", model_id],
        config={
            "architecture": model_id,
            **hyperparams
        }
    )

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

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=no_classes
    )
    wandb.watch(model, log="all", log_freq=100)

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
        text = f"{model_id} training done",
        level="INFO",
        wait_duration=0
    )

    param_counts = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"trainable_parameters": param_counts})

    wandb.finish()

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
    # model_id = model_details["name"]
    # hyperparams = model_details["hyperparams"]

    with open("models.json", "r") as f:
        models = json.load(f)

    for model in models:
        model_id, hyperparams = model["name"], model["hyperparams"]
        train(train_dataset_path, valid_dataset_path, model_path, model_id, no_classes, hyperparams)
    logger.debug("MODEL TRAIN PASS")
