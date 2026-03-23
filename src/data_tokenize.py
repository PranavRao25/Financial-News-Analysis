from datasets import load_dataset
import yaml
import logging
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

parent = Path(__file__).resolve().parent.parent

def tokenize(dataset_path : Path, model_id : str, length : int):
    def tokenize_function(examples):
        tokenized_inputs =  tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=length
        )
        tokenized_inputs["labels"] = [label2id[label] for label in examples["label"]]
        return tokenized_inputs
    
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset("csv", data_files=str(dataset_path))["train"] # type: ignore
    tokenized_dataset = dataset.map(tokenize_function,
                                             batched=True,
                                             remove_columns=["label", "text"])
    return tokenized_dataset

if __name__ == "__main__":
    logging.basicConfig(
        filename=parent / "prog_log.log",
        format='%(levelname)s: %(message)s',
        filemode='a'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info("DATA TOKENIZE START")

    with open(parent / "config.yaml", "r") as f:
        configs = yaml.full_load(f)

    root = configs["sentiment"]["data"]
    length = root["embed_len"]
    model_details = configs["sentiment"]["model"]
    model_id = model_details["name"]
    
    map_paths = {
        "train.csv": "train",
        "valid.csv": "valid",
        "test.csv": "test"
    }

    for key, value in map_paths.items():
        dataset_path = Path(root["processed"]) / key
        token_path = Path(root["processed"]) / value
        tokenized_dataset = tokenize(dataset_path, model_id, length)
        tokenized_dataset.save_to_disk(token_path)
    
    logger.info("DATA TOKENIZE PASS")
    