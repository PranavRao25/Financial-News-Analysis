import csv
import pandas as pd
import datasets
import json
import yaml
import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

parent = Path(__file__).resolve().parent.parent.parent

def split(dataset: Path, split_ratio: float, random_state: int):
    """
        Split into train and valid datasets
    """
    
    df = pd.read_csv(dataset)
    y = df["label"]
    train_df, valid_df = train_test_split(df, test_size=split_ratio,
                                          random_state=random_state, stratify=y)
    train_df, valid_df = pd.DataFrame(train_df), pd.DataFrame(valid_df)

    return train_df, valid_df

if __name__ == "__main__":
    logging.basicConfig(filename = parent / "prog_log.log",
                        format = '%(levelname)s: %(message)s', 
                        filemode = 'a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.debug("DATA PREP")

    with open(parent / "config/config.yaml", "r") as f:
        configs = yaml.full_load(f)

    root = configs["topic"]["data"]
    dataset = Path(root["raw"]) / "train.csv"
    split_ratio = root["train_valid_split"]
    random_state = root["random_state"]
    train_dataset, valid_dataset = split(dataset, split_ratio, random_state)

    train_dataset.to_csv(Path(root["processed"]) / "train.csv", index=False)
    valid_dataset.to_csv(Path(root["processed"]) / "valid.csv", index=False)
    test_dataset = pd.read_csv(Path(root["raw"]) / "test.csv")
    test_dataset.to_csv(Path(root["processed"]) / "test.csv", index=False)

    logger.debug("DATA PREP PASS")
