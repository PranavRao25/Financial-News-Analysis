import csv
import pandas as pd
import datasets
import json
import yaml
import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import re

parent = Path(__file__).resolve().parent.parent

def clean(filepath: Path) -> pd.DataFrame:
    """
        Loads the Financial Phrasebank txt file, parses the text/labels, 
        and applies vectorized regex to repair encoding artifacts.
    """

    with open(filepath, "r", encoding="iso-8859-1") as f:  # Finnish encoding
        lines = f.readlines()

    parsed_data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.rsplit('@', 1)
        if len(parts) == 2:
            parsed_data.append({
                "text": parts[0].strip(), 
                "label": parts[1].strip()
            })
    
    df = pd.DataFrame(parsed_data)

    mojibake_map = {
        r"\+ñ": "ä",
        r"\+Ñ": "å",
        r"\+Â": "ö",
        r"\+à": "Å",
        r"\+®": "é",
        r"\+£": "Ü"
    }

    df["text"] = df["text"].replace(mojibake_map, regex=True)
    
    return df

if __name__ == "__main__":
    logging.basicConfig(filename = parent / "prog_log.log",
                        format = '%(levelname)s: %(message)s', 
                        filemode = 'a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.debug("DATA CLEAN")

    with open(parent / "config.yaml", "r") as f:
        configs = yaml.full_load(f)

    root = configs["sentiment"]["data"]
    file_path = Path(root["raw"]) / "data.txt"
    dataset = clean(file_path)
    dataset.to_csv(Path(root["processed"]) / "data.csv", index=False)
    logger.debug("DATA CLEAN PASS")
