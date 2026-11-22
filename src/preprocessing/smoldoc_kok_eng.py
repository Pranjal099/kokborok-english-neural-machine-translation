"""Hugging Face dataset parser for aligned Kokborok–English sentence pairs.

Extracts parallel sentence pairs from the smoldoc dataset hosted on Hugging Face:
URL: https://huggingface.co/datasets/sdmy/kokborok/raw/main/smoldoc__en_trp.json
"""

import argparse
import json
import logging
import os
import re
from typing import List, Dict
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HF_DATASET_URL = "https://huggingface.co/datasets/sdmy/kokborok/raw/main/smoldoc__en_trp.json"


def normalize_sentence(text: str) -> str:
    """Normalize sentence text and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def download_and_extract_pairs(url: str = HF_DATASET_URL, output_csv: str = "smoldoc_clean_parallel.csv") -> pd.DataFrame:
    """Download the JSON dataset and extract cleaned aligned sentence pairs."""
    logger.info(f"Downloading dataset from: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    rows: List[Dict[str, str]] = []
    for topic in data:
        srcs = topic.get("srcs", [])
        trgs = topic.get("trgs", [])
        for src, trg in zip(srcs, trgs):
            clean_src = normalize_sentence(src)
            clean_trg = normalize_sentence(trg)
            if clean_src and clean_trg:
                rows.append({"english": clean_src, "kokborok": clean_trg})

    df = pd.DataFrame(rows)
    df.drop_duplicates(inplace=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(df):,} cleaned sentence pairs to {output_csv}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Extract sentence pairs from Hugging Face Kokborok dataset.")
    parser.add_argument("--url", type=str, default=HF_DATASET_URL, help="Hugging Face raw JSON URL.")
    parser.add_argument("--output", type=str, default="smoldoc_clean_parallel.csv", help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_and_extract_pairs(args.url, args.output)
