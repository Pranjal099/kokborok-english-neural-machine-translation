"""Topic paragraph aggregation for Kokborok–English parallel text.

Extracts aggregated topic paragraphs from the smoldoc dataset hosted on Hugging Face:
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


def normalize_paragraph(text: str) -> str:
    """Normalize paragraph text and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_paragraphs(
    input_source: str = HF_DATASET_URL,
    local_json: str = "smoldoc__en_trp.json",
    output_csv: str = "smoldoc_cleaned_parallel.csv"
) -> pd.DataFrame:
    """Download (or read local) JSON and extract cleaned parallel paragraphs."""
    data = None
    if os.path.exists(local_json):
        logger.info(f"Loading local JSON file: {local_json}")
        with open(local_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        logger.info(f"Downloading dataset from: {input_source}")
        response = requests.get(input_source, timeout=30)
        response.raise_for_status()
        data = response.json()
        with open(local_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    rows: List[Dict[str, str]] = []
    for topic in data:
        srcs = topic.get("srcs", [])
        trgs = topic.get("trgs", [])
        eng_p = normalize_paragraph(" ".join(srcs))
        kok_p = normalize_paragraph(" ".join(trgs))
        if eng_p and kok_p:
            rows.append({"english": eng_p, "kokborok": kok_p})

    df = pd.DataFrame(rows)
    df.drop_duplicates(inplace=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(df):,} cleaned paragraph pairs to {output_csv}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Extract paragraph pairs from Hugging Face Kokborok dataset.")
    parser.add_argument("--source", type=str, default=HF_DATASET_URL, help="URL or path to dataset JSON.")
    parser.add_argument("--local-json", type=str, default="smoldoc__en_trp.json", help="Local cache JSON path.")
    parser.add_argument("--output", type=str, default="smoldoc_cleaned_parallel.csv", help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_paragraphs(args.source, args.local_json, args.output)
