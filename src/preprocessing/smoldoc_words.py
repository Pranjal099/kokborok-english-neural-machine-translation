"""Hugging Face Dataset converter for cleaned parallel sentence pairs."""

import argparse
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_as_hf_dataset(csv_path: str):
    """Load parallel CSV into Hugging Face Dataset format."""
    try:
        from datasets import Dataset as HFDataset
    except ImportError:
        logger.error("Hugging Face datasets library is not installed (`pip install datasets`).")
        return None

    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]

    src_texts = df["kokborok"].astype(str).tolist()
    tgt_texts = df["english"].astype(str).tolist()

    dataset = HFDataset.from_dict({"src_text": src_texts, "tgt_text": tgt_texts})
    logger.info(f"Loaded Hugging Face Dataset with {len(dataset):,} examples.")
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV dataset to Hugging Face Dataset format.")
    parser.add_argument("--csv", type=str, default="smoldoc_cleaned_parallel.csv", help="Input parallel CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds = load_as_hf_dataset(args.csv)
