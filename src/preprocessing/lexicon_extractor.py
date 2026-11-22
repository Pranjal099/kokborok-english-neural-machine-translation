"""Lexicon and dictionary entry extraction for Kokborok–English pairs."""

import argparse
import logging
import os
import re
from typing import List, Dict
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_lexicon_pairs(input_csv: str, output_csv: str) -> pd.DataFrame:
    """Extract and parse lexicon dictionary word pairs from raw tabular dumps."""
    if not os.path.exists(input_csv):
        logger.error(f"Input file not found: {input_csv}")
        return pd.DataFrame()

    df = pd.read_csv(input_csv, encoding="utf-8")
    raw_col = df[df.columns[0]].astype(str)

    entries = raw_col.str.split("entrpTrue").tolist()
    all_pairs = [x for sublist in entries for x in sublist if x.strip()]

    cleaned: List[Dict[str, str]] = []
    for pair in all_pairs:
        m = re.match(r"^([a-zA-Z\s,']+)(.*)$", pair.strip())
        if m:
            english = m.group(1).strip()
            kokborok = m.group(2).strip()
            if english and kokborok:
                cleaned.append({"english": english, "kokborok": kokborok})

    cleaned_df = pd.DataFrame(cleaned)
    cleaned_df.drop_duplicates(inplace=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    cleaned_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(cleaned_df):,} dictionary word pairs to {output_csv}")
    return cleaned_df


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Kokborok-English lexicon pairs.")
    parser.add_argument("--input", type=str, default="kokborok_dataset (1).csv", help="Input raw CSV.")
    parser.add_argument("--output", type=str, default="kokborok_english_cleaned.csv", help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_lexicon_pairs(args.input, args.output)
