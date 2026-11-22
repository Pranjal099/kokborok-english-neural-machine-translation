"""Dataset verification and statistics calculation for Kokborok-English parallel corpora."""

import argparse
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_statistics(csv_path: str):
    """Compute and display summary statistics for a bilingual parallel dataset."""
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return

    logger.info(f"Analyzing dataset: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines="skip", encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]

    total_rows = len(df)
    print("=" * 60)
    print(f"Dataset File: {os.path.basename(csv_path)}")
    print(f"Total Rows:   {total_rows:,}")
    print("=" * 60)

    for col in ["english", "kokborok"]:
        if col not in df.columns:
            print(f"Column '{col}' not found in dataset columns: {df.columns.tolist()}")
            continue

        series = df[col].astype(str).str.strip()
        empty_count = (series == "").sum() + (series.str.lower() == "nan").sum()
        word_counts = series.apply(lambda s: len(s.split()))

        print(f"--- Column: {col} ---")
        print(f"  Empty / Missing values: {empty_count:,}")
        print(f"  Vocabulary / Word Count:")
        print(f"    Mean words per sentence: {word_counts.mean():.2f}")
        print(f"    Min words:               {word_counts.min()}")
        print(f"    Max words:               {word_counts.max()}")
        print(f"    Median (p50):            {word_counts.median():.0f}")
        print(f"    95th percentile (p95):   {word_counts.quantile(0.95):.0f}")
        print()

    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze parallel corpus statistics.")
    parser.add_argument(
        "--file",
        type=str,
        default="kokborok_full_sentences.csv",
        help="Path to parallel CSV dataset."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_statistics(args.file)
