"""Data merging, normalization, and filtering pipeline for Kokborok–English parallel corpus.

Combines bilingual sentence pairs across:
1. Kokborok–English Bible parallel corpus
2. Learn Kokborok Language guide (TTAADC)
3. Hugging Face datasets (smoldoc aligned sentence pairs & paragraphs)
4. Manual translations

Filters for sentence quality, removes duplicates, handles missing values,
and enforces full sentence length constraints (>2 tokens per side).
"""

import argparse
import logging
import os
from typing import List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_full_sentence(text: str, min_words: int = 2) -> bool:
    """Check if the given text contains strictly more than min_words words."""
    if not isinstance(text, str):
        return False
    return len(text.strip().split()) > min_words


def clean_text(text: str) -> str:
    """Normalize whitespace and strip extra formatting."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def load_and_merge_datasets(
    file_paths: List[str],
    min_words: int = 2,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Load multiple bilingual CSV files, clean, merge, deduplicate, and filter.

    Args:
        file_paths: List of absolute or relative paths to input CSV files.
        min_words: Minimum word threshold per sentence side (default >2 words).
        output_path: Optional path to save the resulting clean dataset.

    Returns:
        pd.DataFrame containing cleaned, deduplicated parallel sentence pairs.
    """
    dfs: List[pd.DataFrame] = []

    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found, skipping: {path}")
            continue

        try:
            logger.info(f"Reading: {os.path.basename(path)}")
            df = pd.read_csv(path, on_bad_lines="skip", encoding="utf-8")
            # Normalize column names
            df.columns = [col.strip().lower() for col in df.columns]

            if "english" not in df.columns or "kokborok" not in df.columns:
                logger.warning(
                    f"Skipping {path}: missing required columns 'english' and 'kokborok' (found {df.columns.tolist()})"
                )
                continue

            df = df[["english", "kokborok"]].dropna()
            df["english"] = df["english"].apply(clean_text)
            df["kokborok"] = df["kokborok"].apply(clean_text)
            dfs.append(df)
            logger.info(f"  -> Loaded {len(df):,} valid rows from {os.path.basename(path)}")
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")

    if not dfs:
        raise ValueError("No valid data loaded from provided file paths.")

    merged_df = pd.concat(dfs, ignore_index=True)
    initial_count = len(merged_df)
    logger.info(f"Total merged raw rows: {initial_count:,}")

    # Deduplication
    merged_df.drop_duplicates(subset=["english", "kokborok"], inplace=True)
    after_dedup = len(merged_df)
    logger.info(f"Rows after deduplication: {after_dedup:,} (removed {initial_count - after_dedup:,} duplicates)")

    # Filter full sentences
    filtered_df = merged_df[
        merged_df["english"].apply(lambda t: is_full_sentence(t, min_words))
        & merged_df["kokborok"].apply(lambda t: is_full_sentence(t, min_words))
    ].copy()

    filtered_count = len(filtered_df)
    logger.info(f"Rows after sentence length filtering (> {min_words} words): {filtered_count:,}")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        filtered_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved cleaned corpus to: {output_path}")

    return filtered_df


def parse_args():
    parser = argparse.ArgumentParser(description="Merge and clean Kokborok-English parallel datasets.")
    parser.add_argument("--data-dir", type=str, default=".", help="Base directory containing dataset files.")
    parser.add_argument("--output", type=str, default="kokborok_full_sentences.csv", help="Output CSV path.")
    parser.add_argument("--min-words", type=int, default=2, help="Minimum words per sentence side (strictly > threshold).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_files = [
        os.path.join(args.data_dir, "bible_parallel_clean_norm.csv"),
        os.path.join(args.data_dir, "learn_kokborok_parallel_clean.csv"),
        os.path.join(args.data_dir, "smoldoc_clean_parallel.csv"),
        os.path.join(args.data_dir, "smoldoc_cleaned_parallel.csv"),
    ]
    load_and_merge_datasets(target_files, min_words=args.min_words, output_path=args.output)
