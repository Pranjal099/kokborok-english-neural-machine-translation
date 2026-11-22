"""OCR extraction pipeline for manual Kokborok–English learning material.

Uses EasyOCR to process photographed/scanned bilingual book pages
(e.g., Learn Kokborok Language guide published by TTAADC), extracts parallel pairs,
cleans punctuation and whitespace, and outputs a formatted CSV.
"""

import argparse
import logging
import os
import re
from typing import List, Dict
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize extracted text string."""
    return " ".join(text.split()).strip()


def extract_pairs_from_images(
    image_folder: str,
    output_file: str,
    ocr_languages: List[str] = ["en"]
) -> pd.DataFrame:
    """Run OCR over images in a folder and extract bilingual pairs.

    Args:
        image_folder: Directory containing scanned/photographed book pages.
        output_file: Path to output CSV.
        ocr_languages: Language list for EasyOCR reader.

    Returns:
        pd.DataFrame containing extracted pairs.
    """
    try:
        import easyocr
    except ImportError:
        logger.error("EasyOCR is not installed. Install with `pip install easyocr`.")
        return pd.DataFrame()

    if not os.path.isdir(image_folder):
        logger.error(f"Image directory does not exist: {image_folder}")
        return pd.DataFrame()

    reader = easyocr.Reader(ocr_languages)
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))
    ]
    image_files.sort()
    logger.info(f"Found {len(image_files)} image files in {image_folder}")

    pairs: List[Dict[str, str]] = []

    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        logger.info(f"Processing OCR on: {img_name}")
        try:
            lines = reader.readtext(img_path, detail=0, paragraph=False)
        except Exception as e:
            logger.error(f"Error processing {img_name}: {e}")
            continue

        for line in lines:
            line_str = clean_text(line)
            if not line_str:
                continue

            if "," in line_str:
                parts = line_str.split(",", 1)
                eng, kok = clean_text(parts[0]), clean_text(parts[1])
            else:
                parts = re.split(r"\s{2,}|\t", line_str)
                if len(parts) >= 2:
                    eng, kok = clean_text(parts[0]), clean_text(" ".join(parts[1:]))
                else:
                    continue

            if eng and kok and len(eng) > 1 and len(kok) > 1:
                pairs.append({"english": eng, "kokborok": kok})

    df = pd.DataFrame(pairs)
    if df.empty:
        logger.warning("No sentence pairs extracted.")
        return df

    df.drop_duplicates(inplace=True)
    df = df[(df["english"] != "") & (df["kokborok"] != "")]
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(df):,} cleaned pairs to {output_file}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Extract parallel text from images via OCR.")
    parser.add_argument("--image-dir", type=str, default="manual_trans", help="Folder containing scanned images.")
    parser.add_argument("--output", type=str, default="learn_kokborok_parallel_clean.csv", help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_pairs_from_images(args.image_dir, args.output)
