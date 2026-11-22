"""Batch tokenization utility for parallel train/val/test corpus splits."""

import argparse
import logging
import os
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def tokenize_file(input_file: str, output_file: str, sp_processor):
    """Tokenize each line in input_file into integer token IDs written to output_file."""
    if not os.path.exists(input_file):
        logger.warning(f"File not found, skipping: {input_file}")
        return

    logger.info(f"Tokenizing {input_file} -> {output_file}")
    count = 0
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line_str = line.strip()
            token_ids = sp_processor.encode(line_str, out_type=int)
            fout.write(" ".join(map(str, token_ids)) + "\n")
            count += 1

    logger.info(f"  Processed {count:,} lines.")


def batch_tokenize_splits(model_path: str, file_pairs: List[Tuple[str, str]]):
    """Run batch tokenization across all specified train/val/test splits."""
    try:
        import sentencepiece as spm
    except ImportError:
        logger.error("SentencePiece is not installed (`pip install sentencepiece`).")
        return

    if not os.path.exists(model_path):
        logger.error(f"Tokenizer model not found: {model_path}")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    logger.info(f"Loaded tokenizer from {model_path} (Vocab size: {sp.get_piece_size():,})")

    for in_f, out_f in file_pairs:
        tokenize_file(in_f, out_f, sp)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch tokenize parallel corpus files.")
    parser.add_argument("--model", type=str, default="kok_eng_tokenizer.model", help="Trained SentencePiece model.")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing split files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    splits = [
        (os.path.join(args.data_dir, "train.kok"), os.path.join(args.data_dir, "train.tok.kok")),
        (os.path.join(args.data_dir, "train.eng"), os.path.join(args.data_dir, "train.tok.eng")),
        (os.path.join(args.data_dir, "val.kok"), os.path.join(args.data_dir, "val.tok.kok")),
        (os.path.join(args.data_dir, "val.eng"), os.path.join(args.data_dir, "val.tok.eng")),
        (os.path.join(args.data_dir, "test.kok"), os.path.join(args.data_dir, "test.tok.kok")),
        (os.path.join(args.data_dir, "test.eng"), os.path.join(args.data_dir, "test.tok.eng")),
    ]
    batch_tokenize_splits(args.model, splits)
