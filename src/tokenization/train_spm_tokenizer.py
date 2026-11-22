"""SentencePiece BPE tokenizer trainer for Kokborok–English parallel corpus."""

import argparse
import logging
import os
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_combined_corpus(input_files: List[str], output_combined: str) -> bool:
    """Concatenate multiple monolingual text files into a single combined training file."""
    valid_files = [f for f in input_files if os.path.exists(f)]
    if not valid_files:
        logger.error(f"None of the specified input files exist: {input_files}")
        return False

    line_count = 0
    with open(output_combined, "w", encoding="utf-8") as outfile:
        for fname in valid_files:
            logger.info(f"Reading: {fname}")
            with open(fname, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)
                    line_count += 1

    logger.info(f"Combined {len(valid_files)} files ({line_count:,} lines) into {output_combined}")
    return True


def train_sentencepiece(
    input_file: str,
    model_prefix: str = "kok_eng_tokenizer",
    vocab_size: int = 32000,
    character_coverage: float = 1.0,
    model_type: str = "bpe"
):
    """Train a SentencePiece subword tokenizer model."""
    try:
        import sentencepiece as spm
    except ImportError:
        logger.error("SentencePiece is not installed (`pip install sentencepiece`).")
        return

    logger.info(
        f"Training SentencePiece tokenizer: type={model_type}, vocab_size={vocab_size}, "
        f"character_coverage={character_coverage}, prefix={model_prefix}"
    )

    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    logger.info(f"Tokenizer training complete! Generated {model_prefix}.model and {model_prefix}.vocab")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE tokenizer on Kokborok & English.")
    parser.add_argument("--kok-file", type=str, default="train.kok", help="Monolingual Kokborok train file.")
    parser.add_argument("--eng-file", type=str, default="train.eng", help="Monolingual English train file.")
    parser.add_argument("--combined-file", type=str, default="combined_train.txt", help="Combined text output file.")
    parser.add_argument("--model-prefix", type=str, default="kok_eng_tokenizer", help="Output model prefix.")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size.")
    parser.add_argument("--model-type", type=str, default="bpe", choices=["bpe", "unigram", "word", "char"])
    parser.add_argument("--character-coverage", type=float, default=1.0, help="Character coverage.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if prepare_combined_corpus([args.kok_file, args.eng_file], args.combined_file):
        train_sentencepiece(
            input_file=args.combined_file,
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            model_type=args.model_type
        )
