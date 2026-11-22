"""SentencePiece encoding inspection and verification script."""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_tokenizer_encoding(model_path: str, sample_text: str):
    """Load SentencePiece tokenizer model and encode a sample sentence."""
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

    tokens = sp.encode(sample_text, out_type=str)
    token_ids = sp.encode(sample_text, out_type=int)

    print("=" * 60)
    print("SentencePiece Tokenizer Inspection")
    print(f"Model:     {model_path}")
    print(f"Vocab Size: {sp.get_piece_size():,}")
    print("=" * 60)
    print(f"Input Text:  {sample_text}")
    print(f"Tokens:      {tokens}")
    print(f"Token IDs:   {token_ids}")
    print(f"Decoded:     {sp.decode(token_ids)}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect SentencePiece encoding.")
    parser.add_argument("--model", type=str, default="kok_eng_tokenizer.model", help="Path to .model file.")
    parser.add_argument("--text", type=str, default="Nwng Bahai Tong.", help="Sample text to tokenize.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_tokenizer_encoding(args.model, args.text)
