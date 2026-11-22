"""Inference pipeline for Kokborok–English Neural Machine Translation.

Supports Kokborok-to-English and English-to-Kokborok translation using
the fine-tuned M2M100 multilingual Transformer model.
"""

import argparse
import logging
import os
import sys
from typing import List, Union
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class KokborokTranslator:
    """Wrapper class for Kokborok-English translation inference."""

    def __init__(
        self,
        model_path_or_name: str = "facebook/m2m100_418M",
        src_lang: str = "hi",
        tgt_lang: str = "en",
        device: Optional[str] = None
    ):
        """Initialize tokenizer and model for inference.

        Args:
            model_path_or_name: Local checkpoint directory or Hugging Face model ID.
            src_lang: M2M100 language code for source (default: 'hi' for Indic transfer).
            tgt_lang: M2M100 language code for target (default: 'en').
            device: Compute device ('cuda', 'mps', 'cpu', or auto-detected if None).
        """
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading model from '{model_path_or_name}' on device: {self.device}")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_path_or_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_path_or_name)
        self.model.to(self.device)
        self.model.eval()

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.src_lang = src_lang
        self.tgt_lang_id = self.tokenizer.get_lang_id(tgt_lang)

    def translate(
        self,
        text: Union[str, List[str]],
        max_length: int = 64,
        num_beams: int = 5
    ) -> Union[str, List[str]]:
        """Translate single string or list of strings.

        Args:
            text: Input sentence or list of sentences.
            max_length: Maximum generated token sequence length.
            num_beams: Beam search width (default: 5).

        Returns:
            Translated string or list of translated strings.
        """
        is_single = isinstance(text, str)
        inputs_list = [text] if is_single else text

        self.tokenizer.src_lang = self.src_lang
        encoded = self.tokenizer(
            inputs_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tgt_lang_id,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return decoded[0] if is_single else decoded


def parse_args():
    parser = argparse.ArgumentParser(description="Translate Kokborok text to English.")
    parser.add_argument("--model-path", type=str, default="facebook/m2m100_418M", help="Model checkpoint path.")
    parser.add_argument("--text", type=str, default=None, help="Input sentence to translate.")
    parser.add_argument("--src-lang", type=str, default="hi", help="Source language code in M2M100 (default: hi).")
    parser.add_argument("--tgt-lang", type=str, default="en", help="Target language code (default: en).")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam search width.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive translation session.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    translator = KokborokTranslator(
        model_path_or_name=args.model_path,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    )

    if args.interactive:
        print("\n=== Kokborok -> English Translation Session ===")
        print("Type your Kokborok sentence and press Enter (or 'quit' / 'exit' to stop):\n")
        while True:
            try:
                line = input("Kokborok > ").strip()
                if not line or line.lower() in ["quit", "exit"]:
                    break
                translation = translator.translate(line, num_beams=args.beam_size)
                print(f"English   > {translation}\n")
            except (KeyboardInterrupt, EOFError):
                break
    elif args.text:
        res = translator.translate(args.text, num_beams=args.beam_size)
        print(f"\nSource:     {args.text}")
        print(f"Translated: {res}\n")
    else:
        sample = "Nwng Bahai Tong."
        res = translator.translate(sample, num_beams=args.beam_size)
        print(f"Sample Source:     {sample}")
        print(f"Sample Translated: {res}")
