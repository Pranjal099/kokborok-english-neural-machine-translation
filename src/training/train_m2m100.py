"""Fine-tuning pipeline for Kokborok–English Neural Machine Translation using M2M100.

Implements the training methodology described in:
"Developing Neural Machine Translation Tool: Kokborok–English, a Step Toward Preserving Indigenous Languages"
(Das, Pranjal, Sarkar, 2026, CRC Press / Taylor & Francis).
"""

import argparse
import logging
import os
import time
from typing import Dict, List, Optional
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_datasets(
    csv_path: str,
    tokenizer,
    max_length: int = 64,
    test_size: float = 0.1,
    random_seed: int = 42
):
    """Load and tokenize parallel dataset for M2M100 fine-tuning.

    Args:
        csv_path: Path to parallel CSV (columns: 'kokborok', 'english').
        tokenizer: Initialized M2M100Tokenizer.
        max_length: Maximum sequence token length (default: 64).
        test_size: Proportion for test/validation holdout.
        random_seed: Random seed for reproducible splitting.

    Returns:
        Tuple of (train_dataset, eval_dataset, raw_test_df)
    """
    from datasets import Dataset as HFDataset

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    logger.info(f"Loading parallel corpus from: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = [c.strip().lower() for c in df.columns]

    if "kokborok" not in df.columns or "english" not in df.columns:
        raise ValueError("Dataset must contain 'kokborok' and 'english' columns.")

    df = df.dropna(subset=["kokborok", "english"])
    logger.info(f"Loaded {len(df):,} valid bilingual sentence pairs.")

    # Tokenization preprocessing function
    def preprocess_batch(batch):
        return tokenizer(
            batch["src_text"],
            text_target=batch["tgt_text"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )

    # Convert to HuggingFace Dataset
    hf_dataset = HFDataset.from_dict({
        "src_text": df["kokborok"].astype(str).tolist(),
        "tgt_text": df["english"].astype(str).tolist()
    })

    split = hf_dataset.train_test_split(test_size=test_size, seed=random_seed)
    tokenized_train = split["train"].map(preprocess_batch, batched=True, remove_columns=["src_text", "tgt_text"])
    tokenized_eval = split["test"].map(preprocess_batch, batched=True, remove_columns=["src_text", "tgt_text"])

    logger.info(f"Tokenized dataset splits: Train={len(tokenized_train):,}, Validation/Test={len(tokenized_eval):,}")
    return tokenized_train, tokenized_eval, df.tail(int(len(df) * test_size))


def train(
    dataset_path: str,
    output_dir: str = "./m2m100_kokborok_finetuned",
    model_name: str = "facebook/m2m100_418M",
    source_lang: str = "hi",
    target_lang: str = "en",
    learning_rate: float = 5e-5,
    batch_size: int = 32,
    epochs: int = 12,
    weight_decay: float = 0.01,
    max_length: int = 64,
    logging_steps: int = 500,
    fp16: bool = True
):
    """Run fine-tuning workflow for M2M100 on Kokborok–English translation."""
    from transformers import (
        M2M100ForConditionalGeneration,
        M2M100Tokenizer,
        Trainer,
        TrainingArguments
    )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Active compute device: {device.upper()}")
    use_fp16 = fp16 and torch.cuda.is_available()

    logger.info(f"Loading pretrained model and tokenizer: {model_name}")
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)

    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang

    train_data, eval_data, test_df = prepare_datasets(
        csv_path=dataset_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=use_fp16,
        report_to=[],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
    )

    logger.info("Starting M2M100 fine-tuning...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds ({elapsed / 3600:.2f} hours).")

    # Save final model artifacts
    final_save_path = os.path.join(output_dir, "final_model")
    logger.info(f"Saving final model to: {final_save_path}")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune M2M100 for Kokborok-to-English translation.")
    parser.add_argument("--data-path", type=str, default="kokborok_full_sentences.csv", help="Parallel CSV path.")
    parser.add_argument("--output-dir", type=str, default="./m2m100_finetuned", help="Output directory for checkpoints.")
    parser.add_argument("--model-name", type=str, default="facebook/m2m100_418M", help="Pretrained model identifier.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per device (default: 32).")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs (default: 12).")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01).")
    parser.add_argument("--max-length", type=int, default=64, help="Max sequence length (default: 64).")
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        fp16=not args.no_fp16
    )
