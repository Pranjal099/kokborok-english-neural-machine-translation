"""Quantitative evaluation script for Kokborok–English translation.

Computes:
1. BLEU / SacreBLEU score
2. METEOR score
3. ROUGE-L score

Matches the evaluation protocol reported in:
"Developing Neural Machine Translation Tool: Kokborok–English, a Step Toward Preserving Indigenous Languages"
(Das, Pranjal, Sarkar, 2026, CRC Press / Taylor & Francis).
"""

import argparse
import logging
import os
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute SacreBLEU, METEOR, and ROUGE-L metrics.

    Args:
        predictions: List of generated translation hypothesis strings.
        references: List of ground-truth target reference strings.

    Returns:
        Dictionary mapping metric names to scores.
    """
    results: Dict[str, float] = {}

    # 1. SacreBLEU (direct local computation without network latency)
    try:
        import sacrebleu
        formatted_refs = [[r] for r in references]
        bleu_res = sacrebleu.corpus_bleu(predictions, formatted_refs)
        results["BLEU"] = round(bleu_res.score, 2)
    except Exception as e:
        try:
            import evaluate
            bleu_metric = evaluate.load("sacrebleu")
            bleu_res = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
            results["BLEU"] = round(bleu_res["score"], 2)
        except Exception as e2:
            logger.warning(f"BLEU computation failed: {e2}")

    # 2. METEOR
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        try:
            nltk.data.find("corpora/wordnet.zip")
        except LookupError:
            try:
                nltk.download("wordnet", quiet=True)
                nltk.download("punkt", quiet=True)
            except Exception:
                pass

        meteor_scores = []
        for pred, ref in zip(predictions, references):
            pred_tok = pred.split()
            ref_tok = ref.split()
            meteor_scores.append(meteor_score([ref_tok], pred_tok))
        results["METEOR"] = round(sum(meteor_scores) / len(meteor_scores), 4)
    except Exception as e:
        logger.warning(f"METEOR computation skipped or failed: {e}")

    # 3. ROUGE-L
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        results["ROUGE_L"] = round(sum(rouge_scores) / len(rouge_scores), 4)
    except Exception as e:
        logger.warning(f"ROUGE-L computation skipped or failed: {e}")

    return results


def print_evaluation_summary(metrics: Dict[str, float]):
    """Print a clean formatted benchmark summary table."""
    print("\n" + "=" * 55)
    print("    Kokborok–English Translation Evaluation Summary")
    print("=" * 55)
    print(f"  {'Metric':<18} | {'Score':<15} | {'Unit / Format':<15}")
    print("-" * 55)
    if "BLEU" in metrics:
        print(f"  {'BLEU (SacreBLEU)':<18} | {metrics['BLEU']:<15} | 0 - 100 scale")
    if "METEOR" in metrics:
        print(f"  {'METEOR':<18} | {metrics['METEOR']:<15} | 0.0 - 1.0 (or {metrics['METEOR']*100:.1f}%)")
    if "ROUGE_L" in metrics:
        print(f"  {'ROUGE-L':<18} | {metrics['ROUGE_L']:<15} | 0.0 - 1.0 (or {metrics['ROUGE_L']*100:.1f}%)")
    print("=" * 55 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate translation predictions against references.")
    parser.add_argument("--preds-file", type=str, default=None, help="Text file with one hypothesis per line.")
    parser.add_argument("--refs-file", type=str, default=None, help="Text file with one reference per line.")
    parser.add_argument("--demo", action="store_true", help="Run self-test with verified examples.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.demo or (not args.preds_file and not args.refs_file):
        # Published paper verification examples
        sample_preds = [
            "in the beginning god created the heavens and the earth.",
            "How are you.",
            "i am well",
            "and the evening and the morning were the third day."
        ]
        sample_refs = [
            "in the beginning god created the heavens and the earth.",
            "How are you.",
            "I am well.",
            "and the evening and the morning were the third day."
        ]
        print("Running demonstration with published reference translation pairs...")
        scores = compute_all_metrics(sample_preds, sample_refs)
        print_evaluation_summary(scores)
    else:
        with open(args.preds_file, "r", encoding="utf-8") as fp:
            preds = [l.strip() for l in fp if l.strip()]
        with open(args.refs_file, "r", encoding="utf-8") as fr:
            refs = [l.strip() for l in fr if l.strip()]

        if len(preds) != len(refs):
            logger.error(f"Length mismatch: {len(preds)} predictions vs {len(refs)} references.")
        else:
            scores = compute_all_metrics(preds, refs)
            print_evaluation_summary(scores)
