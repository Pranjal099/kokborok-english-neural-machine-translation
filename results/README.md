# Experimental Results & Evaluation Benchmark

This document summarizes the quantitative and qualitative experimental results from the published research paper:
> **"Developing Neural Machine Translation Tool: Kokborok–English, a Step Toward Preserving Indigenous Languages"**  
> *Sanchali Das, Pranjal, and Mainak Sarkar*  
> *Sustainable Developments in Computer Engineering, Green Technology and Smart Systems*, CRC Press / Taylor & Francis Group, 2026.  
> DOI: [10.1201/9781003743767-121](https://doi.org/10.1201/9781003743767-121)

---

## 1. Quantitative Evaluation Summary

Quantitative evaluation was performed using standard sequence-to-sequence evaluation metrics on held-out Kokborok–English test sentence pairs:

| Evaluation Metric | Score Reported | Metric Description & Implementation |
| :--- | :--- | :--- |
| **BLEU (SacreBLEU)** | **70.56** | Standard corpus-level BLEU score with benchmark smoothing |
| **METEOR** | **0.76 (76.0%)** | Evaluates harmonic mean of precision and recall with stemming |
| **ROUGE-L** | **0.80 (80.0%)** | Longest Common Subsequence (LCS) F1 measure |

---

## 2. Baseline Architecture Comparison

To contextualize the performance of the fine-tuned M2M100 model on low-resource indigenous translation, performance was compared against classical NMT baseline architectures (as presented in Table 2 of the paper):

| Architecture | Model Configuration | Pretraining | BLEU Score | Experimental Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Seq2Seq (RNN + Attention)** | 2-layer BiLSTM Encoder-Decoder | None (From Scratch) | 5 – 8 | Struggles with long sentences and morphological sparsity |
| **Transformer (Base)** | 6 Encoder / 6 Decoder layers, 8 heads | None (From Scratch) | 9 – 13 | Suffers from severe overfitting due to small corpus size |
| **M2M100 (Fine-Tuned)** | **12 Encoder / 12 Decoder layers, 16 heads** | **Meta Multilingual (100 langs)** | **70.56** | **Cross-lingual transfer learning achieves state-of-the-art alignment** |

---

## 3. Training Convergence Trajectory

The model was fine-tuned for 12 epochs on Google Colab Pro utilizing NVIDIA L4 and A100 Tensor Core GPUs:

- **Initial Training Loss (Step 500):** `1.81`
- **Mid-Training Loss (Step 5,000):** `0.84`
- **Late Training Loss (Step 15,000+):** `~0.27`
- **Wall-Clock Training Time:** `10,047 seconds` (~2.79 hours)
- **Convergence Behavior:** Loss decreased steadily without divergence or overfitting, demonstrating the regularizing effect of multilingual pretraining.

---

## 4. Qualitative Translation Samples (Table 3 in Paper)

Below are representative qualitative translation examples published in the research paper:

### Example 1 (Biblical Domain)
- **Source (Kokborok):** `yaphango kaitor nokharok tei hayungno snam kha.`
- **Model Translation (English):** `in the beginning god created the heavens and the earth.`
- **Reference (English):** `in the beginning god created the heavens and the earth.`
- **Analysis:** Near-perfect grammatical and lexical translation owing to strong alignment in religious scripture.

### Example 2 (Conversational Domain)
- **Source (Kokborok):** `Nwng Bahai Tong.`
- **Model Translation (English):** `How are you.`
- **Reference (English):** `How are you.`
- **Analysis:** Fluent translation of high-frequency social expressions.

### Example 3 (Conversational Domain)
- **Source (Kokborok):** `Ang kahamno`
- **Model Translation (English):** `i am well`
- **Reference (English):** `I am well.`
- **Analysis:** Accurate semantic transfer for colloquial conversational responses.

### Example 4 (Complex Scripture Sentence)
- **Source (Kokborok):** `hakhe hor wngwi phung ai kha tei salthamni sal wng kha.`
- **Model Translation (English):** `and the evening and the morning were the third day.`
- **Reference (English):** `and the evening and the morning were the third day.`
- **Analysis:** Preserves temporal conjunctions and ordinal numbering without hallucination.

---

## 5. Verification Command

To independently verify the metric evaluation logic using the published benchmark test pairs:

```bash
python -m src.evaluation.evaluate_metrics --demo
```
