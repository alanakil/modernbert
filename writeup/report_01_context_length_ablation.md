# Report: Context-Length Ablation — BERT vs. ModernBERT on 20 Newsgroups

## Setup

- **Dataset:** SetFit/20_newsgroups — 20-class newsgroup topic classification
- **Training data:** 20% stratified subsample (~2,260 examples)
- **Models:** `google-bert/bert-base-cased`, `answerdotai/ModernBERT-base`
- **Context lengths swept:** 50, 128, 256, 512 tokens
- **Epochs:** 10 | **LR:** 1e-5 | **Weight decay:** 0.01 | **Effective batch size:** 256
- **Seeds:** 42 (single seed; 422, 1337 reserved for full run)
- **Hardware:** Apple M-series, MPS backend, bf16 mixed precision

---

## Results Summary

| Model | len=50 | len=128 | len=256 | len=512 |
|---|---|---|---|---|
| BERT-base (macro F1) | 0.172 | 0.169 | **0.190** | 0.183 |
| ModernBERT-base (macro F1) | 0.351 | 0.470 | 0.483 | **0.488** |
| BERT accuracy | 0.204 | 0.206 | 0.233 | 0.227 |
| ModernBERT accuracy | 0.360 | 0.480 | 0.492 | 0.497 |
| BERT AUC-ROC | 0.788 | 0.809 | 0.819 | 0.817 |
| ModernBERT AUC-ROC | 0.862 | 0.909 | 0.914 | 0.916 |

Training time and peak memory:

| Model | len=50 | len=128 | len=256 | len=512 |
|---|---|---|---|---|
| BERT time (min) | 4.9 | 12.9 | 28.5 | 69.2 |
| ModernBERT time (min) | 10.3 | 27.4 | 58.8 | 135.8 |
| BERT peak mem (GB) | 5.98 | 6.30 | 6.67 | 19.87 |
| ModernBERT peak mem (GB) | 8.47 | 8.75 | 10.33 | 21.27 |

---

## Finding 1: ModernBERT dominates at every context length

Even at the shortest context (50 tokens), ModernBERT (F1=0.351) outperforms BERT's best result
(F1=0.190 at len=256) by 85%. The gap exists at the very first data point and widens with
context. This is not a context-length story for BERT — ModernBERT is simply a stronger base
model for classification, regardless of sequence length.

The >1 pp gap between the two models exists from len=50 onward (annotated on the plot). AUC-ROC
confirms the same pattern: ModernBERT ranks classes far more reliably (0.916 vs 0.817 at len=512).

## Finding 2: BERT is largely insensitive to context length

BERT macro F1 moves only 0.172 → 0.190 → 0.183 across 50→256→512. There is no clear monotonic
gain. The slight peak at 256 and regression at 512 is consistent with:

1. **Absolute positional embeddings** degrading at long contexts — BERT was trained to 512 but
   the embeddings for positions 400–512 are less well-trained than 0–128.
2. **Corpus statistics** — the median 20 Newsgroups post is ~125–129 tokens (WordPiece), so
   len=128 already covers ~50% of documents. Going to 256 or 512 adds padded or noisy tails
   without adding much discriminative signal for BERT.

Per the token length audit in the plan, ~50% of docs fit within 128 tokens, ~75% within 256,
~90% within 512. Yet BERT gains only ~1 pp going from covering 50% to 90% of documents.

## Finding 3: ModernBERT scales meaningfully with context, but gains taper fast

| Δ F1 | len=50→128 | len=128→256 | len=256→512 |
|---|---|---|---|
| ModernBERT | +**11.8 pp** | +1.4 pp | +0.5 pp |
| BERT | −0.3 pp | +2.1 pp | −0.7 pp |

The big payoff for ModernBERT is the 50→128 jump. This aligns with coverage: going from ~10%
to ~50% of the corpus unlocks a large share of full-length posts. Beyond 128 the marginal
document covered is longer-tail content (threading, quoted text, headers) that contributes
less discriminative signal per token.

The implication: **len=256 is the efficient operating point for ModernBERT** — it captures 75%
of documents and 98.6% of the achievable F1 (0.483 vs 0.488 at len=512), at 43% of the
compute cost.

## Finding 4: BERT has a systematic precision–recall imbalance; ModernBERT does not

| Model | len | macro_precision | macro_recall | macro_F1 |
|---|---|---|---|---|
| BERT | 50 | 0.274 | 0.201 | 0.172 |
| BERT | 128 | 0.289 | 0.205 | 0.169 |
| BERT | 256 | 0.311 | 0.231 | 0.190 |
| BERT | 512 | 0.321 | 0.226 | 0.183 |
| ModernBERT | 50 | 0.353 | 0.353 | 0.351 |
| ModernBERT | 128 | 0.474 | 0.470 | 0.470 |
| ModernBERT | 256 | 0.489 | 0.483 | 0.483 |
| ModernBERT | 512 | 0.494 | 0.488 | 0.488 |

BERT's macro precision exceeds macro recall by ~8–9 pp at every context length. This indicates
BERT over-concentrates predictions on a small number of high-confidence classes — it learns a few
categories well and largely ignores the rest. The per-class heatmap confirms this: class_14
(sci.space) has F1=0.00 at len=128–512, meaning BERT essentially never predicts it. Several
other classes (class_4, class_16) are similarly collapsed. Increasing context length does not
fix this imbalance — precision and recall move in tandem, preserving the ~8 pp gap.

ModernBERT shows near-perfect precision–recall balance (≤1 pp gap at all lengths). Its
predictions are distributed across all 20 classes, which is why its macro F1 tracks so closely
to both component metrics.

## Finding 5: Compute cost is steep and disproportionate at len=512

Going from len=256 to len=512 doubles sequence length and costs:
- **ModernBERT:** 2.3× more time (58.8 → 135.8 min), +11 GB peak memory, +0.5 pp F1
- **BERT:** 2.4× more time, +13.2 GB peak memory, −0.7 pp F1 (negative return)

BERT at len=512 is strictly worse than at len=256: more expensive and lower F1. This makes
len=512 BERT a dominated option on the efficiency frontier.

## Finding 6: Per-class patterns reveal where context matters most

**ModernBERT:** Large 50→128 gains concentrated in:
- `class_17` (talk.politics.mideast): 0.16 → 0.47 (+31 pp) — long political posts, context-heavy
- `class_9` (rec.sport.baseball): 0.49 → 0.66 (+17 pp)
- `class_10` (rec.sport.hockey): 0.59 → 0.75 (+16 pp)

Classes with uniformly weak performance across all lengths:
- `class_19` (talk.religion.misc): 0.14–0.23 — highly confusable with alt.atheism and soc.religion.christian
- `class_0` (alt.atheism): 0.23–0.33 — same confusion cluster

**BERT:** The heatmap is far paler overall. Notable:
- `class_15` (soc.religion.christian): 0.36 → 0.52 at len=512 — BERT's only strong class, and one where verbose religious text helps even BERT
- `class_10` (rec.sport.hockey): 0.04 → 0.24 — strong context dependency for BERT too, consistent with hockey posts being long statistical recaps
- `class_14` (sci.space): stuck at 0.00 for BERT at len=128–512 — complete failure to separate from other sci.* categories

The classes where longer context helps most tend to be those with distinctive vocabulary that
appears later in posts (sport statistics, political arguments) rather than in subject-line-style
opening tokens.

---

## Recommendation for Future Experiments

**Use ModernBERT at len=256** as the default for subsequent ablations:
- Covers 75% of the corpus
- Within 0.5 pp of the maximum observed F1
- ~2.3× cheaper than len=512
- Strong baseline: F1=0.483 on 20% of training data; should comfortably exceed 0.60+ on full data

**Do not pursue BERT further** for main results. The performance gap (0.190 vs 0.488 at their
respective best lengths) is too large to be explained by any remaining hyperparameter search.
BERT can appear as a reference point but not as a competitive baseline.

**Three seeds are needed** before drawing strong per-class conclusions — single-seed variance
could shift individual class F1 by ±5–10 pp. The current results are directionally reliable
but the per-class heatmap values should be treated as estimates.

---

## Plots

- `results/plots/macro_f1_vs_length.png` — macro F1 vs. context length, both models
- `results/plots/train_time_vs_length.png` — training time vs. context length
- `results/plots/per_class_f1_bert.png` — per-class F1 heatmap, BERT
- `results/plots/per_class_f1_modernbert.png` — per-class F1 heatmap, ModernBERT

## Caveats

- Single seed (42) — error bars will be available after the full 3-seed run
- 20% training subsample — absolute F1 values will be higher on full data; relative trends expected to hold
- MPS peak memory readings are post-run snapshots, not true training peaks
