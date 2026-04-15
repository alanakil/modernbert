# Plan 11: Hyperparameter Search — BERT & ModernBERT on 20 Newsgroups

## Objective

Find optimal fine-tuning hyperparameters for `google-bert/bert-base-cased` and
`answerdotai/ModernBERT-base` on the 20 Newsgroups classification task, then use
those winners to run a fair LoRA vs. full fine-tuning comparison.

---

## Hardware Configuration

The script auto-detects device via `helpers.identify_device()` and applies the
appropriate config. **Gradient checkpointing must always be enabled** — it is the
primary guard against OOM and is already present in all existing scripts.

### Memory Estimates (bf16, max_length=512, gradient checkpointing ON)

| Model | Params | Static mem | Batch=32 peak | Batch=64 peak |
|---|---|---|---|---|
| BERT-base | 110M | ~1.76GB | **~2.3GB** | **~2.8GB** |
| ModernBERT-base | 149M | ~2.4GB | **~3.1GB** | **~3.9GB** |

Static memory = weights + gradients + AdamW states (fp32 master copy + m + v).
Peak includes checkpoint tensors (one hidden state per layer) plus the worst-case
single-layer recompute. ModernBERT's local attention (window=128 for 21/22 layers)
keeps its activation memory well below BERT's despite having more layers.

**Both models fit safely within 12GB VRAM at batch=64 with gradient checkpointing.**
Without gradient checkpointing, BERT at batch=64 uses ~6.9GB and ModernBERT ~5.8GB —
still within 12GB, but gradient checkpointing should remain on as a safety margin
against fragmentation and framework overhead (~10–20% buffer consumed by CUDA/MPS).

### Per-Device Batch Config

Effective batch size is fixed at 256 across all runs so results are comparable.

| Device | `per_device_train_batch_size` | `gradient_accumulation_steps` | Notes |
|---|---|---|---|
| MPS (Apple Silicon) | 32 | 8 | Conservative; MPS unified memory is slower |
| CUDA (RTX 3060 12GB) | 32 | 8 | Safe starting point; try 64/4 if runs feel slow |
| CUDA (optional upgrade) | 64 | 4 | Estimated ~2.8–3.9GB peak; safe with grad ckpt ON |

### CUDA-Specific Settings (applied automatically when device=cuda)

```python
torch.backends.cuda.matmul.allow_tf32 = True   # free ~10% speedup on Ampere
dataloader_num_workers = 4                       # parallel data loading
dataloader_pin_memory = True                     # faster CPU→GPU transfers
```

MPS sets `dataloader_num_workers=0` (MPS crashes with >0) and
`dataloader_pin_memory=False`.

---

## Fixed Choices (not searched)

| Setting | Value | Rationale |
|---|---|---|
| Dataset fraction | 20% stratified | ~113 examples/class — stable HP rankings; 10% is too noisy |
| Max sequence length | 512 | Consistent with prior ablation winner |
| Effective batch size | 256 | Batch and LR are coupled — fix one for clean search |
| Epochs | 10 | Consistent with all prior experiments |
| Gradient checkpointing | Always ON | Required for memory safety on both MPS and CUDA |
| Optuna objective | minimize eval loss on test split | Fast, smooth signal. **Note:** test split is used for both HP selection and final reporting — acknowledged leakage, standard for public benchmarks |
| Optimizer | AdamW | HuggingFace Trainer default |

---

## Search Space (Optuna, per trial)

| Parameter | Type | Range / Choices |
|---|---|---|
| `learning_rate` | log-uniform | [1e-5, 3e-4] |
| `weight_decay` | categorical | [0.0, 0.01, 0.1] |
| `warmup_ratio` | uniform | [0.0, 0.1] |
| `lr_scheduler_type` | categorical | ["linear", "cosine"] |
| `classifier_dropout` | categorical | [0.0, 0.1, 0.2] |

---

## Phases

### Phase 1 — BERT Hyperparameter Search

- **Model:** `google-bert/bert-base-cased`
- **Trials:** 30 (Optuna TPE sampler + HyperbandPruner)
- **Seed during search:** 42 (single seed per trial for speed)
- **Output:** `results/results_11_hp_search/bert_optuna_trials.csv`, best HPs logged
- **Runs:** 30

### Phase 2 — ModernBERT Hyperparameter Search

- **Model:** `answerdotai/ModernBERT-base`
- **Trials:** 30 — independent search, not transferred from BERT
  (different architecture: rotary embeddings, GLU FFN — optimal HPs may differ)
- **Output:** `results/results_11_hp_search/modernbert_optuna_trials.csv`, best HPs logged
- **Runs:** 30

### Phase 3 — Final Evaluation with Best HPs

#### 3a. Full Fine-tuning

Use the winning HP set from Phase 1 (BERT) and Phase 2 (ModernBERT).
Run **3 seeds** each to get mean ± std macro F1 for the paper.

| Model | Seeds | Runs |
|---|---|---|
| BERT (best HPs) | 42, 422, 1337 | 3 |
| ModernBERT (best HPs) | 42, 422, 1337 | 3 |

#### 3b. LoRA vs. Full Fine-tuning Comparison

LoRA ranks × target module configs × models × LR grid (1 seed each — trend matters more than exact numbers).

| Axis | Values |
|---|---|
| LoRA ranks | [4, 8, 16, 32, 64] |
| Target module configs | standard, aggressive (per model) |
| Models | bert, modernbert |
| LoRA LR grid | [3e-4, 1e-3, 3e-3] |
| Seeds | 1 (seed=42) |

Runs: 5 ranks × 2 configs × 2 models × 3 LRs = **60 runs**

Full fine-tuning baselines (from 3a) serve as the comparison point.

---

## Run Count Summary

| Phase | Description | Runs |
|---|---|---|
| Phase 1 | BERT Optuna search | 30 |
| Phase 2 | ModernBERT Optuna search | 30 |
| Phase 3a | Full fine-tuning, 3 seeds × 2 models | 6 |
| Phase 3b | LoRA sweep | 60 |
| **Total** | | **126** |

---

## Implementation

New script: `src/train_hp_search.py`

Key design decisions:
- Use `optuna` with `TPESampler` and `HyperbandPruner` (prune bad trials at epoch 3 and 6)
- Intermediate values reported via `trial.report()` calls after each eval epoch via a custom Trainer callback
- Resume-safe: completed trials persisted to Optuna SQLite journal at `results/results_11_hp_search/{model}_optuna.db`
- Device detected once at startup; batch config and dataloader settings derived from device type
- After both searches complete, Phase 3 runs use the same sweep infrastructure as `train_lora.py`

---

## Metrics Reported

- Primary: **macro F1** (mean ± std over seeds for Phase 3a)
- Secondary: accuracy, macro precision, macro recall, AUC-ROC
- Efficiency: trainable parameter %, train time (seconds), peak memory (GB)

---

## Leakage Acknowledgement

The `SetFit/20_newsgroups` dataset has only `train` and `test` splits. Optuna
optimizes eval loss on the `test` split, which is also used for final metric
reporting. This is consistent with how public benchmark results are typically
reported and will be noted explicitly in the writeup.
