# Project Plan: LoRA vs Full Fine-Tuning

## Goal
Compare parameter-efficient fine-tuning (LoRA) against full fine-tuning for both BERT and ModernBERT, and determine how much LoRA's rank needs to increase to match full fine-tuning performance for each model.

## Background
Full fine-tuning updates every parameter in the model. LoRA (Low-Rank Adaptation) injects trainable low-rank matrices A and B into the attention weight matrices, freezing everything else. The trainable parameter count scales with rank r.

ModernBERT differs from BERT structurally: no bias terms, GeGLU activations, fused QKV projections (`Wqkv`), and alternating local/global attention layers (every 3rd layer is global: 0, 3, 6, ..., 21). These differences mean the LoRA target modules differ between models.

## Confirmed Decisions

### Dataset and training setup
- **Dataset**: Full 20 Newsgroups (no subsampling)
- **Seeds**: 1 (seed=42)
- **max_length**: 512
- **Batch config**: `batch_size=32`, `grad_accum=8` (effective batch=256)
- **Epochs**: 10, no early stopping
- **LR**: 1e-5, weight decay=0.01, bf16, MPS device
- **Full fine-tuning**: Re-run fresh (not reused from context-length ablation)

### LoRA hyperparameters
- **Ranks**: {4, 8, 16, 32, 64}; rank=0 denotes full fine-tuning baseline
- **Alpha**: `2 * r` (keeps effective scaling constant across ranks for a clean comparison)
- **Dropout**: 0.05
- **Base model**: Fresh load per config (no adapter swapping); `del model; torch.mps.empty_cache()` between runs

### Target module configurations (verified against actual model architecture)

BERT (`google-bert/bert-base-cased`) linear modules per layer:
- `attention.self.query`, `attention.self.key`, `attention.self.value` (768×768 each)
- `attention.output.dense` (768×768)
- `intermediate.dense` (768×3072), `output.dense` (3072×768)

ModernBERT (`answerdotai/ModernBERT-base`) linear modules per layer:
- `attn.Wqkv` (768×2304, fused Q+K+V), `attn.Wo` (768×768)
- `mlp.Wi` (768×2304, GeGLU gate+value fused), `mlp.Wo` (1152×768)

| Model | Config name | `target_modules` |
|---|---|---|
| BERT | standard | `["query", "value"]` |
| BERT | aggressive | `["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"]` |
| ModernBERT | standard | `["Wqkv"]` |
| ModernBERT | aggressive | `["Wqkv", "Wo", "Wi", "mlp.Wo"]` |

Note: the global-attention-only experiment was cut (implementation complexity, hard to interpret).

### Total runs
5 ranks × 2 target configs × 2 models + 2 full fine-tuning baselines = **22 runs**

### Gradient checkpointing with PEFT
Call `model.enable_input_require_grads()` then `model.gradient_checkpointing_enable()` **before** `get_peft_model()`. Without this, gradient checkpointing silently breaks for LoRA models.

### Memory measurement
Approximate only: record `torch.mps.driver_allocated_memory()` at end of training (same as ablation). No dedicated measurement pass — all configs fit within 48GB with large headroom.

### Smoke test
BERT, rank=8, `["query", "value"]`, 1 epoch. Aborts if loss is NaN or does not decrease.

## Output locations
- Results CSV: `results/results_04_lora_vs_full/lora_vs_full.csv`
- Plots: `results/results_04_lora_vs_full/plots/`
- Checkpoints: `artifacts/lora_vs_full/{model_key}/rank{r}_{target_config}_seed{seed}/`
- Full fine-tuning checkpoints: `artifacts/lora_vs_full/{model_key}/full_seed{seed}/`

## CSV columns
```
model, lora_rank, target_modules, seed,
trainable_params, total_params, trainable_pct,
accuracy, macro_precision, macro_recall, macro_f1, auc_roc,
f1_class_0 .. f1_class_19,
train_time_seconds, peak_mps_memory_gb
```
`lora_rank=0` and `target_modules="full"` denotes full fine-tuning.

## Plots (`src/plot_lora.py`)
1. **Efficiency frontier**: macro F1 vs trainable parameter count (log x-axis), one curve per model. Horizontal dashed reference line at each model's full fine-tuning F1.
2. **Rank curve**: macro F1 vs rank r, one line per target module config, one subplot per model. Same reference line.

## Files to Create
- `src/train_lora.py` — sweep script (smoke test + full sweep, resume-safe via CSV)
- `src/plot_lora.py` — efficiency frontier and rank curve plots
