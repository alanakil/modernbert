# Project Plan: Context Length Ablation

## Goal
Determine at what context length ModernBERT's advantage over BERT becomes meaningful, given that the current project caps at 50 tokens while ModernBERT supports up to 8,192. This ablation is the main result — the sweep table and plots are cited in the writeup directly.

## Background
The current codebase truncates all documents to `max_length=50`. This is almost certainly hurting ModernBERT more than BERT because ModernBERT was pretrained at long contexts (1,024 tokens standard, 8,192 max) and its architectural choices — RoPE, alternating local/global attention, flash attention — are designed to exploit long sequences. BERT's hard 512-token limit and absolute positional embeddings mean it can't benefit beyond that ceiling anyway.

## Data Audit Findings

Token length distributions on the full 20 Newsgroups corpus (18,846 documents, train + test combined). BERT WordPiece and ModernBERT BPE produce nearly identical distributions.

| Statistic | BERT | ModernBERT |
|-----------|------|------------|
| Min       | 0    | 0          |
| Median    | 125  | 129        |
| Mean      | 342  | 340        |
| p75       | 251  | 255        |
| p90       | 488  | 510        |
| p95       | 835  | 876        |
| p99       | 3,087 | 3,268    |
| Max       | 143,937 | 126,224 |

Coverage by threshold:

| max_length | % docs covered |
|------------|---------------|
| 50         | ~10%           |
| 128        | ~50%           |
| 256        | ~75%           |
| 512        | ~90%           |
| 1,024      | ~96%           |
| 2,048      | ~98%           |

The max values (143K, 126K tokens) are extreme outliers — massive quoted email chains, not representative posts. The interesting story is in the 50→512 range where 90% of the corpus lives and BERT hits its hard ceiling.

**8,192 is dropped from the sweep.** Going from 2,048 to 8,192 covers ~1.7% more documents (outlier emails) and adds disproportionate compute cost for negligible signal.

## Steps

### 1. Refactor shared evaluation logic
Add a `compute_metrics(logits, labels, label_names)` function to `src/helpers.py`. It returns a dict with: accuracy, macro precision, macro recall, macro F1, per-class F1 (20 values), and AUC-ROC. Both the ablation script and `eval_compare_models.py` use this function. Write fresh — do not port from `eval_compare_models.py`.

### 2. Define the sweep

Run both models at the following `max_length` values:
`[50, 128, 256, 512, 1024, 2048]`

Note: BERT physically cannot run at 1,024+ (its positional embedding table only goes to 512). For BERT, cap at 512 and mark longer lengths as N/A in the results CSV. For ModernBERT, run all six values.

3 seeds per combination: `[42, 422, 1337]`.

Total runs: 6 lengths × 2 models × 3 seeds = 36 (minus BERT's 2 invalid lengths × 3 seeds = 30 actual training runs).

### 3. Training setup

Use these hyperparameters for all runs:
- lr = 1e-5, epochs = 10, weight_decay = 0.01
- bf16 mixed precision (`bf16=True` in `TrainingArguments`)
- Gradient checkpointing enabled at all times
- `device = torch.device("mps")`; clear between runs with `torch.mps.empty_cache()`
- Only one model in memory at a time

Recommended batch sizes per length (effective batch size held at 256 via gradient accumulation):

| max_length | batch_size | grad_accum_steps |
|------------|------------|-----------------|
| 50         | 256        | 1               |
| 128        | 128        | 2               |
| 256        | 64         | 4               |
| 512        | 32         | 8               |
| 1024       | 16         | 16              |
| 2048       | 8          | 32              |

### 4. Script: `src/ablation_context_length.py`

A standalone runnable script (`python src/ablation_context_length.py`). Key behaviors:

**Smoke test (automatic, runs first):** Before the sweep, run 1 epoch of (ModernBERT, max_length=128, seed=42) in bf16. If loss is NaN or does not decrease, abort with a clear error message.

**Crash recovery:** Before starting each (model, max_length, seed) run, check if a matching row already exists in the results CSV. If so, skip it. Re-running the script after a crash safely resumes from where it left off.

**Results CSV:** Write one row per completed run to `results/ablation_context_length.csv`. Columns:
- `model`, `max_length`, `seed`
- `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `auc_roc`
- `f1_class_0` … `f1_class_19` (per-class F1, 20 columns)
- `train_time_seconds`, `peak_mps_memory_gb`

**Model saving:** Save each trained checkpoint to `artifacts/ablation_context_length/{model_name}/len{max_length}_seed{seed}/`.

**Run order:** Complete the full BERT sweep first, then `del model; torch.mps.empty_cache()`, then the ModernBERT sweep.

### 5. Evaluation
After training each run, call `compute_metrics()` from `helpers.py` on the test set predictions. Log memory with `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()` after each run.

### 6. Plot script: `src/plot_context_length.py` (notebook-style `# %%`)

Reads `results/ablation_context_length.csv`. Generates:

- **Primary plot:** Macro F1 vs. max_length, two lines (BERT, ModernBERT) with mean ± std shading across 3 seeds. Annotate BERT's 512 ceiling. Mark current project baseline (max_length=50) with a vertical dashed line.
- **Secondary plot:** Training time vs. max_length to show compute cost of longer contexts.
- **Per-class heatmap:** Per-class F1 vs. max_length for each model (mean across seeds). Rows = 20 newsgroup categories, columns = max_length values.
- **Crossover annotation:** Identify and annotate the length at which ModernBERT's F1 gain over BERT exceeds 1 percentage point.

## Expected Outcome
ModernBERT should show a steeper F1 improvement as length increases, plateauing later than BERT. BERT likely plateaus around 256–512 tokens. The experiment validates whether the 50-token truncation in the current project masks ModernBERT's real advantage. Long-document-heavy categories (`sci.*`, `talk.*`) are expected to show larger per-class gains at higher lengths.

## Memory Considerations (M4 Pro, 48GB unified)

**Key fact**: on Apple Silicon, CPU and GPU share the same 48GB pool. There is no separate VRAM limit. The MPS backend allocates from this unified pool.

**Memory budget estimates** (bf16, gradient checkpointing on):

| max_length | batch | BERT total est. | ModernBERT total est. |
|------------|-------|-----------------|----------------------|
| 512        | 32    | ~4GB            | ~5GB                 |
| 2048       | 8     | N/A (>512 cap)  | ~6GB                 |

All values fit comfortably within 48GB. The OS and other processes consume ~4–6GB, so the practical budget is ~42GB.

- Log memory with `torch.mps.current_allocated_memory()` and `torch.mps.driver_allocated_memory()` after each run. Reset between runs with `torch.mps.empty_cache()`.
- At lengths ≥ 1,024 set `pin_memory=False` in `DataLoader` — MPS doesn't use pinned memory and enabling it wastes RAM.
- Use HuggingFace's lazy dataset loading and let `DataCollatorWithPadding` batch on the fly. Do not store the entire tokenized dataset in memory at once.

## Files to Create
- `src/ablation_context_length.py` — standalone sweep script
- `src/plot_context_length.py` — notebook-style `# %%` plot script
- `results/ablation_context_length.csv` — written by the sweep script
