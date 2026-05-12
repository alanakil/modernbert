# Project Plan: Hierarchical Classification

## Goal
Exploit the natural two-level label hierarchy in 20 Newsgroups to build a structured classifier, and determine whether enforcing the hierarchy improves accuracy and reduces cross-group confusions.

## Background
20 Newsgroups has a prefix-based hierarchy:

| Top-level group  | Sub-categories (fine-grained labels)                                                    |
|------------------|-----------------------------------------------------------------------------------------|
| comp (5)         | graphics, os.ms-windows.misc, sys.ibm.pc.hardware, sys.mac.hardware, windows.x         |
| rec (4)          | autos, motorcycles, sport.baseball, sport.hockey                                        |
| sci (4)          | crypt, electronics, med, space                                                          |
| talk (4)         | politics.guns, politics.mideast, politics.misc, religion.misc                           |
| misc (1)         | forsale                                                                                 |
| soc (1)          | religion.christian                                                                      |
| alt (1)          | atheism                                                                                 |

The flat 20-class confusion matrix from the existing project reveals two types of errors: within-group confusions (e.g., `comp.graphics` vs `comp.windows.x`) and cross-group confusions (e.g., `sci.med` vs `talk.religion.misc`). If within-group confusions dominate, a hierarchical classifier can help by first narrowing to the right group before making the fine-grained call.

## Steps

### 1. Analyze existing confusion matrix
- Load the confusion matrices saved from `eval_compare_models.py`.
- Compute the fraction of errors that are within-group vs cross-group.
- This motivates whether hierarchical classification will help at all.

### 2. Build the two-stage classifier

**Stage 1 — Group classifier**: 7-class classification (comp, rec, sci, talk, misc, soc, alt). Assign group labels from the existing label_text field using string prefix matching. Fine-tune both BERT and ModernBERT on this 7-class task.

**Stage 2 — Fine-grained classifier within each group**: For each of the 4 multi-class groups (comp, rec, sci, talk), fine-tune a separate classifier on only the documents belonging to that group. The single-category groups (misc, soc, alt) don't need a stage-2 classifier.

**Inference**: for a new document, run stage 1 on the full test set and save predictions to disk. Unload the stage-1 model. Load each stage-2 model only for its group's subset, run inference, then unload before loading the next stage-2 model. Never hold more than one model in GPU memory at a time.

### 3. Alternative: multi-task / shared head
Instead of cascading two separate models, train a single model with two output heads sharing the same encoder: one head predicts the 7-class group, the other predicts the 20-class fine-grained label. Use a multi-task loss: L = L_group + lambda * L_fine. Tune lambda in {0.1, 0.5, 1.0, 2.0}.

### 4. Evaluation
- Report accuracy and macro F1 for both the flat baseline and the hierarchical model.
- Compute per-category F1 and highlight categories that improved most.
- Report error breakdown: within-group errors vs cross-group errors for each approach.
- For the two-stage model, also report stage-1 group accuracy separately to isolate where errors originate.

### 5. Analysis and plots
- Confusion matrix comparison: flat vs hierarchical (normalized, 20×20).
- Bar chart: per-category F1 delta (hierarchical minus flat) — shows which categories benefit.
- For the multi-task model: plot training curves for both loss components.

## Expected Outcome
If within-group confusions are the dominant error type (likely for `comp.*` categories with very similar content), the hierarchical model will improve F1 by reducing impossible cross-group confusions. ModernBERT's stronger representations may already reduce group-level errors enough that the hierarchy adds less benefit for it than for BERT.

## Memory Considerations (M4 Pro, 48GB unified)

**Memory budget**:
- Each fine-tuned model (BERT or ModernBERT, bf16, gradient checkpointing, batch_size=64, max_length=256): ~4-5GB during training.
- Two-stage: up to 5 training runs per base model sequentially — no memory issue since each is run and unloaded in turn.
- Multi-task (shared encoder, two heads): identical memory to a single fine-tuning run. The extra classification head adds < 1MB. Recommended.

- Set `device = torch.device("mps")` and use bf16 + gradient checkpointing for all runs.
- `bitsandbytes` is CUDA-only. Do not use it. bf16 training via `TrainingArguments(bf16=True)` is sufficient on M4.
- For inference only (e.g., running the trained stage-1 classifier to generate group predictions), quantize with `optimum-quanto` (`qint8`) to halve weight memory and speed up MPS inference.
- Train each model, save to disk, `del model; torch.mps.empty_cache()`, then proceed to the next. Never hold stage-1 and stage-2 models in memory simultaneously.
- During the confusion matrix comparison, load only numpy prediction arrays from disk — no model in memory needed for plotting.

## Files to Create
- `src/hierarchical_labels.py` — adds group-level labels to the dataset
- `src/train_hierarchical.py` — two-stage and multi-task training
- `src/eval_hierarchical.py` — hierarchical inference and evaluation
- `src/plot_hierarchical.py` — confusion matrix comparison and per-category deltas
