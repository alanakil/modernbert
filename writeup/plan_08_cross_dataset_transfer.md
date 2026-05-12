# Project Plan: Cross-Dataset Zero-Shot Transfer

## Goal
Evaluate how well representations learned on 20 Newsgroups transfer to other text classification benchmarks without any additional fine-tuning, and determine whether ModernBERT produces more transferable features than BERT.

## Background
A model that truly understands language should produce features that generalize beyond the specific dataset it was fine-tuned on. Transfer quality is a function of both pretraining quality and fine-tuning dynamics — a model fine-tuned too aggressively may overfit to Usenet-specific surface features (e.g., email headers, quoting conventions) rather than underlying topic semantics.

Target benchmarks with semantic overlap to 20 Newsgroups:

| Dataset       | Classes | Description                                      | Overlap with 20NG          |
|---------------|---------|--------------------------------------------------|----------------------------|
| AG News       | 4       | World, Sports, Business, Sci/Tech                | Sports, Science, Tech      |
| DBpedia       | 14      | Ontology-based Wikipedia categories              | Partial (Company, Athlete) |
| Reuters-21578 | ~10     | News wire categories (multi-label, use top-10)   | Politics, Science          |
| OHSUMED       | 23      | Medical journal abstracts                        | sci.med                    |

## Steps

### 1. Zero-shot transfer evaluation
- Load both fine-tuned models from `../artifacts/trained_bert` and `../artifacts/trained_modernbert`.
- For each target dataset, define a label mapping from the fine-tuned 20 Newsgroups classes to the target classes. For example, for AG News "Sports" → labels from `rec.sport.*`, "Sci/Tech" → labels from `sci.*` and `comp.*`.
- The mapping won't be perfect — treat this as a coarse semantic alignment.
- Score: for each target test example, take the prediction from the 20NG model and map it to the target label space. Compute accuracy on the mapped labels.

### 2. Few-shot transfer evaluation
This is a cleaner setup than the coarse label mapping above. For each target dataset:
- Fine-tune the existing checkpoints (starting from the 20NG fine-tuned weights, not from pretrained) on N=100 examples per class from the target dataset.
- Also fine-tune from scratch (starting from the pretrained weights) on the same 100 examples.
- Compare: (a) pretrained → target, (b) pretrained → 20NG → target, (c) pretrained → target directly.

The question is whether the 20NG fine-tuning helps or hurts transfer. If it helps, the model learned generalizable topic features. If it hurts, it overfit to Usenet-specific patterns.

### 3. Intermediate representation analysis
Extract CLS embeddings from the fine-tuned models for both 20NG test documents and target dataset test documents. Save embeddings to disk as float16. Unload the model before running UMAP — the transformer is not needed at visualization time. For UMAP, subsample to 2,000 points per dataset to keep memory manageable.

### 4. Evaluation metrics
- Zero-shot: accuracy with label mapping (report the mapping clearly).
- Few-shot: accuracy and macro F1 at N=100, averaged over 3 seeds.
- Relative transfer gain: F1(pretrained → 20NG → target) / F1(pretrained → target directly).

### 5. Analysis and plots
- Bar chart: zero-shot accuracy per target dataset, BERT vs ModernBERT.
- Line chart: few-shot F1 vs N (N ∈ {10, 50, 100, 250}), showing the learning curve from each initialization.
- UMAP plots: one per target dataset, coloring points by source (20NG vs target).

## Expected Outcome
ModernBERT's features should transfer better due to more diverse and contemporary pretraining. The 20NG fine-tuning step is a double-edged sword: it helps for semantically close targets (AG News) but may hurt for distant ones (DBpedia). BERT fine-tuned on 20NG may overfit more to Usenet syntax.

## Memory Considerations (M4 Pro, 48GB unified)

**Memory budget**:
- Few-shot fine-tuning (N=100, max_length=256, batch_size=32, bf16): ~3GB per run. Trivially fits.
- Embedding extraction (all 4 datasets, ~50K docs total): BERT bf16 inference, batch_size=128 → ~1GB. Save float16 embeddings to disk (~150MB total across all datasets and models).
- UMAP on 2,000 points × 768 dims: CPU-only, ~100MB RAM. Negligible.

- Set `device = torch.device("mps")`. Use bf16 + gradient checkpointing for all training.
- `bitsandbytes` is CUDA-only. For inference-only quantization, use `optimum-quanto` (`qint8`): `quantize(model, weights=qint8)` before embedding extraction passes. This halves weight memory and accelerates MPS throughput.
- Structure the outer loop as: load model → process all target datasets → unload → next model. Avoids reloading the 440MB base weights for each of the 4 datasets separately.
- All embedding extraction uses `torch.no_grad()` + `model.eval()`, streaming float16 arrays to disk. UMAP and plotting load from disk with no transformer in memory.
- `torch.mps.empty_cache()` between model loads.

## Files to Create
- `src/load_target_datasets.py` — loads and preprocesses AG News, DBpedia, Reuters, OHSUMED
- `src/transfer_eval.py` — zero-shot and few-shot transfer evaluation
- `src/plot_transfer.py` — bar charts, learning curves, UMAP visualizations
