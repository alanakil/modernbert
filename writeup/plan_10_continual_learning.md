# Project Plan: Continual Learning / Catastrophic Forgetting

## Goal
Measure how much BERT and ModernBERT forget previously learned categories when trained on new ones sequentially, and evaluate whether Elastic Weight Consolidation (EWC) mitigates forgetting differently for each model.

## Background
Standard fine-tuning assumes all training data is available at once. In practice, new categories or domains emerge over time and retraining from scratch is expensive. Continual learning (also called lifelong learning) studies how models can acquire new knowledge without overwriting old knowledge — the "catastrophic forgetting" problem.

ModernBERT's alternating local/global attention layers may create different forgetting dynamics than BERT's uniform architecture. Local attention layers learn more position-specific features and may be less prone to interference from new tasks. This is an open empirical question.

## Steps

### 1. Define the sequential task schedule
Split the 20 categories into 4 phases of 5 categories each. Group semantically coherent phases to make each task well-defined:

- **Phase 1** (tech hardware/software): comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x
- **Phase 2** (recreation): rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey, misc.forsale
- **Phase 3** (science): sci.crypt, sci.electronics, sci.med, sci.space, soc.religion.christian
- **Phase 4** (politics/religion): talk.politics.guns, talk.politics.mideast, talk.politics.misc, talk.religion.misc, alt.atheism

At each phase, only the 5 current-phase training examples are available. Evaluation always covers all seen-so-far categories.

### 2. Methods to compare

**Baseline — Naive sequential fine-tuning**: Simply fine-tune on each new phase's data, starting from the previous checkpoint. No forgetting mitigation.

**EWC (Elastic Weight Consolidation)**: After training on phase k, compute the Fisher information matrix (diagonal approximation) for the parameters. When training on phase k+1, add a penalty term that discourages large changes to parameters that were important for phase k. Penalty: sum_i F_i * (theta_i - theta_i^*)^2. Tune the EWC lambda coefficient in {0.1, 1, 10, 100}. Store Fisher and theta_star in float16 on CPU to halve memory — they don't need GPU until the penalty term is computed. For multi-phase EWC, accumulate Fisher online (running average) rather than storing a separate Fisher matrix per phase.

**Replay**: Keep a small buffer of 50 examples per past category and mix them into each new phase's training batch. Simplest approach but requires storing old data.

**Upper bound — Joint training**: Train on all 20 categories at once from the start. This is the ceiling that continual learning methods try to approach.

### 3. Metrics
After each phase, evaluate on all categories seen so far. Track:
- **Backward transfer (BWT)**: average drop in accuracy on old categories after training on new ones. Lower is better (less forgetting).
- **Forward transfer (FWT)**: accuracy on new categories at the end of their phase vs what you'd get training them in isolation. Positive FWT means prior phases helped.
- **Final average accuracy**: accuracy across all 20 categories after all 4 phases.

Formula from GEM paper:
- BWT = (1/(T-1)) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i}), where R_{j,i} is accuracy on task i after training on task j.
- FWT = (1/(T-1)) * sum_{i=2}^{T} (R_{i-1,i} - R_0_i), where R_0_i is the random-init accuracy on task i.

### 4. Analysis and plots
- Learning curve grid: one subplot per phase showing accuracy on all previously seen phases over training steps (reveals the forgetting event in real time).
- BWT bar chart: BERT vs ModernBERT, for naive / EWC / replay.
- Final accuracy matrix (T×T heatmap): row = phase after which evaluation was done, column = task being evaluated. Shows both forward and backward transfer.
- Parameter importance comparison: visualize the Fisher information magnitudes across layers for BERT vs ModernBERT. Are different layers considered "important" for each model?

### 5. Optional: per-layer forgetting analysis
After naive sequential training, compare each layer's weights before and after a new phase. Save a copy of each layer's weights as float16 to disk before the new phase begins, then compute L2 norm of the weight delta after training. Load and compare one layer at a time — do not hold snapshots of all layers in memory simultaneously. For ModernBERT, check whether local attention layers change less than global attention layers.

## Expected Outcome
Naive fine-tuning will show significant BWT for both models — accuracy on Phase 1 categories drops after Phase 4 training. EWC will reduce BWT at the cost of some final accuracy. ModernBERT may show different forgetting patterns due to its local/global attention alternation: local attention layers may act as more stable feature extractors, with forgetting concentrated in the global attention layers. Whether ModernBERT forgets more or less than BERT overall is the main empirical question.

## Memory Considerations (M4 Pro, 48GB unified)

**Memory budget**:
- Training per phase (bf16, gradient checkpointing, batch_size=64, max_length=256): ~4GB for BERT, ~5GB for ModernBERT.
- EWC overhead — `theta_star` (float16, CPU) + diagonal Fisher (float16, CPU):
  - BERT-base: 110M params × 2 bytes × 2 = ~440MB on CPU. Negligible.
  - ModernBERT-base: 149M params × 2 bytes × 2 = ~600MB on CPU. Still fine.
- Replay buffer: 1,000 stored texts. Bytes-level, insignificant.
- Total worst case (model on MPS + EWC state on CPU): ~5.5GB. Fits easily within 48GB.

- Set `device = torch.device("mps")`. Use bf16 (`bf16=True`) + gradient checkpointing for all training.
- `bitsandbytes` is CUDA-only and will not work on MPS. There is no memory pressure that would require 4-bit training on a 48GB machine — bf16 training is the right approach.
- Store `theta_star` and Fisher diagonal as float16 CPU tensors. Move to MPS only during the penalty computation step (`F.to("mps")`), compute, then immediately move the result back to CPU and free the MPS copy with `torch.mps.empty_cache()`.
- Accumulate Fisher across phases with a running average in-place rather than storing per-phase Fishers: `F_acc = (F_acc * n + F_phase) / (n + 1)`.
- Replay buffer: store raw text strings to disk, tokenize on the fly in the DataLoader collator. No pre-tokenized tensors kept in memory.
- For the per-layer forgetting analysis (step 5): save each layer's weight snapshot as a float16 `.pt` file before each phase, then load and diff one layer at a time. Do not hold all layer snapshots in memory at once.
- `torch.mps.empty_cache()` after each phase transition before starting the next training loop.

## Files to Create
- `src/continual_data.py` — splits dataset into sequential phases
- `src/train_continual.py` — sequential training with naive, EWC, and replay strategies
- `src/ewc.py` — EWC Fisher information computation and penalty
- `src/eval_continual.py` — BWT/FWT metrics and the accuracy matrix
- `src/plot_continual.py` — learning curves, BWT bar charts, forgetting heatmap
