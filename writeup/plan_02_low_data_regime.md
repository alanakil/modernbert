# Project Plan: Low-Data Regime

## Goal
Find the minimum labeled data per class needed to achieve competitive performance with each model, and quantify how much ModernBERT's richer pretraining translates into data efficiency.

## Background
ModernBERT was pretrained on 1.7T tokens vs BERT's 3.3B words (~roughly 500x more data), using higher-quality sources during the annealing phase. Better pretraining should yield stronger priors that require less labeled data to fine-tune. This is practically important: many real-world classification tasks don't have 11,000+ labeled training examples.

## Steps

### 1. Subsampling strategy
- Stratified sampling: for each N, sample exactly N examples per class, preserving class balance.
- N values: {5, 10, 25, 50, 100, 250, 500, full (~565 per class)}.
- Use 5 different random seeds for each N to estimate variance. Results at low N are noisy and a single seed is misleading.

### 2. Training setup
- Fix `max_length=512` (give both models a fair shot, within BERT's limit).
- Use bf16 mixed precision (`bf16=True` in `TrainingArguments`). Set `device = torch.device("mps")`.
- Because datasets are small at low N, batch size can equal the full dataset (e.g., batch_size=100 when N=5 per class × 20 classes = 100 examples). For large N use batch_size=64.
- Use early stopping (patience=3 on eval loss) to avoid overfitting. At N=5, the model can overfit within 1 epoch.
- Keep gradient checkpointing enabled. Even though small datasets mean fewer steps, the model weights themselves are the dominant memory cost.

### 3. Evaluation
- Report mean and std of macro F1 across 5 seeds at each N.
- Also record: number of epochs until best checkpoint (early stopping behavior).

### 4. Analysis and plots
- Primary plot: macro F1 vs N (log scale on x-axis), with error bands showing std across seeds.
- Annotate the crossover point where both models reach, say, 80% of their full-data F1.
- Secondary analysis: per-class F1 at low N. Some classes may be harder to learn from few examples (e.g., visually similar categories like `comp.sys.ibm.pc.hardware` vs `comp.sys.mac.hardware`).

### 5. Comparison baseline
- Add a TF-IDF + logistic regression baseline. This is a strong classical baseline for text classification and puts the neural results in context.

## Expected Outcome
ModernBERT should outperform BERT significantly at low N (5-50 examples per class) due to better pretrained representations. The gap should narrow as N grows toward the full dataset. The TF-IDF baseline may actually be competitive at very low N where neural models overfit.

## Memory Considerations (M4 Pro, 48GB unified)

**Memory budget**: each training run uses ~4GB (BERT bf16) or ~5GB (ModernBERT bf16) with gradient checkpointing at max_length=512. The 80-run loop poses no memory risk as long as models are unloaded between runs.

- Never load both models simultaneously. After each run: `del model, trainer; torch.mps.empty_cache()`.
- The TF-IDF baseline is CPU-only. Set `max_features=50_000` in `TfidfVectorizer` and keep the matrix sparse throughout — do not call `.toarray()`.
- For the 5-seed loop, store only scalars. Do not accumulate model objects across seeds.
- **Quantization note**: `bitsandbytes` (4-bit / 8-bit training) is CUDA-only and will not run on MPS. For inference-only quantization after training, use `optimum-quanto` with `qint8`. For training, bf16 is the right tool on the M4.
- With 48GB unified memory, there is no need for any additional memory tricks beyond bf16 and gradient checkpointing. The M4 Pro's large memory pool is the main advantage here over a typical 16-24GB GPU setup.

## Files to Create
- `src/low_data_experiment.py` — subsampling + training loop over N and seeds
- `src/baselines.py` — TF-IDF + logistic regression baseline
- `src/plot_low_data.py` — learning curve plots
