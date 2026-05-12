# Project Plan: Probing Layer Representations

## Goal
Identify at which transformer layer each model encodes topic-level information, and determine whether ModernBERT's alternating local/global attention architecture creates a different information flow compared to BERT's uniform global attention.

## Background
A probing classifier is a lightweight model (logistic regression or 1-layer MLP) trained on frozen hidden states from a single layer to predict a label. If the probe achieves high accuracy, that layer encodes information relevant to the task. By running probes on every layer, we get a profile of where information is concentrated.

BERT has 12 layers, all using full (global) self-attention. Every layer can in principle attend to any token in the sequence. ModernBERT has 22 layers with a pattern: every third layer is global attention, the rest are local (attend only to a 128-token window). This means early ModernBERT layers are constrained to local context and can only aggregate global topic information through the repeated global attention layers. We expect topic-level information to build up in ModernBERT later and in a more step-wise pattern than in BERT.

## Steps

### 1. Extract hidden states
- Load both fine-tuned models (from existing artifacts in `../artifacts/`).
- For each document batch, run a forward pass with `output_hidden_states=True` under `torch.no_grad()`.
- Collect only the CLS token (index 0) from each layer's hidden state — discard the rest of the sequence immediately. This avoids storing full-sequence tensors.
- Process in batches of 64. After each batch, move the CLS states to CPU and cast to `float16` before accumulating.
- Save per-layer states to disk as separate files: `hidden_states_layer_{i}.npy` (float16). Do not hold all layers in memory simultaneously — for ModernBERT, storing all 22 layers at once would be ~1.1GB; layer-by-layer on disk avoids this entirely.
- Unload the model from GPU before training the probing classifiers.

### 2. Train probing classifiers
- For each layer, load only that layer's saved `.npy` file, train logistic regression (sklearn), evaluate, record the scalar metric, then discard the array.
- Use L2 regularization with cross-validated C to keep probes simple.
- Also test a 1-layer MLP (hidden_dim=128) to capture nonlinear structure if logistic regression undershoots. The MLP is trained on CPU with sklearn's `MLPClassifier` — no GPU needed.

### 3. Probing with frozen vs fine-tuned weights
Run probing on two versions of each model:
- **Pretrained only** (no fine-tuning): reveals what the pretraining encodes about topic.
- **Fine-tuned**: reveals how fine-tuning redistributes information across layers.

The difference between the two profiles shows which layers are most modified by fine-tuning.

### 4. Per-category probing
Train probes for binary classification (one-vs-rest) per category. Some categories may be encoded earlier (e.g., `sci.space` has distinctive vocabulary) while others may require deeper reasoning.

### 5. Analysis and plots
- Primary plot: probing accuracy vs layer index, two lines (BERT, ModernBERT). Annotate ModernBERT's global attention layers (layers 3, 6, 9, ...) with vertical dashed lines.
- Secondary plot: same but split by pretrained vs fine-tuned models. Show how fine-tuning shifts where information lives.
- Heatmap: (layer × category) probing accuracy for ModernBERT, showing per-category depth.

## Expected Outcome
BERT's probing accuracy should rise steadily through the layers. ModernBERT's accuracy curve should show step-wise jumps at global attention layers (3, 6, 9, ...) with plateaus in between, reflecting the local attention bottleneck. Fine-tuning likely causes both models to concentrate more information in later layers.

## Memory Considerations (M4 Pro, 48GB unified)

**Memory budget**:
- Extraction pass: BERT bf16 in inference mode: ~220MB weights. `output_hidden_states=True` at batch_size=64, seq_len=256 adds ~1.5GB of intermediate activations for all 12 layers. Under `torch.no_grad()` this is safe well within 48GB.
- ModernBERT: ~300MB weights + ~2.5GB activations for all 22 layers at batch_size=64. Still fine.
- Disk storage: ModernBERT, 22 layers, 18,532 docs, 768 dims, float16 → ~497MB. Store as `np.memmap` so it can be accessed without loading fully into RAM.

- The naive approach — accumulating all layers' tensors in GPU memory across the full dataset — is still feasible on 48GB but wasteful. Stream to disk in float16 after each batch. Cast from bfloat16 to float16 before saving: `hidden.to(torch.float16).cpu().numpy()`.
- After extraction, unload the model: `del model; torch.mps.empty_cache()`. Probing classifiers run on CPU via sklearn — the transformer is not needed.
- For the per-category probing (step 4): 20 × 22 = 440 logistic regression fits. Each loads ~22MB from the memmap, fits in seconds. Sequential is fine — no parallelism needed.
- **Quantization note**: for the pretrained-only probing variant, quantize the model with `optimum-quanto` (`qint8`) to speed up the extraction pass. This does not affect the hidden state values meaningfully for float16 storage.

## Files to Create
- `src/extract_hidden_states.py` — runs forward passes and saves CLS states per layer
- `src/probing_classifiers.py` — trains and evaluates probes for each layer
- `src/plot_probing.py` — probing accuracy curves and heatmaps
