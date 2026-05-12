# Project Plan: Adversarial Robustness

## Goal
Measure how sensitive BERT and ModernBERT are to text perturbations at the character, word, and semantic level, and determine whether BPE (ModernBERT) vs WordPiece (BERT) tokenization confers different robustness characteristics.

## Background
BERT uses WordPiece tokenization, which segments tokens greedily to maximize vocabulary coverage. ModernBERT uses BPE (same as OLMo), which builds a vocabulary based on byte-pair merge frequency. When a word is misspelled or novel, BPE tends to segment it into subword units that still resemble known tokens, while WordPiece may produce more `[UNK]` tokens or unusual segmentations.

Real-world classification systems face text with typos, informal spelling, paraphrasing, and adversarial manipulation. A model that degrades sharply under small perturbations is less reliable in production.

## Steps

### 1. Perturbation types

**Level 1 — Character-level (minimal semantic change)**
- Random character swap: swap two adjacent characters in 10% of words (e.g., "politics" → "poltiics").
- Random character deletion: delete one character from 10% of words.
- Keyboard proximity substitution: replace a character with an adjacent key on a QWERTY keyboard (e.g., 'e' → 'r' or 'w') in 10% of words.
- Implement these as deterministic functions with a fixed seed so results are reproducible.

**Level 2 — Word-level (moderate semantic change)**
- Synonym replacement: replace 20% of non-stopword content words with a WordNet synonym.
- Random word shuffle: randomly permute words within each sentence (destroys syntax, tests bag-of-words robustness).
- Random word deletion: delete 10% of words uniformly.

**Level 3 — Semantic-level (paraphrasing)**
- Use an LLM (e.g., Claude via API) to paraphrase each test document while preserving meaning. Prompt: "Rephrase the following text in different words while preserving its meaning and topic."
- This is expensive; apply to a random 500-example subset of the test set.
- This is the strongest test: if a model is sensitive to surface form, paraphrasing will expose it.

### 2. Evaluation
For each perturbation type and severity level:
- Pre-generate all perturbations and save to disk as HuggingFace Datasets before any model is loaded. This decouples dataset generation from model inference.
- Load one model, iterate over all perturbation variants (the model stays loaded, the datasets are swapped), record all scalars, then unload the model before loading the next.
- Compute robustness score: F1_perturbed / F1_clean.
- Use `torch.no_grad()` + `model.eval()` and batch_size=128 for all inference passes.

### 3. Tokenization analysis
For character-level perturbations, log how tokenization changes using only the tokenizers (no model weights needed — tokenizers are CPU-only and tiny in memory):
- Count the number of tokens produced by each tokenizer before and after perturbation.
- Track how often `[UNK]` appears (should be rare with BPE).
- Show example tokenizations for misspelled words: BPE vs WordPiece segmentation.

### 4. Per-category robustness
Some categories may be more robust than others. Categories with highly distinctive vocabulary (e.g., `sci.space` with words like "NASA", "orbit", "shuttle") may degrade differently under synonym replacement than categories with common vocabulary (e.g., `talk.politics.misc`).

### 5. Analysis and plots
- Robustness bar chart: F1 at each perturbation type, BERT vs ModernBERT side-by-side.
- Degradation heatmap: (perturbation type × category) showing per-category F1 drop.
- Tokenization example table: 5-10 perturbed words with their BERT vs ModernBERT tokenizations.
- For LLM paraphrasing: scatter plot of original score vs paraphrased score per document, colored by correct/incorrect.

## Expected Outcome
BPE (ModernBERT) should be more robust to character-level noise because it falls back to subword units more gracefully than WordPiece. Both models will degrade under semantic paraphrasing, but the model with better pretrained semantic representations (expected: ModernBERT) should degrade less. Random word shuffle is a stress test of whether the model uses positional/syntactic cues — ModernBERT's RoPE may behave differently from BERT's absolute PE here.

## Memory Considerations (M4 Pro, 48GB unified)

**Memory budget**:
- Perturbed datasets: 6 perturbation types × 7,532 test docs × ~500 chars avg = ~22MB of text on disk. Negligible.
- Inference per run: BERT/ModernBERT bf16, batch_size=128, max_length=256 → ~1GB under `torch.no_grad()`. Trivial.

- Perturbation generation is CPU-only. Do it once, save all variants to disk (HuggingFace Arrow format) before loading any model.
- Set `device = torch.device("mps")`. Use `torch.no_grad()` + `model.eval()` for all inference.
- **Quantization for inference**: apply `optimum-quanto` int8 quantization to both models before the inference sweep. This halves weight memory (220MB → 110MB for BERT) and improves MPS throughput. Since we only need logits (not gradients), quantization is lossless in practice for robustness comparison.
- Load one model, iterate over all 6 perturbation variants (swap datasets, keep model loaded), record all scalars, then `del model; torch.mps.empty_cache()` before loading the next.
- `bitsandbytes` is CUDA-only and will not work on MPS — do not use it.
- LLM paraphrasing (step 1, level 3): stream API responses one at a time and append to file. Do not buffer.
- Load WordNet once before the synonym loop, not inside it.

## Files to Create
- `src/perturbations.py` — all perturbation functions (character, word, semantic)
- `src/llm_paraphrase.py` — LLM-based paraphrasing using Claude API
- `src/eval_robustness.py` — applies perturbations and evaluates both models
- `src/plot_robustness.py` — robustness bar charts, heatmaps, tokenization examples
