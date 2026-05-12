# Project Plan: Out-of-Distribution Generalization (Temporal Shift)

## Goal
Measure how well models trained on 1993-1995 Usenet posts generalize to modern text from the same topics, and whether ModernBERT's more recent pretraining data gives it an advantage under temporal distribution shift.

## Background
20 Newsgroups was collected from Usenet in 1993-1995. Writing style, vocabulary, slang, and topic framing have shifted significantly over 30 years. ModernBERT was pretrained on a 2024 corpus, so it has internalized contemporary language patterns. BERT's pretraining data is from ~2017 (Wikipedia + BooksCorpus). Neither model has seen 1990s Usenet language during pretraining, but ModernBERT's representations may be more robust because modern internet text is more diverse.

## Steps

### 1. Build the modern test set
Map 20 Newsgroups categories to Reddit subreddits (or other modern sources):

| Newsgroups category         | Reddit subreddit(s)                  |
|-----------------------------|--------------------------------------|
| comp.graphics               | r/graphic_design, r/blender          |
| comp.os.ms-windows.*        | r/windows                            |
| comp.sys.ibm.pc.hardware    | r/buildapc                           |
| comp.sys.mac.hardware       | r/mac                                |
| rec.autos                   | r/cars                               |
| rec.motorcycles             | r/motorcycles                        |
| rec.sport.baseball          | r/baseball                           |
| rec.sport.hockey            | r/hockey                             |
| sci.crypt                   | r/crypto, r/netsec                   |
| sci.electronics             | r/electronics                        |
| sci.med                     | r/medicine, r/askdocs                |
| sci.space                   | r/space                              |
| talk.politics.guns          | r/guns                               |
| talk.politics.misc          | r/politics                           |
| talk.religion.misc          | r/religion                           |
| soc.religion.christian      | r/christianity                       |
| misc.forsale                | r/hardwareswap, r/appleswap          |

Use the Pushshift Reddit dataset or the Reddit API to collect ~500 posts per category from 2022-2024. Filter to posts with at least 50 words.

### 2. Training
- Fine-tune both models on the full 20 Newsgroups training set (standard setup from `train.py`).
- Do NOT fine-tune on any Reddit data.
- Use bf16 mixed precision (`bf16=True`) and gradient checkpointing. Set `device = torch.device("mps")`.

### 3. Evaluation
- Evaluate the fine-tuned models on: (a) the original 20 Newsgroups test set, (b) the Reddit test set.
- Report accuracy and macro F1 for both.
- Compute the OOD degradation: delta_F1 = F1_newsgroups - F1_reddit.

### 4. Analysis
- Which categories degrade most? Expect categories where vocabulary has changed significantly (e.g., `comp.*`, `sci.crypt`) to show larger drops.
- Does ModernBERT degrade less than BERT overall? Does it hold up better on categories with significant vocabulary shift?
- Confusion matrix comparison between in-distribution and OOD settings.

### 5. Optional: fine-tuning on Reddit
As a sanity check, fine-tune both models on the Reddit data and evaluate on Reddit. This upper-bounds the OOD performance and confirms the Reddit labels are clean.

## Expected Outcome
Both models will degrade on Reddit, but ModernBERT should degrade less due to its more contemporary pretraining. Categories with heavy jargon evolution (cryptography, computing hardware) will show the largest shifts.

## Memory Considerations (M4 Pro, 48GB unified)

**Memory budget**: inference-only passes are very lean. BERT-base in bf16 = ~220MB weights + activations. At batch_size=128, seq_len=256: activations ~500MB under `torch.no_grad()`. Total per model ~1GB — trivial on 48GB.

- Collect and preprocess the Reddit dataset once, save to disk as a HuggingFace `Dataset` (Arrow format), and load with `load_from_disk`. Do not hold raw scraped data in memory alongside the processed version.
- At evaluation time, load one model, run inference on both test sets, record scalars, then `del model; torch.mps.empty_cache()` before loading the second.
- Use `torch.no_grad()` + `model.eval()` during all inference passes — disables gradient buffers and roughly halves activation memory.
- **Quantization for inference**: quantize both models with `optimum-quanto` (`qint8`) before inference. This halves weight memory (220MB → 110MB for BERT) and speeds up MPS inference. The training checkpoint stays in bf16; quantize a separate copy for eval. `bitsandbytes` is CUDA-only and will not work here.
- If the optional Reddit fine-tuning step is run, launch it as a separate process so the Python runtime starts clean — avoids fragmented MPS memory from the prior training run.

## Files to Create
- `src/collect_reddit_data.py` — Reddit data collection and preprocessing
- `src/eval_ood.py` — evaluation on the Reddit test set using trained checkpoints
- `src/plot_ood.py` — bar chart of per-category F1 degradation
