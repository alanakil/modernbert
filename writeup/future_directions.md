# Future Directions

## 1. Context Length Ablation

The current project truncates all documents to 50 tokens — a severe bottleneck that almost certainly disadvantages ModernBERT. ModernBERT was pretrained at up to 8,192 tokens, while BERT tops out at 512.

Run a sweep over max_length values: 50, 128, 256, 512, 1024, 2048, 8192. For each value, fine-tune both models and record F1/accuracy. Plot performance as a function of context length. This will reveal:
- At what length does ModernBERT's advantage become significant?
- Does BERT plateau earlier (as expected at 512)?
- Is more context always better, or does noise hurt beyond a certain length?

This is the single most natural next experiment given what already exists in the codebase.

---

## 2. Low-Data Regime

BERT was pretrained on 3.3B words. ModernBERT was pretrained on 1.7T tokens — roughly 500x more data, with higher quality sources during annealing. This should translate into stronger representations that require less labeled data to fine-tune.

Design: subsample the training set to N examples per class for N in {5, 10, 25, 50, 100, 250, 500, full}. Fine-tune both models at each N and plot the accuracy/F1 curves. The crossover point — where ModernBERT's data efficiency advantage shrinks — is particularly informative. Also test variance across multiple seeds at low N since results are noisy.

---

## 3. Out-of-Distribution Generalization (Temporal Shift)

20 Newsgroups was collected from Usenet in 1993-1995. Language, slang, and writing conventions have changed significantly since then. ModernBERT's pretraining corpus is from 2024, which means it has seen much more contemporary text.

Design: find or scrape modern Reddit posts from subreddits that map to the 20 Newsgroups categories (e.g., r/linux → comp.os.linux, r/atheism → talk.religion.misc, r/hockey → rec.sport.hockey). Train both models on the original 20 Newsgroups training set and evaluate zero-shot on the Reddit posts. This measures robustness to a 30-year language shift and tests whether recency of pretraining data matters for domain generalization.

---

## 4. LoRA vs Full Fine-Tuning

Full fine-tuning updates all parameters. LoRA injects trainable low-rank matrices into attention layers and freezes the rest, dramatically reducing the number of trainable parameters. The optimal LoRA rank is model-dependent and tied to the intrinsic dimensionality of the task.

Design: benchmark both models under (a) full fine-tuning and (b) LoRA at ranks {4, 8, 16, 32, 64}. Track accuracy, F1, trainable parameter count, and training time. ModernBERT's structural differences from BERT — no bias terms, GeGLU activations, RoPE positional encodings — may shift which layers benefit most from LoRA and what rank is sufficient. This is practically relevant: for deployment you often want the smallest fine-tuned footprint that maintains accuracy.

---

## 5. Dense Retrieval / Semantic Search

Instead of training a classification head, use the models as text encoders: embed every document in the test set, then for each query embed it and retrieve the top-k nearest neighbors by cosine similarity. Assign the majority label among retrieved neighbors as the prediction.

This tests the quality of the raw contextual embeddings without any task-specific fine-tuning. It also allows benchmarking throughput and latency, where ModernBERT's use of flash attention and removal of padding tokens (packing) should show measurable speedups. Use FAISS for the nearest-neighbor index. Compare retrieval P@1, P@5, and MAP against the fine-tuned classification results from the main project.

---

## 6. Probing Layer Representations

At which layer does each model "know" what topic a document belongs to? Train a simple linear classifier (logistic regression or a 1-layer MLP) on the hidden states extracted from each transformer layer independently — without any end-to-end fine-tuning.

ModernBERT alternates between local attention (attending to a 128-token window) and global attention (full sequence) every third layer. This architectural choice should cause information about document-level topic to accumulate differently compared to BERT's uniform global attention across all layers. Expect to see ModernBERT encode global topic information later in the stack since local layers can only aggregate short-range context. Visualize the probing accuracy curve per layer for both models side-by-side.

---

## 7. Hierarchical Classification

The 20 Newsgroups dataset has an implicit two-level label hierarchy:

- `comp.*` (5 categories): graphics, os.ms-windows, sys.ibm.pc.hardware, sys.mac.hardware, windows.x
- `rec.*` (4 categories): autos, motorcycles, sport.baseball, sport.hockey
- `sci.*` (4 categories): crypt, electronics, med, space
- `talk.*` (4 categories): politics.guns, politics.mideast, politics.misc, religion.misc
- `misc.forsale` and `soc.religion.christian` (2 standalone)

Design a two-stage classifier: first predict the top-level group (6 groups), then within each group predict the fine-grained label. Compare this against the flat 20-class baseline in terms of accuracy and confusion patterns. The confusion matrix from the flat classifier should reveal whether the models confuse categories across groups (harder) or within groups (easier). If cross-group confusions dominate, the hierarchy won't help much; if within-group confusions dominate, the hierarchical model should win.

---

## 8. Cross-Dataset Zero-Shot Transfer

After fine-tuning on 20 Newsgroups, evaluate both models on other text classification benchmarks without any additional training. Good candidates:

- **AG News** (4 classes: World, Sports, Business, Sci/Tech) — coarser labels, overlapping topics
- **DBpedia** (14 classes: ontology categories) — more structured, less noisy
- **Reuters-21578** (multi-label news categories) — different genre, overlapping topics

This tests whether fine-tuning on 20 Newsgroups produces representations that generalize, or ones that overfit to Usenet-specific language patterns. ModernBERT's better pretraining should result in features that transfer more broadly. Track both zero-shot accuracy (no adaptation) and few-shot accuracy (fine-tune on 100 examples from the target dataset).

---

## 9. Adversarial Robustness

Real-world text contains typos, unusual spacing, and paraphrases that can fool models trained on clean corpora. BPE tokenization (ModernBERT) and WordPiece tokenization (BERT) handle character-level noise very differently: BPE may segment unknown spellings into sub-word units that resemble known tokens, while WordPiece may produce more `[UNK]` tokens.

Design perturbations at three levels:
- **Character-level**: inject random typos (swap, delete, insert characters), simulate keyboard proximity errors
- **Word-level**: replace content words with WordNet synonyms, shuffle word order within sentences
- **Semantic-level**: use an LLM to paraphrase each test document while preserving meaning

For each perturbation type, measure accuracy degradation from the clean baseline. Expect BPE to be more robust to character-level noise. The semantic-level perturbation will reveal which model's representations are more meaning-focused vs surface-form-focused.

---

## 10. Continual Learning / Catastrophic Forgetting

Instead of training on all 20 categories at once, train sequentially: first on categories 1-5, then 6-10, then 11-15, then 16-20. After each phase, evaluate on all previously seen categories. This simulates a real-world scenario where new topics appear over time and you can't retrain from scratch.

Measure:
- **Backward transfer**: how much does accuracy on old categories drop after training on new ones?
- **Forward transfer**: does training on early categories help with later ones (shared vocabulary, writing style)?

Compare naive sequential fine-tuning against Elastic Weight Consolidation (EWC), which penalizes changes to weights important for previous tasks. ModernBERT's different inductive biases (local vs global attention, RoPE) may produce fundamentally different forgetting dynamics compared to BERT's uniform architecture. This is an underexplored direction in the ModernBERT literature.
