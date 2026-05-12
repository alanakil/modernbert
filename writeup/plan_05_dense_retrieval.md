# Project Plan: Dense Retrieval / Semantic Search

## Goal
Reframe 20 Newsgroups as a retrieval problem: embed all documents and classify via nearest-neighbor lookup. This tests raw embedding quality without a task-specific head, and benchmarks throughput where ModernBERT's flash attention provides real-world latency gains.

## Background
Fine-tuning with a classification head tells us how well the model adapts to a task. Dense retrieval tells us how good the raw (or lightly trained) representations are for capturing semantic similarity. ModernBERT's removal of padding tokens (sequence packing) and use of flash attention should make it significantly faster at embedding large document collections. This direction also opens the door to production-style search systems where you don't retrain for every new category.

## Steps

### 1. Embedding strategy — three variants to compare
- **Frozen embeddings**: Use the pretrained model without any fine-tuning. Take the CLS token as the document embedding.
- **Fine-tuned CLS**: Use the models already fine-tuned in the main project. Take the CLS token from the classification model's encoder.
- **Contrastive (SimCSE-style)**: Fine-tune with a contrastive objective on (document, same-label document) pairs to make embeddings for same-class documents cluster together.

### 2. Index construction
- Use FAISS (Facebook AI Similarity Search) with a flat L2 or cosine index.
- Embed the entire training set (~11,000 documents) as the retrieval database. Save embeddings to disk as float16 numpy arrays (`np.float16`) — 11,000 × 768 × 2 bytes ≈ 16MB per model, trivial.
- Load embeddings from disk to build the FAISS index; do not keep the transformer model in memory while running FAISS queries.
- At query time, embed a test document and retrieve the top-k most similar training documents.

### 3. Classification from retrieval
- **k-NN voting**: assign the majority label among the top-k retrieved documents.
- Test k ∈ {1, 3, 5, 10, 20}.
- Report accuracy and macro F1 for each k.

### 4. Throughput benchmark
- Measure time to embed the entire training set (batch embedding).
- Measure per-query latency (embedding + FAISS search).
- Run on MPS (GPU) and CPU separately — on Apple Silicon both share the same physical memory, but MPS has higher compute throughput.
- Compare BERT vs ModernBERT throughput in documents/second.
- **Note on flash attention**: the official `flash-attn` package requires CUDA and will not install on macOS/MPS. ModernBERT on MPS falls back to PyTorch's `scaled_dot_product_attention` (SDPA), which also uses a memory-efficient attention kernel. The throughput advantage of ModernBERT is still real, just via SDPA rather than flash attention.
- Also benchmark `optimum-quanto` int8-quantized models at inference: `from optimum.quanto import quantize, qint8; quantize(model, weights=qint8)`. Expect ~2x throughput gain at the cost of minor accuracy.

### 5. Qualitative analysis
- For a sample of misclassified test documents, inspect which training documents were retrieved. Do the retrieved neighbors make intuitive sense even when the top label is wrong?
- Visualize embeddings with UMAP: project a random subsample of 2,000 test documents to 2D colored by label (not the full test set — UMAP on 7,500 × 768 is slow and memory-heavy). Load embeddings from disk for this step; the model need not be in memory.

### 6. Comparison table
Summarize all results: frozen CLS k-NN vs fine-tuned CLS k-NN vs contrastive k-NN vs the classification head baseline from the main project.

## Expected Outcome
Frozen embeddings will underperform fine-tuned classification, but the gap should be smaller for ModernBERT due to its richer pretraining. Contrastive fine-tuning should close the gap significantly. ModernBERT should be 1.5-3x faster than BERT on embedding throughput due to flash attention and packing.

## Memory Considerations (M4 Pro, 48GB unified)

**Memory budget**:
- BERT-base bf16 inference: ~220MB model weights + ~300MB activations at batch_size=128 → ~520MB total. Trivial.
- ModernBERT-base bf16 inference: ~300MB weights + ~400MB activations → ~700MB total.
- FAISS flat index for 11,000 × 768 float16 vectors: ~16MB. Negligible.
- int8 quantized models (via `optimum-quanto`): half the weight footprint — ~110MB and ~150MB respectively. Recommended for the throughput benchmark since inference quality is the same.

- The pipeline is naturally staged: embed → save to disk → `del model; torch.mps.empty_cache()` → build FAISS index → query. Never hold the transformer and the FAISS index simultaneously (though at these sizes it would be fine anyway).
- For contrastive training, use in-batch negatives to avoid a large negative queue.
- Set `device = torch.device("mps")`. FAISS runs on CPU — this is fine and expected.
- `flash-attn` is CUDA-only. ModernBERT on MPS will use PyTorch SDPA automatically — no code change needed, just don't explicitly install `flash-attn`.

## Files to Create
- `src/embed_documents.py` — batch embedding with both models, saves to disk
- `src/retrieval_eval.py` — FAISS index construction and k-NN evaluation
- `src/contrastive_train.py` — SimCSE-style contrastive fine-tuning
- `src/throughput_benchmark.py` — timing benchmark for embedding speed
- `src/plot_retrieval.py` — UMAP visualization and results table
