"""Context-length ablation sweep.

Runs both BERT and ModernBERT at max_length ∈ [50, 128, 256, 512, 1024, 2048]
with 3 seeds each. Results are written to results/ablation_context_length.csv.
Re-running the script after a crash resumes from where it left off.

Usage:
    python src/ablation_context_length.py
"""

# %%
import csv
import os
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

import helpers

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = {
    "bert": "google-bert/bert-base-cased",
    "modernbert": "answerdotai/ModernBERT-base",
}
BERT_MAX_LENGTH = 512  # BERT's positional embedding hard cap

LENGTHS = [50, 128, 256, 512]
SEEDS = [42]  # add 422, 1337 for full run

# Fraction of training data to use (1.0 = full dataset, 0.2 = 20% stratified)
TRAIN_SUBSAMPLE = 0.2

# Effective batch size = 256; adjust per-device batch + grad-accum accordingly
BATCH_CONFIG = {
    50:   {"batch_size": 256, "grad_accum": 1},
    128:  {"batch_size": 128, "grad_accum": 2},
    256:  {"batch_size": 64,  "grad_accum": 4},
    512:  {"batch_size": 32,  "grad_accum": 8},
    1024: {"batch_size": 16,  "grad_accum": 16},
    2048: {"batch_size": 8,   "grad_accum": 32},
}

LR = 1e-5
EPOCHS = 10
WEIGHT_DECAY = 0.01

RESULTS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "results", "ablation_context_length.csv"
)
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "ablation_context_length"
)

CSV_COLUMNS = (
    ["model", "max_length", "seed",
     "accuracy", "macro_precision", "macro_recall", "macro_f1", "auc_roc"]
    + [f"f1_class_{i}" for i in range(20)]
    + ["train_time_seconds", "peak_mps_memory_gb"]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def subsample_dataset(dataset, fraction, seed=42):
    """Return a stratified subsample of a HuggingFace dataset split."""
    if fraction >= 1.0:
        return dataset
    n = int(len(dataset) * fraction)
    rng = np.random.default_rng(seed)
    labels = np.array(dataset["label"])
    classes = np.unique(labels)
    indices = []
    n_per_class = max(1, n // len(classes))
    for c in classes:
        class_idx = np.where(labels == c)[0]
        chosen = rng.choice(class_idx, size=min(n_per_class, len(class_idx)), replace=False)
        indices.extend(chosen.tolist())
    # top up to exactly n if rounding left us short
    remaining = list(set(range(len(dataset))) - set(indices))
    if len(indices) < n:
        extra = rng.choice(remaining, size=n - len(indices), replace=False)
        indices.extend(extra.tolist())
    indices = sorted(indices[:n])
    return dataset.select(indices)


def load_completed_runs(csv_path):
    """Return a set of (model, max_length, seed) tuples already in the CSV."""
    if not os.path.exists(csv_path):
        return set()
    completed = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["model"], int(row["max_length"]), int(row["seed"])))
    return completed


def append_row(csv_path, row: dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def mps_memory_gb():
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


def clear_mps():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def smoke_test(device):
    print("\n=== Smoke test: ModernBERT, max_length=128, seed=42, 1 epoch ===")
    set_seed(42)
    newsgroups = load_dataset("SetFit/20_newsgroups")
    num_labels = len(set(newsgroups["train"]["label"]))
    label_names = sorted(set(newsgroups["train"]["label_text"]))

    model_id = MODELS["modernbert"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    encoded = newsgroups.map(tok, batched=True, remove_columns=["text", "label_text"])
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    model.gradient_checkpointing_enable()
    model.to(device)

    args = TrainingArguments(
        output_dir="/tmp/smoke_test",
        num_train_epochs=1,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=2,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        data_collator=collator,
    )
    result = trainer.train()
    final_loss = result.training_loss
    if np.isnan(final_loss):
        print("ABORT: smoke test produced NaN loss.")
        sys.exit(1)

    log_history = trainer.state.log_history
    losses = [e["loss"] for e in log_history if "loss" in e]
    if len(losses) >= 2 and losses[-1] >= losses[0]:
        print(f"ABORT: loss did not decrease ({losses[0]:.4f} → {losses[-1]:.4f}).")
        sys.exit(1)

    print(f"Smoke test passed. Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    del model
    clear_mps()


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------
def run_one(model_key, model_id, max_length, seed, device, newsgroups, num_labels, label_names):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pin_memory = max_length < 1024

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    encoded = newsgroups.map(tok, batched=True, remove_columns=["text", "label_text"])
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    cfg = BATCH_CONFIG[max_length]
    checkpoint_dir = os.path.join(
        ARTIFACTS_DIR, model_key, f"len{max_length}_seed{seed}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_pin_memory=pin_memory,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    model.gradient_checkpointing_enable()
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # Save final checkpoint
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Evaluate
    pred_output = trainer.predict(encoded["test"])
    logits = pred_output.predictions
    true_labels = pred_output.label_ids

    metrics = helpers.compute_metrics(logits, true_labels, label_names)
    peak_mem = mps_memory_gb()

    del model
    clear_mps()

    row = {
        "model": model_key,
        "max_length": max_length,
        "seed": seed,
        "train_time_seconds": round(train_time, 1),
        "peak_mps_memory_gb": round(peak_mem, 3),
        **metrics,
    }
    return row


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    device = helpers.identify_device()
    smoke_test(device)

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    completed = load_completed_runs(RESULTS_CSV)
    print(f"\nAlready completed runs: {len(completed)}")

    newsgroups = load_dataset("SetFit/20_newsgroups")
    num_labels = len(set(newsgroups["train"]["label"]))
    label_names = sorted(set(newsgroups["train"]["label_text"]))
    if TRAIN_SUBSAMPLE < 1.0:
        n_before = len(newsgroups["train"])
        newsgroups["train"] = subsample_dataset(newsgroups["train"], TRAIN_SUBSAMPLE)
        print(f"Dataset loaded. {num_labels} classes. "
              f"Train subsampled: {n_before} → {len(newsgroups['train'])} "
              f"({TRAIN_SUBSAMPLE:.0%}, stratified)")
    else:
        print(f"Dataset loaded. {num_labels} classes. Train size: {len(newsgroups['train'])}")

    sweep = [
        ("bert",       MODELS["bert"],       length, seed)
        for length in LENGTHS if length <= BERT_MAX_LENGTH
        for seed in SEEDS
    ] + [
        ("modernbert", MODELS["modernbert"], length, seed)
        for length in LENGTHS
        for seed in SEEDS
    ]

    total = len(sweep)
    sweep_t0 = time.time()
    for idx, (model_key, model_id, max_length, seed) in enumerate(sweep, 1):
        tag = (model_key, max_length, seed)
        if tag in completed:
            print(f"[{idx}/{total}] SKIP  {model_key:12s} len={max_length:4d}  seed={seed} (already done)")
            continue

        cfg = BATCH_CONFIG[max_length]
        elapsed = time.time() - sweep_t0
        print(
            f"\n{'='*70}\n"
            f"[{idx}/{total}] START  {model_key}  len={max_length}  seed={seed}\n"
            f"  batch={cfg['batch_size']}  grad_accum={cfg['grad_accum']}  "
            f"effective_batch=256  elapsed={elapsed/60:.1f}min\n"
            f"{'='*70}"
        )
        row = run_one(model_key, model_id, max_length, seed, device, newsgroups, num_labels, label_names)
        append_row(RESULTS_CSV, row)
        completed.add(tag)
        print(
            f"[{idx}/{total}] DONE   {model_key}  len={max_length}  seed={seed}  "
            f"macro_f1={row['macro_f1']:.4f}  "
            f"time={row['train_time_seconds']:.0f}s  "
            f"mem={row['peak_mps_memory_gb']:.2f}GB"
        )

    print(f"\nSweep complete. Results written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
