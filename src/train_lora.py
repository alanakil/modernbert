"""LoRA vs Full Fine-Tuning sweep.

Compares parameter-efficient fine-tuning (LoRA) against full fine-tuning
for both BERT and ModernBERT across ranks {4, 8, 16, 32, 64} and two
target-module configurations (standard / aggressive).

Results are written to results/results_04_lora_vs_full/lora_vs_full.csv.
Re-running the script after a crash resumes from where it left off.

Usage:
    python src/train_lora.py
"""

# %%
import csv
import os
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
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
# Fraction of training data to use (1.0 = full dataset, 0.2 = 20% stratified)
TRAIN_SUBSAMPLE = 0.2

MODELS = {
    "bert": "google-bert/bert-base-cased",
    "modernbert": "answerdotai/ModernBERT-base",
}

MAX_LENGTH = 512
BATCH_SIZE = 32
GRAD_ACCUM = 8  # effective batch = 256
LR_FULL = 1e-5   # full fine-tuning
LR_LORA = 1e-3   # LoRA (LR search confirmed 1e-3 best for both BERT and ModernBERT)
WEIGHT_DECAY = 0.01
EPOCHS = 10
SEED = 42

LORA_RANKS = [4, 8, 16, 32, 64]
LORA_DROPOUT = 0.05

# Target module configurations per model
TARGET_MODULES = {
    "bert": {
        "standard":   ["query", "value"],
        "aggressive": ["query", "key", "value", "dense"],
    },
    "modernbert": {
        "standard":   ["Wqkv"],
        "aggressive": ["Wqkv", "Wo", "Wi", "mlp.Wo"],
    },
}

RESULTS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "results",
    "results_04_lora_vs_full", "lora_vs_full.csv"
)
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "lora_vs_full"
)

CSV_COLUMNS = (
    ["model", "lora_rank", "target_modules", "seed",
     "trainable_params", "total_params", "trainable_pct",
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
    remaining = list(set(range(len(dataset))) - set(indices))
    if len(indices) < n:
        extra = rng.choice(remaining, size=n - len(indices), replace=False)
        indices.extend(extra.tolist())
    indices = sorted(indices[:n])
    return dataset.select(indices)


def load_completed_runs(csv_path):
    """Return a set of (model, lora_rank, target_modules, seed) tuples already in CSV."""
    if not os.path.exists(csv_path):
        return set()
    completed = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["model"], int(row["lora_rank"]), row["target_modules"], int(row["seed"])))
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


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def tokenize_dataset(newsgroups, tokenizer, max_length):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    encoded = newsgroups.map(tok, batched=True, remove_columns=["text", "label_text"])
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return encoded


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def smoke_test(device, newsgroups, num_labels, label_names):
    print("\n=== Smoke test: BERT, rank=8, ['query', 'value'], 1 epoch ===")
    set_seed(SEED)
    model_id = MODELS["bert"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoded = tokenize_dataset(newsgroups, tokenizer, MAX_LENGTH)

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)

    args = TrainingArguments(
        output_dir="/tmp/smoke_lora",
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR_LORA,
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
# Single LoRA run
# ---------------------------------------------------------------------------
def run_lora(model_key, model_id, rank, target_config, target_modules_list,
             seed, device, encoded, tokenizer, num_labels, label_names):
    set_seed(seed)

    checkpoint_dir = os.path.join(
        ARTIFACTS_DIR, model_key, f"rank{rank}_{target_config}_seed{seed}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules_list,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)

    trainable_params, total_params = count_params(model)
    trainable_pct = 100.0 * trainable_params / total_params

    args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR_LORA,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
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

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    trainer.save_model(checkpoint_dir)

    pred_output = trainer.predict(encoded["test"])
    metrics = helpers.compute_metrics(pred_output.predictions, pred_output.label_ids, label_names)
    peak_mem = mps_memory_gb()

    del model
    clear_mps()

    row = {
        "model": model_key,
        "lora_rank": rank,
        "target_modules": target_config,
        "seed": seed,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(trainable_pct, 4),
        "train_time_seconds": round(train_time, 1),
        "peak_mps_memory_gb": round(peak_mem, 3),
        **metrics,
    }
    return row


# ---------------------------------------------------------------------------
# Single full fine-tuning run
# ---------------------------------------------------------------------------
def run_full(model_key, model_id, seed, device, encoded, tokenizer, num_labels, label_names):
    set_seed(seed)

    checkpoint_dir = os.path.join(ARTIFACTS_DIR, model_key, f"full_seed{seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
    model.gradient_checkpointing_enable()
    model.to(device)

    trainable_params, total_params = count_params(model)
    trainable_pct = 100.0 * trainable_params / total_params

    args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR_FULL,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
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

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    pred_output = trainer.predict(encoded["test"])
    metrics = helpers.compute_metrics(pred_output.predictions, pred_output.label_ids, label_names)
    peak_mem = mps_memory_gb()

    del model
    clear_mps()

    row = {
        "model": model_key,
        "lora_rank": 0,
        "target_modules": "full",
        "seed": seed,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(trainable_pct, 4),
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

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    completed = load_completed_runs(RESULTS_CSV)
    print(f"Already completed runs: {len(completed)}")

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

    smoke_test(device, newsgroups, num_labels, label_names)

    # Build the full sweep list:
    # (model_key, model_id, lora_rank, target_config, seed)
    # lora_rank=0 + target_config="full" means full fine-tuning
    sweep = []

    # Full fine-tuning baselines (rank=0)
    for model_key, model_id in MODELS.items():
        sweep.append((model_key, model_id, 0, "full", SEED))

    # LoRA runs
    for model_key, model_id in MODELS.items():
        for target_config in ["standard", "aggressive"]:
            for rank in LORA_RANKS:
                sweep.append((model_key, model_id, rank, target_config, SEED))

    total = len(sweep)
    sweep_t0 = time.time()

    for idx, (model_key, model_id, rank, target_config, seed) in enumerate(sweep, 1):
        tag = (model_key, rank, target_config, seed)
        if tag in completed:
            label = "full" if rank == 0 else f"rank={rank} {target_config}"
            print(f"[{idx}/{total}] SKIP  {model_key:12s}  {label}  seed={seed} (already done)")
            continue

        label = "full fine-tuning" if rank == 0 else f"rank={rank} {target_config}"
        elapsed = time.time() - sweep_t0
        print(
            f"\n{'='*70}\n"
            f"[{idx}/{total}] START  {model_key}  {label}  seed={seed}\n"
            f"  batch={BATCH_SIZE}  grad_accum={GRAD_ACCUM}  effective_batch=256  "
            f"elapsed={elapsed/60:.1f}min\n"
            f"{'='*70}"
        )

        # Tokenize fresh for each model (tokenizer may differ)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        encoded = tokenize_dataset(newsgroups, tokenizer, MAX_LENGTH)

        if rank == 0:
            row = run_full(model_key, model_id, seed, device, encoded, tokenizer, num_labels, label_names)
        else:
            modules = TARGET_MODULES[model_key][target_config]
            row = run_lora(model_key, model_id, rank, target_config, modules,
                           seed, device, encoded, tokenizer, num_labels, label_names)

        append_row(RESULTS_CSV, row)
        completed.add(tag)
        print(
            f"[{idx}/{total}] DONE   {model_key}  {label}  seed={seed}  "
            f"macro_f1={row['macro_f1']:.4f}  "
            f"trainable={row['trainable_params']:,} ({row['trainable_pct']:.2f}%)  "
            f"time={row['train_time_seconds']:.0f}s  "
            f"mem={row['peak_mps_memory_gb']:.2f}GB"
        )

    print(f"\nSweep complete. Results written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
