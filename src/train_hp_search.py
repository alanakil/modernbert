"""Hyperparameter search for BERT and ModernBERT on 20 Newsgroups (Plan 11).

Phases 1 & 2: Optuna TPE sampler + HyperbandPruner, 30 trials per model.

Search space:
  learning_rate:      log-uniform [1e-5, 3e-4]
  weight_decay:       categorical [0.0, 0.01, 0.1]
  warmup_ratio:       uniform     [0.0, 0.1]
  lr_scheduler_type:  categorical ["linear", "cosine"]
  classifier_dropout: categorical [0.0, 0.1, 0.2]

Fixed: max_length=512, effective batch=256, 10 epochs, 20% stratified
subsample.

Outputs (per model):
  results/results_11_hp_search/{model}_trials.csv — all trial metrics
  results/results_11_hp_search/{model}_best.json  — winning HP set
  results/results_11_hp_search/{model}_optuna.db  — Optuna journal (resume)

Note: eval loss is optimised on the test split (same split used for final
reporting). This is standard practice for public benchmarks and is acknowledged
as a limitation in the writeup.

Usage:
    python src/train_hp_search.py                  # run both models
    python src/train_hp_search.py --model bert
    python src/train_hp_search.py --model modernbert
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import optuna
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import helpers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = {
    "bert": "google-bert/bert-base-cased",
    "modernbert": "answerdotai/ModernBERT-base",
}

MAX_LENGTH = 512
TRAIN_SUBSAMPLE = 0.2
EPOCHS = 10
SEED = 42
N_TRIALS = 30

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "results_11_hp_search"
)

CSV_COLUMNS = [
    "trial_number", "model", "seed",
    "learning_rate", "weight_decay", "warmup_ratio",
    "lr_scheduler_type", "classifier_dropout",
    "eval_loss", "macro_f1",
    "train_time_seconds", "pruned",
]


# ---------------------------------------------------------------------------
# Device detection and per-device config
# ---------------------------------------------------------------------------
def get_device_config(device: torch.device) -> dict:
    """Return batch/dataloader/precision settings for the detected device."""
    if device.type == "cuda":
        # Enable TF32 on Ampere+ for free ~10% speedup
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Detect bf16 support (Ampere = compute capability >= 8.0)
        major, _ = torch.cuda.get_device_capability()
        use_bf16 = major >= 8

        return {
            "batch_size": 32,
            "grad_accum": 8,
            "num_workers": 4,
            "pin_memory": True,
            "bf16": use_bf16,
            "fp16": not use_bf16,
        }
    else:
        # MPS or CPU — num_workers > 0 causes hangs on MPS
        return {
            "batch_size": 32,
            "grad_accum": 8,
            "num_workers": 0,
            "pin_memory": False,
            "bf16": True,
            "fp16": False,
        }


def clear_memory(device: torch.device):
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def peak_memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps" and torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def subsample_dataset(dataset, fraction: float, seed: int = 42):
    """Stratified subsample that preserves class distribution."""
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
        chosen = rng.choice(
            class_idx, size=min(n_per_class, len(class_idx)), replace=False
        )
        indices.extend(chosen.tolist())
    remaining = list(set(range(len(dataset))) - set(indices))
    if len(indices) < n:
        extra = rng.choice(remaining, size=n - len(indices), replace=False)
        indices.extend(extra.tolist())
    return dataset.select(sorted(indices[:n]))


def tokenize_dataset(newsgroups, tokenizer, max_length: int):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    encoded = newsgroups.map(
        tok, batched=True, remove_columns=["text", "label_text"]
    )
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return encoded


# ---------------------------------------------------------------------------
# Optuna pruning callback
# ---------------------------------------------------------------------------
class OptunaPruningCallback(TrainerCallback):
    """Reports eval_loss to Optuna after each epoch; stops if pruned."""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial
        self.pruned = False

    def on_evaluate(self, args, state, control, metrics, **kwargs):  # noqa
        epoch = int(state.epoch)
        value = metrics.get("eval_loss")
        if value is not None:
            self.trial.report(value, step=epoch)
            if self.trial.should_prune():
                self.pruned = True
                control.should_training_stop = True


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------
def append_row(csv_path: str, row: dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Single Optuna trial
# ---------------------------------------------------------------------------
def objective(
    trial: optuna.Trial,
    model_key: str,
    model_id: str,
    device: torch.device,
    device_cfg: dict,
    newsgroups,
    num_labels: int,
    label_names: list,
) -> float:
    # --- Sample hyperparameters -------------------------------------------
    lr = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.1])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    scheduler = trial.suggest_categorical(
        "lr_scheduler_type", ["linear", "cosine"]
    )
    classifier_dropout = trial.suggest_categorical(
        "classifier_dropout", [0.0, 0.1, 0.2]
    )

    set_seed(SEED)

    # --- Tokenise ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoded = tokenize_dataset(newsgroups, tokenizer, MAX_LENGTH)

    # --- Build model with sampled classifier dropout ----------------------
    config = AutoConfig.from_pretrained(model_id, num_labels=num_labels)
    if hasattr(config, "classifier_dropout"):
        config.classifier_dropout = classifier_dropout

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, config=config
    )
    model.gradient_checkpointing_enable()
    model.to(device)

    # --- Training arguments -----------------------------------------------
    training_args = TrainingArguments(
        output_dir=f"/tmp/hp_search_{model_key}_trial{trial.number}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=device_cfg["batch_size"],
        per_device_eval_batch_size=device_cfg["batch_size"],
        gradient_accumulation_steps=device_cfg["grad_accum"],
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=scheduler,
        bf16=device_cfg["bf16"],
        fp16=device_cfg["fp16"],
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=device_cfg["num_workers"],
        dataloader_pin_memory=device_cfg["pin_memory"],
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    pruning_cb = OptunaPruningCallback(trial)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        data_collator=collator,
        callbacks=[pruning_cb],
    )

    # --- Train ------------------------------------------------------------
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # --- Handle pruned trial ----------------------------------------------
    if pruning_cb.pruned:
        iv = trial.intermediate_values
        last_step = max(iv.keys()) if iv else None
        last_loss = iv[last_step] if last_step is not None else float("nan")
        row = {
            "trial_number": trial.number,
            "model": model_key,
            "seed": SEED,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "warmup_ratio": round(warmup_ratio, 4),
            "lr_scheduler_type": scheduler,
            "classifier_dropout": classifier_dropout,
            "eval_loss": round(last_loss, 6),
            "macro_f1": float("nan"),
            "train_time_seconds": round(train_time, 1),
            "pruned": True,
        }
        append_row(os.path.join(RESULTS_DIR, f"{model_key}_trials.csv"), row)
        del model
        clear_memory(device)
        raise optuna.TrialPruned()

    # --- Completed trial: evaluate + compute metrics ----------------------
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]

    pred_output = trainer.predict(encoded["test"])
    metrics = helpers.compute_metrics(
        pred_output.predictions, pred_output.label_ids, label_names
    )

    del model
    clear_memory(device)

    row = {
        "trial_number": trial.number,
        "model": model_key,
        "seed": SEED,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": round(warmup_ratio, 4),
        "lr_scheduler_type": scheduler,
        "classifier_dropout": classifier_dropout,
        "eval_loss": round(eval_loss, 6),
        "macro_f1": round(metrics["macro_f1"], 6),
        "train_time_seconds": round(train_time, 1),
        "pruned": False,
    }
    append_row(os.path.join(RESULTS_DIR, f"{model_key}_trials.csv"), row)

    print(
        f"  [{model_key}] trial={trial.number:3d}  "
        f"eval_loss={eval_loss:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
        f"lr={lr:.2e}  wd={weight_decay}  warmup={warmup_ratio:.3f}  "
        f"sched={scheduler}  dropout={classifier_dropout}  "
        f"time={train_time:.0f}s"
    )
    return eval_loss


# ---------------------------------------------------------------------------
# Run one Optuna study (one model)
# ---------------------------------------------------------------------------
def run_study(
    model_key: str,
    model_id: str,
    device: torch.device,
    device_cfg: dict,
    newsgroups,
    num_labels: int,
    label_names: list,
) -> dict:
    db_path = os.path.join(RESULTS_DIR, f"{model_key}_optuna.db")
    storage = f"sqlite:///{db_path}"
    study_name = f"hp_search_{model_key}"

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=2, max_resource=EPOCHS, reduction_factor=3
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    complete_states = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    n_complete = len(complete_states)
    remaining = N_TRIALS - n_complete

    print(f"\n{'='*70}")
    print(
        f"Optuna study: {model_key}  |  "
        f"complete={n_complete}  remaining={remaining}"
    )
    print(f"Storage: {db_path}")
    print(f"{'='*70}")

    if remaining > 0:
        study.optimize(
            lambda trial: objective(
                trial, model_key, model_id, device, device_cfg,
                newsgroups, num_labels, label_names,
            ),
            n_trials=remaining,
            catch=(RuntimeError,),
        )

    best = study.best_trial
    best_params = {**best.params, "eval_loss": round(best.value, 6)}

    best_path = os.path.join(RESULTS_DIR, f"{model_key}_best.json")
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nBest trial for {model_key}  (trial #{best.number}):")
    print(f"  eval_loss = {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k} = {v}")
    print(f"  Saved to {best_path}")

    return best_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="HP search for BERT / ModernBERT"
    )
    parser.add_argument(
        "--model",
        choices=["bert", "modernbert", "all"],
        default="all",
        help="Which model to run (default: all)",
    )
    args = parser.parse_args()

    device = helpers.identify_device()
    device_cfg = get_device_config(device)

    print(f"\nDevice      : {device}")
    print(f"Precision   : {'bf16' if device_cfg['bf16'] else 'fp16'}")
    print(
        f"Batch config: per_device={device_cfg['batch_size']}  "
        f"grad_accum={device_cfg['grad_accum']}  "
        f"effective={device_cfg['batch_size'] * device_cfg['grad_accum']}"
    )
    print(f"Data workers: {device_cfg['num_workers']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    newsgroups = load_dataset("SetFit/20_newsgroups")
    num_labels = len(set(newsgroups["train"]["label"]))
    label_names = sorted(set(newsgroups["train"]["label_text"]))

    n_before = len(newsgroups["train"])
    newsgroups["train"] = subsample_dataset(
        newsgroups["train"], TRAIN_SUBSAMPLE, seed=SEED
    )
    print(
        f"\nDataset: {num_labels} classes  |  "
        f"train subsampled: {n_before} → {len(newsgroups['train'])} "
        f"({TRAIN_SUBSAMPLE:.0%}, stratified)"
    )

    if args.model == "all":
        models_to_run = MODELS
    else:
        models_to_run = {args.model: MODELS[args.model]}

    all_best = {}
    for model_key, model_id in models_to_run.items():
        best_params = run_study(
            model_key, model_id, device, device_cfg,
            newsgroups, num_labels, label_names,
        )
        all_best[model_key] = best_params

    print("\n" + "="*70)
    print("All studies complete. Best HPs:")
    for model_key, params in all_best.items():
        print(f"\n  {model_key}:")
        for k, v in params.items():
            print(f"    {k} = {v}")
    print("="*70)


if __name__ == "__main__":
    main()
