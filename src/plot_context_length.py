# %%
"""Plot results of the context-length ablation sweep.

Reads results/ablation_context_length.csv and produces:
  1. Macro F1 vs. max_length (mean ± std across 3 seeds, per model)
  2. Training time vs. max_length
  3. Per-class F1 heatmap (mean across seeds) for each model
  4. Annotates BERT ceiling, project baseline, and ModernBERT crossover point

Usage (as a script):
    python src/plot_context_length.py

Or run cell-by-cell in VS Code / Jupyter via the # %% markers.
"""

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "ablation_context_length.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(RESULTS_CSV)
print(df.shape)
df.head()

# %%
# Aggregate across seeds
agg = (
    df.groupby(["model", "max_length"])
    .agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        train_time_mean=("train_time_seconds", "mean"),
        train_time_std=("train_time_seconds", "std"),
    )
    .reset_index()
)

bert_agg = agg[agg["model"] == "bert"].sort_values("max_length")
mbert_agg = agg[agg["model"] == "modernbert"].sort_values("max_length")

# %%
# ── Plot 1: Macro F1 vs. max_length ─────────────────────────────────────────
BERT_CEILING = 512
BASELINE = 50

fig, ax = plt.subplots(figsize=(9, 5))

# BERT line
ax.plot(bert_agg["max_length"], bert_agg["macro_f1_mean"],
        marker="o", label="BERT-base", color="#2166ac")
ax.fill_between(
    bert_agg["max_length"],
    bert_agg["macro_f1_mean"] - bert_agg["macro_f1_std"],
    bert_agg["macro_f1_mean"] + bert_agg["macro_f1_std"],
    alpha=0.2, color="#2166ac",
)

# ModernBERT line
ax.plot(mbert_agg["max_length"], mbert_agg["macro_f1_mean"],
        marker="s", label="ModernBERT-base", color="#d6604d")
ax.fill_between(
    mbert_agg["max_length"],
    mbert_agg["macro_f1_mean"] - mbert_agg["macro_f1_std"],
    mbert_agg["macro_f1_mean"] + mbert_agg["macro_f1_std"],
    alpha=0.2, color="#d6604d",
)

# Annotations
ax.axvline(BASELINE, linestyle="--", color="gray", linewidth=1.2, label=f"Project baseline ({BASELINE})")
ax.axvline(BERT_CEILING, linestyle=":", color="#2166ac", linewidth=1.2, label=f"BERT ceiling ({BERT_CEILING})")

# Crossover annotation: first length where ModernBERT leads BERT by >1pp
merged = pd.merge(
    bert_agg[["max_length", "macro_f1_mean"]].rename(columns={"macro_f1_mean": "bert_f1"}),
    mbert_agg[["max_length", "macro_f1_mean"]].rename(columns={"macro_f1_mean": "mbert_f1"}),
    on="max_length",
    how="inner",
)
merged["gap"] = merged["mbert_f1"] - merged["bert_f1"]
crossover_rows = merged[merged["gap"] > 0.01]
if not crossover_rows.empty:
    cx = crossover_rows.iloc[0]["max_length"]
    cy = crossover_rows.iloc[0]["mbert_f1"]
    ax.annotate(
        f">1pp gap\nat {int(cx)}",
        xy=(cx, cy), xytext=(cx * 1.15, cy - 0.015),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

ax.set_xscale("log", base=2)
ax.set_xticks(mbert_agg["max_length"].tolist())
ax.set_xticklabels(mbert_agg["max_length"].tolist())
ax.set_xlabel("max_length (tokens, log₂ scale)")
ax.set_ylabel("Macro F1 (mean ± std, 3 seeds)")
ax.set_title("Macro F1 vs. Context Length — BERT vs. ModernBERT")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "macro_f1_vs_length.png"), dpi=150)
plt.show()

# %%
# ── Plot 2: Training time vs. max_length ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(bert_agg["max_length"], bert_agg["train_time_mean"] / 60,
        marker="o", label="BERT-base", color="#2166ac")
ax.fill_between(
    bert_agg["max_length"],
    (bert_agg["train_time_mean"] - bert_agg["train_time_std"]) / 60,
    (bert_agg["train_time_mean"] + bert_agg["train_time_std"]) / 60,
    alpha=0.2, color="#2166ac",
)

ax.plot(mbert_agg["max_length"], mbert_agg["train_time_mean"] / 60,
        marker="s", label="ModernBERT-base", color="#d6604d")
ax.fill_between(
    mbert_agg["max_length"],
    (mbert_agg["train_time_mean"] - mbert_agg["train_time_std"]) / 60,
    (mbert_agg["train_time_mean"] + mbert_agg["train_time_std"]) / 60,
    alpha=0.2, color="#d6604d",
)

ax.set_xscale("log", base=2)
ax.set_xticks(mbert_agg["max_length"].tolist())
ax.set_xticklabels(mbert_agg["max_length"].tolist())
ax.set_xlabel("max_length (tokens, log₂ scale)")
ax.set_ylabel("Training time (minutes, mean ± std)")
ax.set_title("Training Time vs. Context Length")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "train_time_vs_length.png"), dpi=150)
plt.show()

# %%
# ── Plot 3: Per-class F1 heatmaps ───────────────────────────────────────────
f1_cols = [f"f1_class_{i}" for i in range(20)]

for model_key, model_label in [("bert", "BERT-base"), ("modernbert", "ModernBERT-base")]:
    sub = df[df["model"] == model_key]
    pivot = (
        sub.groupby("max_length")[f1_cols]
        .mean()
        .T
    )
    pivot.index = [f"class_{i}" for i in range(20)]
    pivot.columns = [str(c) for c in pivot.columns]

    valid_lengths = [c for c in pivot.columns if int(c) <= (BERT_CEILING if model_key == "bert" else 99999)]
    pivot = pivot[valid_lengths]

    fig, ax = plt.subplots(figsize=(len(valid_lengths) * 1.2 + 2, 7))
    sns.heatmap(
        pivot.astype(float),
        ax=ax,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "F1"},
    )
    ax.set_xlabel("max_length")
    ax.set_ylabel("Class")
    ax.set_title(f"Per-class F1 vs. Context Length — {model_label} (mean across seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"per_class_f1_{model_key}.png"), dpi=150)
    plt.show()

# %%
print(f"All plots saved to {PLOTS_DIR}")
