"""Plot results of the LoRA vs Full Fine-Tuning sweep.

Reads results/results_04_lora_vs_full/lora_vs_full.csv and produces:
  1. Efficiency frontier: macro F1 vs trainable parameter count (log x-axis),
     one curve per model, horizontal dashed reference at each model's full FT F1.
  2. Rank curve: macro F1 vs rank r, one line per target-module config,
     one subplot per model, same full FT reference line.

Usage:
    python src/plot_lora.py
"""

# %%
import os

import matplotlib.pyplot as plt
import pandas as pd

# %%
RESULTS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "results",
    "results_04_lora_vs_full", "lora_vs_full.csv"
)
PLOTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results",
    "results_04_lora_vs_full", "plots"
)
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(RESULTS_CSV)
print(df.shape)
df.head()

# %%
# Split full FT rows from LoRA rows
full_df = df[df["lora_rank"] == 0].copy()
lora_df = df[df["lora_rank"] > 0].copy()

MODEL_LABELS = {"bert": "BERT-base", "modernbert": "ModernBERT-base"}
MODEL_COLORS = {"bert": "#2166ac", "modernbert": "#d6604d"}
CONFIG_STYLES = {"standard": ("o", "-"), "aggressive": ("s", "--")}

# %%
# ── Plot 1: Efficiency frontier ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

for model_key in ["bert", "modernbert"]:
    color = MODEL_COLORS[model_key]
    label = MODEL_LABELS[model_key]

    # Full FT reference line
    full_row = full_df[full_df["model"] == model_key]
    if not full_row.empty:
        full_f1 = full_row["macro_f1"].values[0]
        full_params = full_row["trainable_params"].values[0]
        ax.axhline(full_f1, color=color, linestyle=":", linewidth=1.2,
                   label=f"{label} full FT ({full_f1:.3f})")
        ax.scatter([full_params], [full_f1], color=color, marker="*", s=120, zorder=5)

    # LoRA points — aggregate across configs to get the frontier
    sub = lora_df[lora_df["model"] == model_key].sort_values("trainable_params")
    for target_config in ["standard", "aggressive"]:
        cfg_sub = sub[sub["target_modules"] == target_config]
        if cfg_sub.empty:
            continue
        marker, linestyle = CONFIG_STYLES[target_config]
        ax.plot(
            cfg_sub["trainable_params"],
            cfg_sub["macro_f1"],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=f"{label} {target_config}",
        )

ax.set_xscale("log")
ax.set_xlabel("Trainable parameters (log scale)")
ax.set_ylabel("Macro F1")
ax.set_title("Efficiency Frontier: Macro F1 vs Trainable Parameters")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "efficiency_frontier.png"), dpi=150)
plt.show()

# %%
# ── Plot 2: Rank curve (one subplot per model) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

for ax, model_key in zip(axes, ["bert", "modernbert"]):
    color = MODEL_COLORS[model_key]
    label = MODEL_LABELS[model_key]

    # Full FT reference
    full_row = full_df[full_df["model"] == model_key]
    if not full_row.empty:
        full_f1 = full_row["macro_f1"].values[0]
        ax.axhline(full_f1, color="black", linestyle=":", linewidth=1.4,
                   label=f"Full FT ({full_f1:.3f})")

    sub = lora_df[lora_df["model"] == model_key]
    for target_config in ["standard", "aggressive"]:
        cfg_sub = sub[sub["target_modules"] == target_config].sort_values("lora_rank")
        if cfg_sub.empty:
            continue
        marker, linestyle = CONFIG_STYLES[target_config]
        ax.plot(
            cfg_sub["lora_rank"],
            cfg_sub["macro_f1"],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=target_config,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.set_xticklabels([4, 8, 16, 32, 64])
    ax.set_xlabel("LoRA rank r (log₂ scale)")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Rank Curve — {label}")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

plt.suptitle("Macro F1 vs LoRA Rank — BERT vs ModernBERT", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rank_curve.png"), dpi=150)
plt.show()

# %%
# ── Summary table ────────────────────────────────────────────────────────────
print("\n=== Summary ===")
summary_cols = ["model", "lora_rank", "target_modules", "trainable_params",
                "trainable_pct", "macro_f1", "train_time_seconds"]
print(df[summary_cols].sort_values(["model", "lora_rank", "target_modules"]).to_string(index=False))

print(f"\nAll plots saved to {PLOTS_DIR}")
