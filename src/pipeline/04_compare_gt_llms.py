"""
This script compares dataset/model/metric distributions produced by multiple LLMs against
ground-truth (GT) data.

It loads per-item counts from CSV files, converts them to percentages,
aligns all metrics across models, and computes the ratio of each model’s
percentage to the GT percentage. The results are saved to a CSV file and
visualized as a scatter plot showing over- and under-representation of
metrics relative to GT, with GT percentages overlaid on a secondary axis.
"""


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

FILES = {
    "GT": "../analysis_outputs_names/1000_GT_extracted_with_fuzz/GT_Models_counts_and_percentages.csv",
    "GPT-5.1": "../analysis_outputs_names/1000_GPT51_suggested_with_fuzz/GPT51_suggested_model_counts_and_percentages.csv",
    "Gemini": "../analysis_outputs_names/1000_gemini_suggested_with_fuzz/gemini_suggested_model_counts_and_percentages.csv",
    "Deepseek": "../analysis_outputs_names/1000_deepseek_suggested_with_fuzz/deepseek_suggested_model_counts_and_percentages.csv",
}


EPS = 1e-12  # small number for safe division if you choose to use smoothing
RATIO_MODE = "nan_if_gt_zero"  

# ----------------------------
# Load CSVs
# ----------------------------
def load_csv(path):
    df = pd.read_csv(
        path,
        header=0,            # count / percent have headers
        index_col=0          # dataset/model/metric name is the index
    )

    df.index = df.index.astype(str).str.strip()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)

    total = df["count"].sum()
    if total > 0:
        df["percent"] = 100.0 * df["count"] / total
    else:
        df["percent"] = 0.0

    return df[["percent"]]


# ----------------------------
# Load all files
# ----------------------------
dfs = {name: load_csv(path) for name, path in FILES.items()}

# Union of all [dataset/model/metric] names
all_metrics = sorted(set().union(*[df.index for df in dfs.values()]))

# ----------------------------
# Align percentages
# ----------------------------
aligned = pd.DataFrame(index=all_metrics)

for name, df in dfs.items():
    aligned[f"{name}_percent"] = df.reindex(all_metrics).fillna(0)["percent"]

# ----------------------------
# Compute percent ratios vs GT
# ratio = percent_model / percent_GT
# ----------------------------
gt_pct = aligned["GT_percent"].values

def safe_ratio(model_pct, gt_pct):
    if RATIO_MODE == "eps_smooth":
        return (model_pct + EPS) / (gt_pct + EPS)

    if RATIO_MODE == "inf_if_gt_zero":
        out = np.zeros_like(model_pct)
        mask = gt_pct > 0
        out[mask] = model_pct[mask] / gt_pct[mask]
        out[~mask] = np.where(model_pct[~mask] > 0, np.inf, 0.0)
        return out

    # default: NaN if GT == 0
    out = np.full_like(model_pct, np.nan, dtype=float)
    mask = gt_pct > 0
    out[mask] = model_pct[mask] / gt_pct[mask]
    return out


for name in FILES:
    if name == "GT":
        continue
    aligned[f"{name}_over_GT"] = safe_ratio(
        aligned[f"{name}_percent"].values,
        gt_pct
    )


# ----------------------------
# Save result
# ----------------------------
out_file = Path("../analysis_outputs_names/ratios/model_percent_ratio_vs_GT.csv")
aligned = aligned.sort_values("GT_percent", ascending=False)
aligned.reset_index(names="metric").to_csv(out_file, index=False)
print(f"Saved: {out_file.resolve()}")


# ----------------------------
# Visualization 
# ----------------------------
df = pd.read_csv("../analysis_outputs_names/ratios/model_percent_ratio_vs_GT.csv")

# Number of [datasets] to plot
TOP_K = 25
df = df.head(TOP_K)   # keeps CSV order

metrics = df["metric"].tolist()
x = np.arange(len(metrics))
width = 0.2


# Ratio bars
LLMs = ["GPT-5.1", "Gemini", "Deepseek"]
offsets = {
    "GPT-5.1": 0,
    "Gemini": width,
    "Deepseek": 2 * width
}

# plot: 
fig, ax = plt.subplots(figsize=(14, 5))

for LLM in LLMs:
    # Cap extreme ratios so values >=10 are visualized at the same upper bound.
    ratio_capped = df[f"{LLM}_over_GT"].clip(upper=10)
    ax.scatter(
        x + offsets[LLM],
        ratio_capped,
        label=LLM,
        s=60,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.8
    )

# Reference line: perfect match to GT
ax.axhline(1.0, linestyle="--", linewidth=1)

ax.set_ylabel("LLM / GT ratio")
ax.set_ylim(-0.25, 10.2)
ax.set_yticks([0, 1, 2, 4, 6, 8, 10])
ax.set_yticklabels(["0", "1", "2", "4", "6", "8", "10+"])
ax.grid(axis="y", alpha=0.3)

# --- Second axis: GT percent (right axis)
ax2 = ax.twinx()
ax2.plot(x, df["GT_percent"], color="black", linewidth=2, linestyle="-", label="GT %")
ax2.set_ylabel("GT percentage (%)", fontweight="bold")
for t in ax2.get_yticklabels():
    t.set_fontweight("bold")

# --- X axis
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha="right")

ax.set_title("Model Over- / Under-Representation Relative to GT")

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper right"
)

plt.tight_layout()
plt.savefig(
    "../analysis_outputs_names/ratios/model_ratio_scatter_with_GT_percent_capped_10plus.png",
    dpi=300
)
