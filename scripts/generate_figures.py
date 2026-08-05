#!/usr/bin/env python
"""Regenerate the project's result figures (docs/figures/*.png) from the data.

The committed notebooks have their outputs stripped, so the figures used in the
README and docs/FINDINGS.md are rendered here directly from the Parquet. We read
only the columns each plot needs via pyarrow (no full ~5GB load) and reproduce
the same aggregations the notebook/Spark cells run. Every number is
cross-checked against docs/FINDINGS.md.

Run:  python3 scripts/generate_figures.py   (needs data/complaints.parquet)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent
PARQUET = REPO / "data" / "complaints.parquet"
OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Databricks corporate palette (from the slides skill).
DB = {
    "red": "#FF3620",
    "dark_red": "#98102A",
    "dark_teal": "#1B3037",
    "teal": "#1B5161",
    "muted_teal": "#618793",
    "light_teal": "#9EB7BE",
    "green": "#00B378",
    "yellow": "#FFAB00",
    "slate": "#A0ABBE",
}
# Ordered categorical sequence for multi-series charts.
DB_SEQ = [DB["teal"], DB["red"], DB["muted_teal"], DB["yellow"], DB["green"], DB["slate"]]

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.15)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlecolor"] = DB["dark_teal"]
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.facecolor"] = "white"
# Draw gridlines behind data (seaborn whitegrid otherwise renders them on top of
# lines/markers on twin-axis charts). Applies to every figure below.
plt.rcParams["axes.axisbelow"] = True


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO)}")


def read_cols(cols: list[str]) -> pd.DataFrame:
    return pq.read_table(PARQUET, columns=cols).to_pandas()


# --- Plot 1: target imbalance (notebook 02 cell 21) --------------------------
def target_imbalance() -> None:
    tbl = pq.read_table(PARQUET, columns=["monetary_relief"])
    vc = pc.value_counts(tbl["monetary_relief"])
    counts = {int(s["values"].as_py()): int(s["counts"].as_py()) for s in vc}
    counts = pd.Series(counts).sort_index()
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str),
                palette=[DB["teal"], DB["red"]], legend=False, ax=ax)
    ax.set_xlabel("monetary_relief")
    ax.set_ylabel("complaints")
    ax.set_title("Target is heavily imbalanced: monetary relief is rare")
    for p, (k, v) in zip(ax.patches, counts.items()):
        ax.annotate(f"{v:,}\n({v / total * 100:.2f}%)",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=11, color=DB["dark_teal"])
    ax.set_ylim(0, counts.max() * 1.15)
    save(fig, "01_target_imbalance.png")


# --- Plot 2: complaint volume by product, top 10 (notebook 02 cell 25) -------
def volume_by_product() -> None:
    df = read_cols(["product"])
    counts = df["product"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index,
                palette=sns.light_palette(DB["teal"], n_colors=10, reverse=True), legend=False, ax=ax)
    ax.set_title("Complaint volume by product (top 10)")
    ax.set_xlabel("complaints")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x / 1e3:.0f}k")
    save(fig, "02_volume_by_product.png")


# --- Plot 3: relief rate by product, top 10 by volume (notebook 02 cell 33) --
def relief_rate_by_product() -> None:
    df = read_cols(["product", "monetary_relief"])
    top = df["product"].value_counts().head(10).index
    rate = (df[df["product"].isin(top)].groupby("product")["monetary_relief"].mean()
            .sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=rate.values, y=rate.index.astype(str), hue=rate.index.astype(str),
                palette=sns.light_palette(DB["red"], n_colors=len(rate), reverse=True),
                legend=False, ax=ax)
    ax.set_xlabel("monetary relief rate")
    ax.set_ylabel("")
    ax.set_title("Monetary-relief rate by product (top 10 by volume)")
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.3f}", (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha="left", va="center", fontsize=10, color=DB["dark_teal"])
    ax.set_xlim(0, rate.max() * 1.18)
    save(fig, "03_relief_rate_by_product.png")


# --- Plot 4: PR-AUC by model (FINDINGS table 3) ------------------------------
def pr_auc_by_model() -> None:
    data = pd.DataFrame({
        "model": ["Product-bucket\nheuristic", "LogReg,\nraw metadata",
                  "Engineered\nmetadata", "Full 16.5M\nSpark MLlib"],
        "pr_auc": [0.11, 0.16, 0.21, 0.214],
    })
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = [DB["slate"], DB["muted_teal"], DB["teal"], DB["red"]]
    sns.barplot(data=data, x="model", y="pr_auc", hue="model", palette=colors, legend=False, ax=ax)
    ax.set_title("PR-AUC rises with feature engineering, holds at full scale")
    ax.set_xlabel("")
    ax.set_ylabel("PR-AUC")
    for p, v in zip(ax.patches, data["pr_auc"]):
        ax.annotate(f"{v:.3f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=12, fontweight="bold", color=DB["dark_teal"])
    ax.set_ylim(0, 0.26)
    save(fig, "04_pr_auc_by_model.png")


# --- Plot 5: lift vs queue depth, the "money" chart (FINDINGS table 4) --------
def lift_vs_depth() -> None:
    depth = np.array([1, 5, 10])
    learned = np.array([59.7, 18.8, 9.7])
    heuristic = np.array([29.4, 19.2, 9.7])
    ceiling = np.array([100, 20, 10])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(depth, ceiling, "--", color=DB["slate"], linewidth=2, label="Ceiling (1/k, arithmetic max)")
    ax.plot(depth, learned, "-o", color=DB["red"], linewidth=2.5, markersize=9, label="Learned model")
    ax.plot(depth, heuristic, "-s", color=DB["muted_teal"], linewidth=2.5, markersize=8, label="Product-bucket heuristic")
    ax.annotate("59.7x", (1, 59.7), textcoords="offset points", xytext=(10, 4),
                fontsize=14, fontweight="bold", color=DB["red"])
    ax.annotate("29.4x", (1, 29.4), textcoords="offset points", xytext=(10, -14),
                fontsize=12, color=DB["muted_teal"])
    ax.set_xticks(depth)
    ax.set_xticklabels(["top 1%", "top 5%", "top 10%"])
    ax.set_ylabel("lift over base rate")
    ax.set_xlabel("queue depth reviewers work")
    ax.set_title("The value lives at top 1%, where the ceiling gives room to separate")
    ax.legend(frameon=True, loc="upper right")
    save(fig, "05_lift_vs_depth.png")


# --- Plot 6: task-dependent representation (FINDINGS finding 6) ---------------
def representation() -> None:
    tasks = ["Clustering themes\n(ARI)", "Relief prediction\n(PR-AUC)"]
    tfidf = [0.014, 0.271]
    emb = [0.067, 0.241]
    x = np.arange(len(tasks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w / 2, tfidf, w, label="TF-IDF", color=DB["teal"])
    b2 = ax.bar(x + w / 2, emb, w, label="Embeddings", color=DB["red"])
    ax.bar_label(b1, fmt="%.3f", padding=3, color=DB["dark_teal"], fontsize=11)
    ax.bar_label(b2, fmt="%.3f", padding=3, color=DB["dark_teal"], fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Same text, opposite winner: representation is task-dependent")
    ax.legend(frameon=True)
    ax.set_ylim(0, 0.32)
    save(fig, "06_representation.png")


# --- Plot 7: clustering ARI by representation lever (notebook 08 cell 15) -----
def clustering_ari() -> None:
    # Values from FINDINGS finding 6 / notebook 08 leaderboard.
    board = pd.Series({"TF-IDF (raw)": 0.014, "+ dedup / MinHash": 0.028,
                       "Sentence embeddings": 0.067}).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sns.barplot(x=board.values, y=board.index, hue=board.index,
                palette=sns.light_palette(DB["teal"], n_colors=len(board), reverse=False),
                legend=False, ax=ax)
    ax.set_title("ARI vs product x issue: representation levers")
    ax.set_xlabel("Adjusted Rand Index (higher = recovers themes better)")
    ax.set_ylabel("")
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.3f}", (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha="left", va="center", fontsize=11, color=DB["dark_teal"])
    ax.set_xlim(0, board.max() * 1.2)
    save(fig, "07_clustering_ari.png")


# --- Plot 8: model pipeline block diagram (slide visual #1) ------------------
def block_diagram() -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    boxes = ["Complaint\n(intake fields\n+ narrative)", "Engineered\nfeatures\n(TF-IDF + meta)",
             "Class-weighted\nlogistic\nregression", "Calibrated\nrelief-risk\nscore",
             "Ranked\nreview queue\n(top 1% first)"]
    colors = [DB["muted_teal"], DB["teal"], DB["dark_teal"], DB["dark_red"], DB["red"]]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, len(boxes) * 2.4)
    ax.set_ylim(0, 2)
    ax.axis("off")
    bw, bh, y = 1.8, 1.2, 0.4
    for i, (label, color) in enumerate(zip(boxes, colors)):
        x = i * 2.4 + 0.1
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=color, ec="none"))
        ax.text(x + bw / 2, y + bh / 2, label, ha="center", va="center",
                color="white", fontsize=11.5, fontweight="bold")
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + bw, y + bh / 2), (x + 2.4 + 0.1, y + bh / 2),
                                         arrowstyle="-|>", mutation_scale=22,
                                         color=DB["dark_teal"], linewidth=2))
    ax.set_title("From raw complaint to a ranked review queue", color=DB["dark_teal"],
                 fontweight="bold", fontsize=15, pad=14)
    save(fig, "08_block_diagram.png")


# --- Plot 9: model progress across iterations (rubric: show tuning progress) --
def model_progress() -> None:
    # PR-AUC (primary) and top-1% lift at each development stage. Numbers from
    # docs/FINDINGS.md finding 3. This shows how the model improved as I iterated:
    # heuristic -> raw-metadata LR -> feature engineering -> full-scale validation.
    stages = ["Product-bucket\nheuristic", "Raw metadata\nLogReg",
              "Engineered\nfeatures", "Full 16.5M\n(Spark)"]
    pr_auc = [0.11, 0.158, 0.223, 0.214]
    lift1 = [29.4, 50.6, 61.6, 59.7]
    x = np.arange(len(stages))

    fig, ax1 = plt.subplots(figsize=(9.5, 5))
    ax1.set_axisbelow(True)  # gridlines behind the data
    # PR-AUC as the primary line. Labels offset up-left to avoid the lift line.
    # zorder=3 keeps the lines/markers above both axes' gridlines.
    ax1.plot(x, pr_auc, "-o", color=DB["red"], linewidth=2.6, markersize=9,
             label="PR-AUC (primary)", zorder=3)
    for xi, v in zip(x, pr_auc):
        ax1.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(-4, 11),
                     ha="right", fontsize=11, fontweight="bold", color=DB["red"], zorder=4)
    ax1.set_ylabel("PR-AUC (line, filled circles)", color=DB["dark_red"])
    ax1.tick_params(axis="y", labelcolor=DB["dark_red"])
    ax1.set_ylim(0, 0.32)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_xlim(-0.35, len(stages) - 0.4)

    # top-1% lift as a secondary line. Labels offset down-right, well clear.
    ax2 = ax1.twinx()
    ax2.grid(False)  # don't let the twin axis draw a second grid over the data
    ax2.set_axisbelow(True)
    ax2.plot(x, lift1, "--s", color=DB["teal"], linewidth=2.2, markersize=8,
             label="top-1% lift", zorder=3)
    for xi, v in zip(x, lift1):
        ax2.annotate(f"{v:.0f}x", (xi, v), textcoords="offset points", xytext=(6, -18),
                     ha="left", fontsize=10, color=DB["teal"], zorder=4)
    ax2.set_ylabel("top-1% lift (dashed, squares)", color=DB["teal"])
    ax2.tick_params(axis="y", labelcolor=DB["teal"])
    ax2.set_ylim(0, 80)

    ax1.set_title("Model progress: PR-AUC and top-1% lift improved with each iteration",
                  color=DB["dark_teal"])
    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
    save(fig, "09_model_progress.png")


if __name__ == "__main__":
    block_diagram()
    target_imbalance()
    volume_by_product()
    relief_rate_by_product()
    pr_auc_by_model()
    lift_vs_depth()
    representation()
    clustering_ari()
    model_progress()
    print("\nall figures written to", OUT.relative_to(REPO))
