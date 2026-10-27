"""
Visualization and Chart Generation Script for Empirical Analysis.
Reads results from:
- results/eval_summary.json
- results/retrieval_metrics.json
- results/ablations_summary.json

Generates publication-quality charts into results/plots/:
1. faithfulness_vs_hallucination.png
2. category_breakdown.png
3. latency_vs_tokens.png
4. retrieval_chunking_ablation.png
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "results/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_faithfulness_and_hallucination(eval_summary: dict) -> Path:
    systems = list(eval_summary["systems"].keys())
    clean_names = [s.replace(" (", "\n(").replace(")", ")") for s in systems]

    faith_means = [eval_summary["systems"][s]["faithfulness"]["mean"] * 100 for s in systems]
    faith_err_low = [
        (eval_summary["systems"][s]["faithfulness"]["mean"] - eval_summary["systems"][s]["faithfulness"]["ci_lower"]) * 100
        for s in systems
    ]
    faith_err_high = [
        (eval_summary["systems"][s]["faithfulness"]["ci_upper"] - eval_summary["systems"][s]["faithfulness"]["mean"]) * 100
        for s in systems
    ]

    halluc_means = [eval_summary["systems"][s]["hallucination_rate"]["mean"] * 100 for s in systems]
    halluc_err_low = [
        (eval_summary["systems"][s]["hallucination_rate"]["mean"] - eval_summary["systems"][s]["hallucination_rate"]["ci_lower"]) * 100
        for s in systems
    ]
    halluc_err_high = [
        (eval_summary["systems"][s]["hallucination_rate"]["ci_upper"] - eval_summary["systems"][s]["hallucination_rate"]["mean"]) * 100
        for s in systems
    ]

    x = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))

    rects1 = ax.bar(
        x - width / 2,
        faith_means,
        width,
        yerr=[faith_err_low, faith_err_high],
        label="Faithfulness (%)",
        color="#2E7D32",
        capsize=4,
        alpha=0.9,
    )
    rects2 = ax.bar(
        x + width / 2,
        halluc_means,
        width,
        yerr=[halluc_err_low, halluc_err_high],
        label="Hallucination Rate (%)",
        color="#C62828",
        capsize=4,
        alpha=0.9,
    )

    ax.set_ylabel("Metric Score (%)", fontweight="bold")
    ax.set_title("Faithfulness vs. Hallucination Rate across Systems (with 95% Bootstrap CIs)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_names)
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="white", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Label values
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = PLOTS_DIR / "faithfulness_vs_hallucination.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_category_breakdown(eval_summary: dict) -> Path:
    systems = list(eval_summary["systems"].keys())
    categories = ["simple_factual", "multi_hop", "numeric_calculation", "out_of_scope", "adversarial"]
    cat_labels = ["Simple\nFactual", "Multi-Hop\nReasoning", "Numeric\nCalculation", "Out-of-Scope\n(Abstain)", "Adversarial\n(False Premise)"]

    colors = ["#78909C", "#FFA726", "#42A5F5", "#66BB6A"]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(categories))
    width = 0.18

    for idx, (s_name, color) in enumerate(zip(systems, colors)):
        scores = []
        for cat in categories:
            val = eval_summary["systems"][s_name]["category_breakdown"].get(cat, {}).get("correctness", 0.0) * 100
            scores.append(val)
        offset = (idx - 1.5) * width
        rects = ax.bar(x + offset, scores, width, label=s_name, color=color, alpha=0.9)

    ax.set_ylabel("Answer Correctness (%)", fontweight="bold")
    ax.set_title("Per-Category Answer Correctness across 4 Systems", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, facecolor="white", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

    plt.tight_layout()
    out_path = PLOTS_DIR / "category_breakdown.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_latency_vs_tokens(eval_summary: dict) -> Path:
    systems = list(eval_summary["systems"].keys())
    tokens = [eval_summary["systems"][s]["total_tokens"]["mean"] for s in systems]
    latencies = [eval_summary["systems"][s]["latency_seconds"]["p95"] * 1000 for s in systems] # ms

    colors = ["#E53935", "#FB8C00", "#1E88E5", "#43A047"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, s in enumerate(systems):
        ax.scatter(tokens[i], latencies[i], color=colors[i], marker=markers[i], s=180, label=s, zorder=5)
        ax.annotate(
            f"  {s}\n  ({tokens[i]:.0f} tok, {latencies[i]:.1f}ms)",
            (tokens[i], latencies[i]),
            textcoords="offset points",
            xytext=(10, -5),
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Mean Total Tokens per Query", fontweight="bold")
    ax.set_ylabel("P95 Latency (milliseconds)", fontweight="bold")
    ax.set_title("Computational Efficiency: Token Footprint vs. Latency", fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = PLOTS_DIR / "latency_vs_tokens.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_retrieval_chunking_ablation(retrieval_metrics: dict) -> Path:
    sec = retrieval_metrics["benchmark_summary"]["section_aware"]
    fix = retrieval_metrics["benchmark_summary"]["fixed_size"]

    metrics = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
    sec_scores = [sec["hit@1"] * 100, sec["hit@3"] * 100, sec["hit@5"] * 100, sec["mrr"] * 100]
    fix_scores = [fix["hit@1"] * 100, fix["hit@3"] * 100, fix["hit@5"] * 100, fix["mrr"] * 100]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    r1 = ax.bar(x - width / 2, sec_scores, width, label="Section-Aware Chunking (Proposed)", color="#1976D2", alpha=0.9)
    r2 = ax.bar(x + width / 2, fix_scores, width, label="Fixed-Size Chunking (500 chars)", color="#9E9E9E", alpha=0.9)

    ax.set_ylabel("Score (%) / MRR x 100", fontweight="bold")
    ax.set_title("Retrieval Benchmark: Section-Aware vs. Fixed-Size Chunking", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="white", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    for r in r1:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in r2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = PLOTS_DIR / "retrieval_chunking_ablation.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def main() -> None:
    eval_summary_path = PROJECT_ROOT / "results/eval_summary.json"
    retrieval_metrics_path = PROJECT_ROOT / "results/retrieval_metrics.json"

    with open(eval_summary_path, "r", encoding="utf-8") as f:
        eval_summary = json.load(f)

    with open(retrieval_metrics_path, "r", encoding="utf-8") as f:
        retrieval_metrics = json.load(f)

    p1 = plot_faithfulness_and_hallucination(eval_summary)
    p2 = plot_category_breakdown(eval_summary)
    p3 = plot_latency_vs_tokens(eval_summary)
    p4 = plot_retrieval_chunking_ablation(retrieval_metrics)

    print("=" * 60)
    print("EMPIRICAL PLOTS GENERATED SUCCESSFULLY:")
    print(f"  1. {p1}")
    print(f"  2. {p2}")
    print(f"  3. {p3}")
    print(f"  4. {p4}")
    print("=" * 60)


if __name__ == "__main__":
    main()

