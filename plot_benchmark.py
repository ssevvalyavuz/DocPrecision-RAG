import os
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = os.path.join("evaluation", "ragas_collections_results.json")
OUTPUT_IMG = os.path.join("assets", "benchmark_comparison.png")


def generate_benchmark_chart() -> None:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_sum = data["summary"]["naive_rag"]
    a_sum = data["summary"]["advanced_rag"]
    top_k = data["metadata"].get("top_k", 3)
    total_q = data["metadata"].get("evaluated_questions", 30)

    # Renk paleti (Mühendislik/Teknik rapor estetiği)
    bg_color = "#0F172A"
    panel_color = "#1E293B"
    naive_color = "#64748B"      # Nötr arduvaz grisi (Baseline)
    adv_color = "#10B981"        # Zümrüt yeşili (Advanced RAG kazanımı)
    latency_adv_color = "#38BDF8"  # Açık mavi (Sistem ödünleşimi)
    text_color = "#F8FAFC"
    muted_color = "#94A3B8"
    grid_color = "#334155"

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    fig.patch.set_facecolor(bg_color)

    # --- SOL PANEL: CONTEXT PRECISION & RECALL ---
    ax1.set_facecolor(panel_color)
    metrics = ["Context Precision", "Context Recall"]
    naive_vals = [n_sum["context_precision"], n_sum["context_recall"]]
    adv_vals = [a_sum["context_precision"], a_sum["context_recall"]]

    x = np.arange(len(metrics))
    width = 0.34

    rects1 = ax1.bar(
        x - width / 2, naive_vals, width,
        label="Naive RAG (500-char + Dense)", color=naive_color, edgecolor="none", zorder=3
    )
    rects2 = ax1.bar(
        x + width / 2, adv_vals, width,
        label="Advanced RAG (Semantic + Hybrid + RRF)", color=adv_color, edgecolor="none", zorder=3
    )

    ax1.set_ylim(0, 1.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, color=text_color, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", colors=muted_color, labelsize=10)
    ax1.set_ylabel("Ragas 0.4.3 Score (0.0 – 1.0)", color=muted_color, fontsize=10)
    ax1.set_title(
        f"Retrieval Quality (Top-{top_k}, n={total_q})",
        color=text_color, fontsize=12.5, fontweight="bold", pad=14
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.5, color=grid_color, zorder=0)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax1.spines[spine].set_color(grid_color)

    # Değer etiketleri
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", color=muted_color, fontsize=10, fontweight="bold"
        )

    for idx, rect in enumerate(rects2):
        h = rect.get_height()
        diff_pct = ((adv_vals[idx] - naive_vals[idx]) / naive_vals[idx]) * 100
        ax1.annotate(
            f"{h:.4f}\n(+{diff_pct:.1f}%)",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", color=adv_color, fontsize=10, fontweight="bold"
        )

    ax1.legend(
        loc="lower right", frameon=True, facecolor=bg_color,
        edgecolor=grid_color, labelcolor=text_color, fontsize=9.5
    )

    # --- SAĞ PANEL: RETRIEVAL LATENCY TRADE-OFF ---
    ax2.set_facecolor(panel_color)
    systems = ["Naive RAG\nBaseline", "Advanced RAG\nPipeline"]
    latencies = [n_sum["avg_retrieval_latency"], a_sum["avg_retrieval_latency"]]
    bar_colors = [naive_color, latency_adv_color]

    bars = ax2.bar(systems, latencies, width=0.45, color=bar_colors, zorder=3)
    ax2.set_ylim(0, max(latencies) * 1.32)
    ax2.set_xticks(range(len(systems)))
    ax2.set_xticklabels(systems, color=text_color, fontsize=10.5, fontweight="bold")
    ax2.tick_params(axis="y", colors=muted_color, labelsize=10)
    ax2.set_ylabel("Avg. Latency per Query (sec)", color=muted_color, fontsize=10)
    ax2.set_title(
        "Retrieval Latency Trade-off",
        color=text_color, fontsize=12.5, fontweight="bold", pad=14
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.5, color=grid_color, zorder=0)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax2.spines[spine].set_color(grid_color)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.4f} s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", color=text_color, fontsize=10.5, fontweight="bold"
        )

    fig.suptitle(
        "DocPrecision-RAG — Retrieval Architecture Benchmark (Ragas 0.4.3 Collections)",
        color=text_color, fontsize=14.5, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300, facecolor=fig.get_facecolor())
    print(f"Grafik kaydedildi: {OUTPUT_IMG}")


if __name__ == "__main__":
    generate_benchmark_chart()