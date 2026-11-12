"""Generate Figures 2, 3, and 4 from docs/ablation_results.json for the technical report."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
})

OUR_COLOR = "#E76F51"       # Burnt coral
MID_COLOR = "#2A9D8F"       # Teal
BASELINE_COLOR = "#264653"  # Deep slate
GRAY_COLOR = "#94A3B8"      # Light slate


def plot_fig2_benchmarks(data: dict, out_dir: Path):
    """Figure 2: Zero-shot held-out retrieval on 106 unseen fonts (v1.1 vs v2.2 vs v3.0)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    plt.subplots_adjust(wspace=0.28)

    models_data = data["ablation_zero_shot"]["models"]
    models = ["v1_1_raster_1ch", "v2_2_vectorsdf_2ch", "v3_0_vectorsdf_scratch"]
    colors = [BASELINE_COLOR, MID_COLOR, OUR_COLOR]

    # Panel A: Text-Aligned Top-1 and MRR
    top1_aligned = [models_data[m]["text_aligned"]["top1"] for m in models]
    mrr_aligned = [models_data[m]["text_aligned"]["mrr"] * 100.0 for m in models]

    # Panel B: Blind Auto-Detection Top-1 and MRR
    top1_blind = [models_data[m]["blind_auto"]["top1"] for m in models]
    mrr_blind = [models_data[m]["blind_auto"]["mrr"] * 100.0 for m in models]

    x = np.arange(len(models))
    w = 0.35

    # Panel 1: Text-Aligned
    bars1 = ax1.bar(x - w / 2, top1_aligned, w * 0.90, label="Top-1 Acc (%)", color=colors, edgecolor="white", linewidth=0.6)
    bars2 = ax1.bar(x + w / 2, mrr_aligned, w * 0.90, label="MRR (x100)", color=colors, alpha=0.52, hatch="//", edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["v1.1\n(1-ch Raster)", "v2.2\n(SDF Warm)", "v3.0\n(SDF Scratch)"], fontsize=8.2)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 115)
    ax1.set_title("(a) Text-Aligned Retrieval (106 Test Fonts)")
    ax1.legend(loc="upper left", fontsize=7.8)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.8,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    # Panel 2: Blind Auto-Detection
    bars3 = ax2.bar(x - w / 2, top1_blind, w * 0.90, label="Blind Top-1 (%)", color=colors, edgecolor="white", linewidth=0.6)
    bars4 = ax2.bar(x + w / 2, mrr_blind, w * 0.90, label="Blind MRR (x100)", color=colors, alpha=0.52, hatch="//", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["v1.1\n(1-ch Raster)", "v2.2\n(SDF Warm)", "v3.0\n(SDF Scratch)"], fontsize=8.2)
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 122)
    ax2.set_title("(b) Blind Auto-Detection (Zero Text Hint)")
    ax2.legend(loc="upper left", fontsize=7.8)

    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.8,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    fig.savefig(out_dir / "fig2_benchmarks.pdf")
    fig.savefig(out_dir / "fig2_benchmarks.png", dpi=300)
    plt.close(fig)
    print("Saved fig2_benchmarks.pdf and .png")


def plot_fig3_weight_gap(data: dict, out_dir: Path):
    """Figure 3: Cross-weight jump sensitivity (Delta-W: 0, 200, 300, 400)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))

    cw_data = data["ablation_cross_weight"]["models"]
    x_pos = [0, 1, 2, 3]
    keys = ["delta_0", "delta_200", "delta_300", "delta_400"]

    lines = [
        ("v1.1 (1-ch Raster)", cw_data.get("v1_1_raster_1ch", {}), BASELINE_COLOR, "s", "--"),
        ("v2.2 (2-ch SDF, gap<=200)", cw_data.get("v2_2_vectorsdf_2ch", {}), MID_COLOR, "^", "-."),
        ("v3.0 (2-ch SDF, gap<=400)", cw_data.get("v3_0_vectorsdf_scratch", {}), OUR_COLOR, "o", "-"),
    ]

    for label, mdata, color, marker, ls in lines:
        if not mdata:
            continue
        vals = [mdata[k]["blind_top1"] for k in keys if k in mdata]
        ax.plot(x_pos[:len(vals)], vals, label=label, color=color, marker=marker,
                linestyle=ls, markersize=6, linewidth=1.9, zorder=4)
        for xp, v in zip(x_pos[:len(vals)], vals):
            offset_y = 6 if label.startswith("v3") else (-11 if label.startswith("v1") else 6)
            ax.annotate(f"{v:.1f}%", (xp, v), textcoords="offset points", xytext=(0, offset_y),
                        ha="center", fontsize=7.5, color=color, fontweight="bold")

    ax.set_xlabel(r"Stroke Weight Gap ($\Delta w = |w_{\mathrm{query}} - w_{\mathrm{index}}|$)")
    ax.set_ylabel("Blind Top-1 Accuracy (%)")
    ax.set_title("Cross-Weight Invariance vs. Weight Jump (30 Families)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r"$\Delta 0$" + "\n(Same Wt)", r"$\Delta 200$" + "\n(Med $\\to$ Bold)",
                        r"$\Delta 300$" + "\n(Reg $\\to$ Bold)", r"$\Delta 400$" + "\n(Light $\\to$ Bold)"])
    ax.set_xlim(-0.25, 3.25)
    ax.set_ylim(15, 110)
    ax.legend(loc="lower left", fontsize=8.0)

    fig.savefig(out_dir / "fig3_weight_gap_ablation.pdf")
    fig.savefig(out_dir / "fig3_weight_gap_ablation.png", dpi=300)
    plt.close(fig)
    print("Saved fig3_weight_gap_ablation.pdf and .png")


def plot_fig4_ann_scaling(data: dict, out_dir: Path):
    """Figure 4: Vector store efficiency & recall across catalog sizes (Exact vs IVF-Cosine)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    plt.subplots_adjust(wspace=0.32)

    benchmarks = data["ablation_ann_scaling"]["benchmarks"]
    sizes = [b["n_vecs"] for b in benchmarks]
    exact_unf = [b["exact_unfiltered_ms"] for b in benchmarks]
    exact_fil = [b["exact_char_filtered_ms"] for b in benchmarks]

    ann_25_unf = [b["ann_configs"]["nprobe_0.25"]["unfiltered_ms"] for b in benchmarks]
    ann_25_fil = [b["ann_configs"]["nprobe_0.25"]["char_filtered_ms"] for b in benchmarks]

    # Panel 1: Latency scaling
    ax1.plot(sizes, exact_unf, label="Exact Unfiltered (Linear)", color=GRAY_COLOR, linestyle="--", marker="x")
    ax1.plot(sizes, ann_25_unf, label="IVF-Cosine Unfiltered", color=MID_COLOR, marker="^")
    ax1.plot(sizes, exact_fil, label="Exact Char-Filtered $O(1)$", color=BASELINE_COLOR, linestyle="--", marker="s")
    ax1.plot(sizes, ann_25_fil, label="IVF-Cosine + Char Bucket", color=OUR_COLOR, marker="o", linewidth=2.2)

    ax1.set_xlabel("Catalog Size (Indexed Glyph Vectors)")
    ax1.set_ylabel("Query Latency (ms / glyph)")
    ax1.set_title("(a) Retrieval Latency vs Index Size")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left", fontsize=7.6)

    # Panel 2: Recall@1 vs Recall@5
    rec_r1 = [b["ann_configs"]["nprobe_0.25"]["char_filtered_recall1"] for b in benchmarks]
    rec_r5 = [b["ann_configs"]["nprobe_0.25"]["char_filtered_recall5"] for b in benchmarks]
    rec_unf1 = [b["ann_configs"]["nprobe_0.25"]["unfiltered_recall1"] for b in benchmarks]

    ax2.plot(sizes, rec_r1, label="Recall@1 (Char-Filtered)", color=OUR_COLOR, marker="o", linewidth=2.0)
    ax2.plot(sizes, rec_r5, label="Recall@5 (Char-Filtered)", color=MID_COLOR, marker="^", linewidth=2.0)
    ax2.plot(sizes, rec_unf1, label="Recall@1 (Unfiltered)", color=BASELINE_COLOR, marker="s", linestyle="--")

    ax2.set_xlabel("Catalog Size (Indexed Glyph Vectors)")
    ax2.set_ylabel("Search Recall vs Exact (%)")
    ax2.set_title("(b) IVF-Cosine Approximation Recall (nprobe=25%)")
    ax2.set_ylim(85, 102)
    ax2.legend(loc="lower left", fontsize=8.0)

    fig.savefig(out_dir / "fig4_ann_efficiency.pdf")
    fig.savefig(out_dir / "fig4_ann_efficiency.png", dpi=300)
    plt.close(fig)
    print("Saved fig4_ann_efficiency.pdf and .png")


def main():
    res_path = Path("docs/ablation_results.json")
    if not res_path.exists():
        print(f"Error: {res_path} not found. Run scripts/run_ablations.py first.")
        return
    data = json.loads(res_path.read_text(encoding="utf-8"))
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_fig2_benchmarks(data, out_dir)
    plot_fig3_weight_gap(data, out_dir)
    plot_fig4_ann_scaling(data, out_dir)
    print("All quantitative figures generated successfully!")


if __name__ == "__main__":
    main()
