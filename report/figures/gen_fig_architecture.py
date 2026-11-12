"""Generate Figure 1: Symmetrical Dual-Domain Vector-SDF Architecture (vector PDF + 300 DPI PNG)."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def add_box(ax, x, y, w, h, title, subtitle="", facecolor="#FFFFFF", edgecolor="#264653",
            lw=1.4, title_color="#1E293B", sub_color="#475569"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=3
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=title_color, zorder=4)
        ax.text(x + w / 2, y + h * 0.31, subtitle, ha="center", va="center",
                fontsize=6.7, color=sub_color, linespacing=1.28, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=title_color, zorder=4)


def add_arrow(ax, x1, y1, x2, y2, color="#264653", lw=1.5, ls="-", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=5,head_width=3",
        color=color, linewidth=lw, linestyle=ls, zorder=2
    )
    ax.add_patch(arr)


def main():
    fig, ax = plt.subplots(figsize=(13.2, 5.5))
    ax.set_xlim(0, 18.0)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    # Background swimlanes (generous horizontal margin so headers never cross boundaries)
    canon_bg = FancyBboxPatch((0.2, 3.95), 9.0, 3.15, boxstyle="round,pad=0.05,rounding_size=0.22",
                              facecolor="#F0F7F6", edgecolor="#2A9D8F", linewidth=1.1, linestyle="--", alpha=0.85, zorder=1)
    vis_bg = FancyBboxPatch((0.2, 0.25), 9.0, 3.15, boxstyle="round,pad=0.05,rounding_size=0.22",
                            facecolor="#FFF8F3", edgecolor="#E76F51", linewidth=1.1, linestyle="--", alpha=0.85, zorder=1)
    ret_bg = FancyBboxPatch((9.6, 0.25), 8.2, 6.85, boxstyle="round,pad=0.05,rounding_size=0.22",
                            facecolor="#F8FAFC", edgecolor="#264653", linewidth=1.1, alpha=0.95, zorder=1)
    for p in (canon_bg, vis_bg, ret_bg):
        ax.add_patch(p)

    ax.text(0.45, 6.82, "CANONICAL VECTOR DOMAIN (Zero-Shot Indexing & Training)",
            fontsize=8.0, fontweight="bold", color="#1D6F63", zorder=4)
    ax.text(0.45, 3.12, "VISUAL INFERENCE DOMAIN (Real Crops & Online Degradation)",
            fontsize=8.0, fontweight="bold", color="#C8553D", zorder=4)
    ax.text(9.85, 6.82, "SHARED METRIC SPACE, ANN INDEX & CONSENSUS",
            fontsize=8.0, fontweight="bold", color="#264653", zorder=4)

    # Top row (Canonical Domain)
    add_box(ax, 0.45, 4.35, 2.50, 2.00, "Font Source",
            ".ttf / .otf / fvar Axes\nOS/2 Weight (100–900)\nBasePen Spline Contours",
            facecolor="#FFFFFF", edgecolor="#2A9D8F")
    add_box(ax, 3.40, 4.35, 2.75, 2.00, "VectorSDFRenderer",
            "Adaptive Bézier Subdiv.\n4x Non-Zero Winding\nExact L2 TSDF ($\\sigma=8.0$)",
            facecolor="#FFFFFF", edgecolor="#2A9D8F")
    add_box(ax, 6.60, 4.35, 2.30, 2.00, "Canonical Tensor",
            "$[2, 224, 224]$ Dual Grid\nOccupancy + TSDF\n+ Canonical ortho_ratio",
            facecolor="#E6F4F1", edgecolor="#2A9D8F")

    add_arrow(ax, 2.97, 5.35, 3.38, 5.35, color="#2A9D8F")
    add_arrow(ax, 6.17, 5.35, 6.58, 5.35, color="#2A9D8F")

    # Bottom row (Visual Domain)
    add_box(ax, 0.45, 0.65, 2.50, 2.00, "Visual Query / Crop",
            "Photo / Web Crop\nor Multi-Scale Morph ($k$=3..9)\nPixel-Preserving Deg.",
            facecolor="#FFFFFF", edgecolor="#E76F51")
    add_box(ax, 3.40, 0.65, 2.75, 2.00, "Segmentation & TSDF",
            "Deskew + Otsu Polarity\nScale-Adaptive Sauvola\n168px Norm + L2 TSDF",
            facecolor="#FFFFFF", edgecolor="#E76F51")
    add_box(ax, 6.60, 0.65, 2.30, 2.00, "Visual Tensor",
            "$[N, 2, 224, 224]$ Crops\nOccupancy + TSDF\n+ Query ortho_ratio",
            facecolor="#FDECE6", edgecolor="#E76F51")

    add_arrow(ax, 2.97, 1.65, 3.38, 1.65, color="#E76F51")
    add_arrow(ax, 6.17, 1.65, 6.58, 1.65, color="#E76F51")

    # Central Shared Backbone
    add_box(ax, 9.95, 2.55, 3.15, 2.30, "Shared GlyphConvNeXt",
            "20.58M Params (--scratch)\n4 Stages (80–640 dims)\nWide Cross-Wt ($\\Delta w \\leq 400$)\nGAP + LayerNorm $\\to h \\in \\mathbb{R}^{640}$",
            facecolor="#264653", edgecolor="#1B323C", title_color="#FFFFFF", sub_color="#E2E8F0")

    add_arrow(ax, 8.92, 5.35, 9.93, 4.10, color="#2A9D8F", lw=1.7, rad=-0.12)
    add_arrow(ax, 8.92, 1.65, 9.93, 3.25, color="#E76F51", lw=1.7, rad=0.12)

    # Right column: Metric Projection & Store, Output Pill, Blind OCR & Consensus
    add_box(ax, 13.70, 4.65, 3.80, 1.85, "Metric Projection & ANN Store",
            "Proj(640$\\to$1024$\\to$512) $\\to \\mathbb{S}^{511}$\nGlyphVectorStore (IVF-Cosine ANN)\n$O(1)$ Char Buckets + Persistence",
            facecolor="#FFFFFF", edgecolor="#264653")
    add_box(ax, 13.70, 0.65, 3.80, 1.85, "Blind OCR Head & Consensus",
            "char_head (640$\\to$52) Top-3 Filter\nBayesianConsensusEngine:\nEntropy $\\times$ Missing $\\times$ ortho_ratio",
            facecolor="#FFFFFF", edgecolor="#D97706")
    add_box(ax, 13.70, 2.92, 3.80, 1.35, "Ranked Typeface Matches",
            "Top-k Families, Weights & Confidence\n(11/11 Prod. Acc | ~50–110 ms/query)",
            facecolor="#E9C46A", edgecolor="#B7791F")

    add_arrow(ax, 13.12, 4.10, 13.68, 5.45, color="#264653", lw=1.6)
    add_arrow(ax, 13.12, 3.20, 13.68, 1.60, color="#D97706", lw=1.6)
    add_arrow(ax, 15.60, 4.63, 15.60, 4.29, color="#264653", lw=1.5)
    add_arrow(ax, 15.60, 2.52, 15.60, 2.90, color="#D97706", lw=1.5)

    # InfoNCE Bridge pill
    ax.text(11.52, 5.65, "Symmetric InfoNCE\n($\\tau = 0.0390$)", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#264653",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#FFFDF7", edgecolor="#CBD5E1"))

    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig1_architecture.pdf")
    fig.savefig(out_dir / "fig1_architecture.png", dpi=300)
    plt.close(fig)
    print("Saved fig1_architecture.pdf and .png")


if __name__ == "__main__":
    main()
