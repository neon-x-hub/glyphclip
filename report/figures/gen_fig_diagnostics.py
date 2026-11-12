"""Generate Figure 5: Symmetrical 2-Channel Vector-SDF & Polarity/Pixel Visual Diagnostics."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from glyphclip.canvas.sdf import VectorSDFRenderer, compute_edge_orthogonality
from glyphclip.pipeline.binarization import adaptive_binarize
from glyphclip.pipeline.segmentation import VisualSegmentationPipeline


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def main():
    sdf = VectorSDFRenderer()
    seg = VisualSegmentationPipeline(include_sdf=True, min_aspect=0.06)

    # Row 1: Canonical Vector-SDF on Funnel Display
    funnel_ttf = Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\FunnelDisplay-VariableFont_wght.ttf")
    if not funnel_ttf.exists():
        funnel_ttf = Path(r"C:\Windows\Fonts\arial.ttf")
    canon_dual = sdf.render_dual_tensor(funnel_ttf, "R", weight=600).numpy()
    canon_occ, canon_tsdf = canon_dual[0], canon_dual[1]
    canon_ortho = compute_edge_orthogonality(canon_occ)

    # Row 2: Ram-Altman@Funnel.png (white on dark green)
    ram_path = Path("Ram-Altman@Funnel.png")
    ram_bgr = cv2.imread(str(ram_path))
    ram_rgb = cv2.cvtColor(ram_bgr, cv2.COLOR_BGR2RGB)
    ram_gray = cv2.cvtColor(ram_bgr, cv2.COLOR_BGR2GRAY)
    h, w = ram_gray.shape[:2]
    eff_w = max(25, min(151, (min(h, w) // 3) | 1))
    ram_bin, _ = adaptive_binarize(ram_gray, window_size=eff_w)
    ram_crops = seg.process(ram_bgr)
    ram_r = ram_crops[0].canvas_tensor.numpy()
    ram_ortho = compute_edge_orthogonality(ram_r[0])

    # Row 3: ENSIA@PixelOperator8.png (pixel/bitmap font)
    ensia_path = Path("ENSIA@PixelOperator8.png")
    ensia_bgr = cv2.imread(str(ensia_path))
    ensia_rgb = cv2.cvtColor(ensia_bgr, cv2.COLOR_BGR2RGB)
    ensia_gray = cv2.cvtColor(ensia_bgr, cv2.COLOR_BGR2GRAY)
    eh, ew = ensia_gray.shape[:2]
    ensia_bin, _ = adaptive_binarize(ensia_gray, window_size=max(25, min(151, (min(eh, ew) // 3) | 1)))
    ensia_crops = seg.process(ensia_bgr)
    ensia_e = ensia_crops[0].canvas_tensor.numpy()
    ensia_ortho = compute_edge_orthogonality(ensia_e[0])

    fig, axes = plt.subplots(3, 4, figsize=(10.8, 7.6))
    plt.subplots_adjust(wspace=0.14, hspace=0.34)

    # Row 1 plots
    axes[0, 0].axis("off")
    axes[0, 0].text(0.5, 0.52, "Canonical Font\n(.ttf / .otf)\n\nFunnel Display\nWeight: 600\nChar: 'R'",
                    ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F7F6", edgecolor="#2A9D8F"))
    axes[0, 0].set_title("1. Analytical Source")

    axes[0, 1].imshow(canon_occ, cmap="gray")
    axes[0, 1].set_title("Ch 0: 4x Winding Occ.")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(canon_tsdf, cmap="RdBu_r", vmin=0, vmax=1)
    axes[0, 2].set_title("Ch 1: Smooth TSDF (sig=8)")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(np.abs(canon_tsdf - 0.5) < 0.08, cmap="viridis")
    axes[0, 3].set_title(f"Zero Level Set (ortho={canon_ortho:.2f})")
    axes[0, 3].axis("off")

    # Row 2 plots
    axes[1, 0].imshow(ram_rgb)
    axes[1, 0].set_title("Dark Banner Input")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(ram_bin, cmap="gray")
    axes[1, 1].set_title("Adaptive Polarity Binarize")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(ram_r[0], cmap="gray")
    axes[1, 2].set_title("Segmented Ch 0 ('R')")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(ram_r[1], cmap="RdBu_r", vmin=0, vmax=1)
    axes[1, 3].set_title(f"Segmented TSDF (ortho={ram_ortho:.2f})")
    axes[1, 3].axis("off")

    # Row 3 plots
    axes[2, 0].imshow(ensia_rgb)
    axes[2, 0].set_title("Unindexed Bitmap Query")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(ensia_bin, cmap="gray")
    axes[2, 1].set_title("Pixel-Preserving Mask")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(ensia_e[0], cmap="gray")
    axes[2, 2].set_title("Segmented Ch 0 ('E')")
    axes[2, 2].axis("off")

    axes[2, 3].imshow(ensia_e[1], cmap="RdBu_r", vmin=0, vmax=1)
    axes[2, 3].set_title(f"Bitmap TSDF (ortho={ensia_ortho:.2f})")
    axes[2, 3].axis("off")

    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig5_visual_diagnostics.pdf")
    fig.savefig(out_dir / "fig5_visual_diagnostics.png", dpi=300)
    plt.close(fig)
    print(f"Saved fig5_visual_diagnostics.pdf/.png (canon_ortho={canon_ortho:.3f}, ram_ortho={ram_ortho:.3f}, ensia_ortho={ensia_ortho:.3f})")


if __name__ == "__main__":
    main()
