"""Run Leave-One-Out Top-K Visual & Geometric Fidelity Benchmark for GlyphCLIP.

When the exact target font family is absent from the index (e.g. proprietary or custom brand fonts),
retrieval models must recommend the closest visual and geometric substitute.

This script evaluates:
  1. Mean Glyph IoU (mIoU@K) over 52-character canonical alphabet masks.
  2. Symmetric Boundary Chamfer Distance (CD@K, in pixels) on zero-level set contours.
  3. Continuous TSDF-L1 distance (TSDF-L1@K) across 224x224 distance fields.
  4. Perceptual Specimen SSIM (SSIM@K) on the classic 'Hamburgefonstiv' typographic test string.

Compares:
  - Random Heuristic Baseline
  - v1.1 Scaled 1-ch FreeType Raster Baseline
  - GlyphCLIP v3.0 Vector-SDF (Ours)

Outputs:
  - docs/substitute_benchmark_results.json
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.sdf import VectorSDFRenderer
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.pipeline.dataset import FontDatasetSplitter
from glyphclip.pipeline.evaluator import IndependentTestHarness


FONT_DIRS = [
    Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts"),
    Path(r"C:\Windows\Fonts"),
]


def load_encoder(ckpt_path: Path, in_channels: int, num_chars: int, device: torch.device) -> DualEncoder:
    model = DualEncoder(
        embed_dim=512,
        backbone_type="convnext",
        shared_backbone=True,
        in_channels=in_channels,
        num_chars=num_chars,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def compute_ssim(img1: np.ndarray, img2: np.ndarray, k1: float = 0.01, k2: float = 0.03) -> float:
    """Compute SSIM between two grayscale images in [0, 1]."""
    c1 = (k1 * 1.0) ** 2
    c2 = (k2 * 1.0) ** 2
    mu1 = gaussian_filter(img1, 1.5)
    mu2 = gaussian_filter(img2, 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 ** 2, 1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, 1.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-7
    )
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def render_specimen_image(font_path: Path, text: str = "Hamburgefonstiv", size: int = 44, width: int = 540, height: int = 72) -> np.ndarray:
    """Render a standard typographic specimen string at fixed dimensions."""
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(str(font_path), size=size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w_t = bbox[2] - bbox[0]
        h_t = bbox[3] - bbox[1]
        x = max(8, (width - w_t) // 2)
        y = max(8, (height - h_t) // 2)
        draw.text((x, y), text, font=font, fill=0)
    except Exception:
        pass
    return np.array(img, dtype=np.float32) / 255.0


class GeometricFidelityEvaluator:
    """Caches 52-glyph canonical geometry for fast objective metric evaluation."""

    def __init__(self, font_paths: List[Path], charset_subset: str = "alpha"):
        self.font_paths = font_paths
        self.charset = get_charset(charset_subset)
        self.sdf_renderer = VectorSDFRenderer(canvas_size=224)
        self.kernel = np.ones((3, 3), np.uint8)

        # Map family -> font_path
        self.family_to_path: Dict[str, Path] = {}
        for p in font_paths:
            try:
                m = FontParser.parse(p)
                fam_key = FontDatasetSplitter.normalize_family_key(p, m.family)
                if fam_key not in self.family_to_path:
                    self.family_to_path[fam_key] = p
            except Exception:
                continue

        # Precompute geometry cache: family -> char -> {mask, tsdf, boundary, dist}
        print(f"Precomputing canonical 52-glyph geometric cache for {len(self.family_to_path)} families...")
        t0 = time.time()
        self.cache: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self.specimen_cache: Dict[str, np.ndarray] = {}

        for fam, p in self.family_to_path.items():
            fam_geom = {}
            for ch in self.charset:
                dual = self.sdf_renderer.render_dual_numpy(p, ch)
                mask = (dual[0] >= 0.5)
                tsdf = dual[1]
                b = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, self.kernel) > 0
                dist = cv2.distanceTransform((~b).astype(np.uint8) * 255, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                fam_geom[ch] = {"mask": mask, "tsdf": tsdf, "boundary": b, "dist": dist}
            self.cache[fam] = fam_geom
            self.specimen_cache[fam] = render_specimen_image(p)

        print(f"Geometry precomputation complete in {time.time() - t0:.2f}s.")

    def compute_pair_metrics(self, fam_a: str, fam_b: str) -> Dict[str, float]:
        """Compute mIoU, CD (px), TSDF-L1, and specimen SSIM between two font families."""
        if fam_a not in self.cache or fam_b not in self.cache:
            return {"miou": 0.0, "chamfer": 15.0, "tsdf_l1": 0.20, "ssim": 0.0}

        geom_a = self.cache[fam_a]
        geom_b = self.cache[fam_b]

        ious: List[float] = []
        cds: List[float] = []
        l1s: List[float] = []

        for ch in self.charset:
            ga = geom_a[ch]
            gb = geom_b[ch]

            # IoU
            inter = np.logical_and(ga["mask"], gb["mask"]).sum()
            union = np.logical_or(ga["mask"], gb["mask"]).sum()
            ious.append(inter / union if union > 0 else 1.0)

            # Chamfer Distance (px)
            pts_a = ga["boundary"].sum()
            pts_b = gb["boundary"].sum()
            if pts_a > 0 and pts_b > 0:
                cd = 0.5 * (gb["dist"][ga["boundary"]].mean() + ga["dist"][gb["boundary"]].mean())
                cds.append(float(cd))
            elif pts_a == 0 and pts_b == 0:
                cds.append(0.0)
            else:
                cds.append(10.0)

            # TSDF-L1
            l1s.append(float(np.abs(ga["tsdf"] - gb["tsdf"]).mean()))

        # Specimen SSIM
        spec_ssim = compute_ssim(self.specimen_cache[fam_a], self.specimen_cache[fam_b])

        return {
            "miou": float(np.mean(ious)),
            "chamfer": float(np.mean(cds)),
            "tsdf_l1": float(np.mean(l1s)),
            "ssim": spec_ssim,
        }


def run_substitute_benchmark(device: torch.device) -> Dict[str, Any]:
    print("\n=======================================================")
    print("=== Leave-One-Out Top-K Visual & Geometric Benchmark ===")
    print("=======================================================")

    _, _, test_fonts = FontDatasetSplitter.get_font_splits(
        FONT_DIRS,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        max_total_fonts=None,
        max_per_family=4,
        seed=42,
    )
    print(f"Loaded {len(test_fonts)} held-out test font files.")

    # Build geometric fidelity evaluator
    geom_eval = GeometricFidelityEvaluator(test_fonts, charset_subset="alpha")
    all_test_fams = sorted(list(geom_eval.family_to_path.keys()))
    print(f"Unique test families available for Leave-One-Out: {len(all_test_fams)}")

    test_words = ["Graphic", "Typography", "Sample", "Geometric"]

    models_config = [
        ("v1_1_raster_1ch", Path("checkpoints/glyphclip_scaled.pt"), 1, 0),
        ("v3_0_vectorsdf_scratch", Path("checkpoints/glyphclip_v3.pt"), 2, 52),
    ]

    all_results = {
        "num_test_fonts": len(test_fonts),
        "num_test_families": len(all_test_fams),
        "metrics": {},
        "qualitative_cases": [],
    }

    # First evaluate Random Heuristic Baseline
    random.seed(42)
    rand_metrics = defaultdict(list)
    for fam_gt in all_test_fams:
        pool = [f for f in all_test_fams if f != fam_gt]
        if not pool:
            continue
        choices = random.sample(pool, min(5, len(pool)))
        pair_scores = [geom_eval.compute_pair_metrics(fam_gt, f) for f in choices]

        for k in [1, 3, 5]:
            top_k_scores = pair_scores[:k]
            rand_metrics[f"miou@{k}"].append(np.mean([s["miou"] for s in top_k_scores]))
            rand_metrics[f"chamfer@{k}"].append(np.mean([s["chamfer"] for s in top_k_scores]))
            rand_metrics[f"tsdf_l1@{k}"].append(np.mean([s["tsdf_l1"] for s in top_k_scores]))
            rand_metrics[f"ssim@{k}"].append(np.mean([s["ssim"] for s in top_k_scores]))

    all_results["metrics"]["random_baseline"] = {
        k: round(float(np.mean(v)), 4) for k, v in rand_metrics.items()
    }
    print(f"Random Baseline: {json.dumps(all_results['metrics']['random_baseline'], indent=2)}", flush=True)

    # Evaluate neural models
    for model_name, ckpt_path, in_ch, num_chars in models_config:
        if not ckpt_path.exists():
            print(f"Skipping {model_name}: {ckpt_path} not found", flush=True)
            continue

        print(f"\n--- Evaluating {model_name} ({ckpt_path}) under Leave-One-Out ---", flush=True)
        encoder = load_encoder(ckpt_path, in_ch, num_chars, device)
        harness = IndependentTestHarness(
            dual_encoder=encoder,
            test_font_paths=test_fonts,
            device=device,
            embed_dim=512,
            charset_subset="alpha",
        )

        model_metrics = defaultdict(list)
        stored_cases = []

        print(f"Running queries for {model_name}...", flush=True)

        # Run queries
        for fpath in test_fonts:
            try:
                meta = FontParser.parse(fpath)
                fam_gt = FontDatasetSplitter.normalize_family_key(fpath, meta.family)
                if fam_gt not in geom_eval.family_to_path:
                    continue

                for word in test_words[:2]:  # 2 words per font = ~212 queries
                    q_img = harness.render_word_image(fpath, word=word, size=72)
                    _, preds = harness.evaluate_word_query(
                        q_img, ground_truth_family=meta.family, top_k=20, query_text=None
                    )

                    # LEAVE-ONE-OUT FILTER: remove ground-truth family from predictions
                    substitutes = []
                    seen_fams = set()
                    for p in preds:
                        p_fam_key = FontDatasetSplitter.normalize_family_key(Path(p.family), p.family)
                        if p_fam_key != fam_gt and p_fam_key in geom_eval.family_to_path and p_fam_key not in seen_fams:
                            substitutes.append(p_fam_key)
                            seen_fams.add(p_fam_key)
                        if len(substitutes) >= 5:
                            break

                    if not substitutes:
                        continue

                    # Compute geometric fidelity for substitutes
                    pair_scores = [geom_eval.compute_pair_metrics(fam_gt, sub) for sub in substitutes]

                    for k in [1, 3, 5]:
                        top_k_scores = pair_scores[:min(k, len(pair_scores))]
                        model_metrics[f"miou@{k}"].append(np.mean([s["miou"] for s in top_k_scores]))
                        model_metrics[f"chamfer@{k}"].append(np.mean([s["chamfer"] for s in top_k_scores]))
                        model_metrics[f"tsdf_l1@{k}"].append(np.mean([s["tsdf_l1"] for s in top_k_scores]))
                        model_metrics[f"ssim@{k}"].append(np.mean([s["ssim"] for s in top_k_scores]))

                    # Save interesting qualitative cases for Figure 6
                    if len(stored_cases) < 8 and model_name == "v3_0_vectorsdf_scratch":
                        stored_cases.append({
                            "gt_family": meta.family,
                            "gt_key": fam_gt,
                            "word": word,
                            "top_substitutes": [
                                {
                                    "rank": i + 1,
                                    "sub_key": sub,
                                    "miou": round(pair_scores[i]["miou"], 4),
                                    "chamfer": round(pair_scores[i]["chamfer"], 2),
                                    "ssim": round(pair_scores[i]["ssim"], 4),
                                }
                                for i, sub in enumerate(substitutes[:3])
                            ],
                        })
            except Exception as e:
                import traceback
                print(f"Error querying {fpath.name}: {e}", flush=True)
                traceback.print_exc()
                continue

        all_results["metrics"][model_name] = {
            k: round(float(np.mean(v)), 4) for k, v in model_metrics.items()
        }
        if stored_cases:
            all_results["qualitative_cases"] = stored_cases

        print(f"Results for {model_name}:")
        print(json.dumps(all_results["metrics"][model_name], indent=2))

    # Save to docs/substitute_benchmark_results.json
    out_path = PROJECT_ROOT / "docs" / "substitute_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved structured substitute benchmark results to: {out_path}")

    return all_results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run_substitute_benchmark(device)
