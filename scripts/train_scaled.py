"""Scaled ~20.5M ConvNeXt training harness with hard-negative batching and validation."""

import argparse
import random
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from tqdm import tqdm

from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.renderer import DeterministicCanvasRenderer
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.models.loss import InfoNCELoss
from glyphclip.pipeline.dataset import FontDatasetSplitter
from glyphclip.pipeline.degradation import TypographicDegradationPipeline
from glyphclip.pipeline.evaluator import IndependentTestHarness


def pre_render_font_catalog(
    font_paths: List[Path],
    charset: List[str],
    renderer: DeterministicCanvasRenderer,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], List[dict]]:
    """Pre-render canonical glyphs indexed by [font_id][char] -> Tensor [1, 224, 224]."""
    catalog = {}
    metas = []

    print(f"Pre-rendering {len(font_paths)} fonts over {len(charset)} glyphs...")
    t0 = time.perf_counter()

    for fpath in tqdm(font_paths, desc="Rendering catalog"):
        try:
            meta = FontParser.parse(fpath)
            font_id = meta.postscript_name
            font_dict = {}

            for ch in charset:
                arr = renderer.render_glyph_numpy(fpath, ch)
                if arr.max() < 80:  # Skip blank glyphs
                    continue
                # Store as uint8 tensor [1, 224, 224]
                t = torch.from_numpy(arr).unsqueeze(0)
                font_dict[ch] = t
                metas.append({
                    "family": meta.family,
                    "postscript_name": meta.postscript_name,
                    "weight": meta.weight,
                    "glyph_char": ch,
                    "source_path": str(fpath),
                })

            if len(font_dict) >= len(charset) * 0.8:  # Good coverage
                catalog[font_id] = font_dict
        except Exception:
            continue

    elapsed = time.perf_counter() - t0
    total_glyphs = sum(len(d) for d in catalog.values())
    print(f"Rendered {total_glyphs} valid glyphs across {len(catalog)} fonts in {elapsed:.1f}s.")
    return catalog, metas


def train_scaled(
    font_dirs: List[Path],
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 3e-4,
    embed_dim: int = 512,
    max_fonts: int = 250,
    output_dir: Path = Path("./checkpoints"),
    index_output_dir: Path = Path("./font_index"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    index_output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"GlyphCLIP Scaled ConvNeXt Training (~20.5M Parameters)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Embedding Space: {embed_dim}-d on unit hypersphere S^{embed_dim-1}")
    print(f"=======================================================\n")

    # 1. Independent Dataset Splitting
    train_fonts, val_fonts, test_fonts = FontDatasetSplitter.get_font_splits(
        font_dirs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, max_total_fonts=max_fonts
    )
    print(f"Dataset Split: {len(train_fonts)} Train | {len(val_fonts)} Validation | {len(test_fonts)} Test")

    # Character set: standard alphanumeric letters (A-Z, a-z)
    charset = get_charset("alpha")
    renderer = DeterministicCanvasRenderer()

    # Pre-render training catalog
    train_catalog, train_metas = pre_render_font_catalog(train_fonts, charset, renderer)
    train_font_ids = list(train_catalog.keys())
    if len(train_font_ids) < batch_size:
        raise RuntimeError("Not enough valid fonts for training batch size.")

    # 2. Model Initialization with Shared Backbone
    dual_encoder = DualEncoder(
        embed_dim=embed_dim,
        backbone_type="convnext",
        shared_backbone=True,    # Both branches share the ~20.5M parameter representation
        freeze_canonical=False,  # Unfrozen: canonical space actively organizes by font geometry
    ).to(device)

    degradation = TypographicDegradationPipeline().to(device)
    criterion = InfoNCELoss(init_temperature=0.05, learnable=True).to(device)

    optimizer = torch.optim.AdamW(
        list(dual_encoder.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    steps_per_epoch = 120
    print(f"\nBeginning training: {epochs} epochs, {steps_per_epoch} hard-negative steps/epoch, batch size {batch_size}...")

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        t_epoch = time.perf_counter()
        dual_encoder.train()

        running_loss = 0.0
        running_acc = 0.0
        running_pos_sim = 0.0
        running_neg_sim = 0.0

        for _ in range(steps_per_epoch):
            # HARD-NEGATIVE SAMPLING:
            # Pick ONE random character for this batch
            ch = random.choice(charset)

            # Sample B distinct fonts that have rendered this character
            eligible_fonts = [fid for fid in train_font_ids if ch in train_catalog[fid]]
            if len(eligible_fonts) < batch_size:
                continue
            sampled_fids = random.sample(eligible_fonts, batch_size)

            # Extract canonical renders [B, 1, 224, 224]
            canonical_batch = torch.stack(
                [train_catalog[fid][ch] for fid in sampled_fids], dim=0
            ).to(device, non_blocking=True).float() / 255.0

            # Online synthetic degradation
            with torch.no_grad():
                visual_batch = degradation(canonical_batch)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    z_v, z_c = dual_encoder(visual_batch, canonical_batch)
                    loss, metrics = criterion(z_v, z_c)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z_v, z_c = dual_encoder(visual_batch, canonical_batch)
                loss, metrics = criterion(z_v, z_c)
                loss.backward()
                optimizer.step()

            running_loss += metrics["loss"]
            running_acc += metrics["in_batch_top1_acc"]
            running_pos_sim += metrics["pos_sim"]
            running_neg_sim += metrics["neg_sim"]

        scheduler.step()
        epoch_time = time.perf_counter() - t_epoch

        epoch_loss = running_loss / steps_per_epoch
        epoch_acc = (running_acc / steps_per_epoch) * 100.0
        pos_sim = running_pos_sim / steps_per_epoch
        neg_sim = running_neg_sim / steps_per_epoch

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) - "
            f"Loss: {epoch_loss:.4f} | "
            f"In-Batch Top-1 Acc: {epoch_acc:.1f}% | "
            f"Pos Sim: {pos_sim:.3f} | "
            f"Neg Sim: {neg_sim:.3f} | "
            f"Tau: {criterion.temperature:.4f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(dual_encoder.state_dict(), output_dir / "glyphclip_scaled.pt")

    print(f"\nTraining complete! Best checkpoint saved to {output_dir / 'glyphclip_scaled.pt'}")

    # 3. Independent Test Evaluation & Benchmark
    print("\n-------------------------------------------------------")
    print("Evaluating on Held-Out Independent Test Set...")
    print("-------------------------------------------------------")
    test_harness = IndependentTestHarness(
        dual_encoder=dual_encoder,
        test_font_paths=test_fonts,
        device=device,
        embed_dim=embed_dim,
        charset_subset="alpha",
    )

    metrics = test_harness.run_benchmark(test_words=["Funnel", "Typography", "Sample", "Geometric"])
    print(f"Benchmark Results on {metrics.total_queries} Test Queries:")
    print(f"  • Top-1 Retrieval Accuracy: {metrics.top1_accuracy:.1f}%")
    print(f"  • Top-3 Retrieval Accuracy: {metrics.top3_accuracy:.1f}%")
    print(f"  • Top-5 Retrieval Accuracy: {metrics.top5_accuracy:.1f}%")
    print(f"  • Mean Reciprocal Rank (MRR): {metrics.mrr:.4f}")

    # Test funnel.png directly
    funnel_path = Path("funnel.png")
    if funnel_path.exists():
        print("\nTesting funnel.png on the updated model...")
        funnel_img = cv2.imread(str(funnel_path))
        rank, preds = test_harness.evaluate_word_query(funnel_img, "Funnel Display", top_k=5)
        print(f"funnel.png -> Funnel Display Rank: {rank if rank != -1 else 'Not in Top-5'}")
        for p in preds:
            print(f"   Rank #{p.rank}: {p.family} (confidence: {p.confidence:.4f})")

    # 4. Save Vector Store for Production Inference
    test_harness.vector_store.save(index_output_dir)
    print(f"\nPersisted updated {embed_dim}-d index to {index_output_dir} ({len(test_harness.vector_store)} vectors).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Scaled GlyphCLIP with hard-negative sampling.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    font_directories = [
        Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts"),
        Path(r"C:\Windows\Fonts"),
    ]
    train_scaled(
        font_dirs=font_directories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
