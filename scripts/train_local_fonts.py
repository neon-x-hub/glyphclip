"""High-throughput GPU training script for GlyphCLIP on local font catalogs.

Pre-renders canonical glyphs into fast RAM tensors, executes online synthetic
degradations, and trains MobileNetV4 via symmetric InfoNCE on CUDA.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Tuple

import numpy as np
import torch
from fontTools.ttLib import TTFont
from tqdm import tqdm

from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.renderer import DeterministicCanvasRenderer
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.models.loss import InfoNCELoss
from glyphclip.pipeline.degradation import TypographicDegradationPipeline


def discover_valid_fonts(font_dirs: List[Path], max_fonts: int = 500) -> List[Path]:
    """Scan directories and return fonts that have full coverage of canonical charset."""
    charset = set(get_charset())
    valid: List[Path] = []

    all_files = []
    for d in font_dirs:
        if d.exists():
            all_files.extend(list(d.glob("*.ttf")) + list(d.glob("*.otf")))

    print(f"Scanning {len(all_files)} total font files for glyph coverage...")
    for fpath in all_files:
        try:
            tt = TTFont(fpath)
            cmap = tt.getBestCmap()
            if cmap and all(ord(ch) in cmap for ch in charset):
                valid.append(fpath)
                if len(valid) >= max_fonts:
                    break
        except Exception:
            continue

    print(f"Selected {len(valid)} verified typefaces for training.")
    return valid


def pre_render_dataset(
    font_paths: List[Path],
    renderer: DeterministicCanvasRenderer,
) -> Tuple[torch.Tensor, List[str], List[dict]]:
    """Render canonical glyphs into a contiguous RAM tensor [N, 1, 224, 224]."""
    charset = get_charset()
    tensors = []
    keys = []
    metas = []

    print(f"Pre-rendering {len(font_paths) * len(charset)} canonical glyphs to memory...")
    t0 = time.perf_counter()

    for fpath in tqdm(font_paths, desc="Rendering glyphs"):
        try:
            meta = FontParser.parse(fpath)
            for ch in charset:
                arr = renderer.render_glyph_numpy(fpath, ch)
                # Ensure stroke pixels exist
                if arr.max() < 100:
                    continue
                # Store as uint8 tensor [1, 224, 224] to save RAM
                t = torch.from_numpy(arr).unsqueeze(0)
                tensors.append(t)
                keys.append(f"{meta.postscript_name}_{ch}")
                metas.append({
                    "family": meta.family,
                    "postscript_name": meta.postscript_name,
                    "weight": meta.weight,
                    "glyph_char": ch,
                    "source_path": str(fpath),
                })
        except Exception:
            continue

    stacked = torch.stack(tensors, dim=0)
    elapsed = time.perf_counter() - t0
    ram_mb = stacked.element_size() * stacked.nelement() / (1024 * 1024)
    print(f"Cached {len(stacked)} glyphs in {elapsed:.1f}s ({ram_mb:.1f} MB RAM).")
    return stacked, keys, metas


def run_training(
    font_dirs: List[Path],
    max_fonts: int = 500,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 2e-4,
    output_dir: Path = Path("./checkpoints"),
    index_output_dir: Path = Path("./font_index"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    index_output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    renderer = DeterministicCanvasRenderer()
    valid_fonts = discover_valid_fonts(font_dirs, max_fonts=max_fonts)
    if not valid_fonts:
        raise RuntimeError("No valid fonts found for training.")

    cached_tensors, record_keys, metadatas = pre_render_dataset(valid_fonts, renderer)
    num_samples = len(cached_tensors)

    # Initialize model components
    dual_encoder = DualEncoder(embed_dim=256, freeze_canonical=True).to(device)
    degradation = TypographicDegradationPipeline().to(device)
    criterion = InfoNCELoss(init_temperature=0.07, learnable=True).to(device)

    optimizer = torch.optim.AdamW(
        list(dual_encoder.visual_encoder.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    steps_per_epoch = max(10, num_samples // batch_size)
    print(f"\nStarting contrastive training: {epochs} epochs, {steps_per_epoch} steps/epoch, batch size {batch_size}...")

    for epoch in range(1, epochs + 1):
        t_epoch = time.perf_counter()
        dual_encoder.visual_encoder.train()
        running_loss = 0.0
        running_acc = 0.0
        running_pos_sim = 0.0
        running_neg_sim = 0.0

        for _ in range(steps_per_epoch):
            # Sample random batch indices
            indices = torch.randint(0, num_samples, (batch_size,))
            # Fetch uint8 slices and convert to normalized float32 [B, 1, 224, 224] on device
            canonical_batch = cached_tensors[indices].to(device, non_blocking=True).float() / 255.0

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
            f"Top-1 Acc: {epoch_acc:.1f}% | "
            f"Pos Sim: {pos_sim:.3f} | "
            f"Neg Sim: {neg_sim:.3f} | "
            f"Tau: {criterion.temperature:.4f}"
        )

    # Save trained checkpoint
    checkpoint_path = output_dir / "glyphclip_trained.pt"
    torch.save(dual_encoder.state_dict(), checkpoint_path)
    print(f"\nModel checkpoint saved to: {checkpoint_path}")

    # Build and save vector index
    print("\nBuilding canonical vector index for indexed fonts...")
    dual_encoder.eval()
    vector_store = GlyphVectorStore(embed_dim=256)

    # Encode in batches
    idx_batch_size = 128
    for i in range(0, num_samples, idx_batch_size):
        end = min(i + idx_batch_size, num_samples)
        batch = cached_tensors[i:end].to(device).float() / 255.0
        with torch.no_grad():
            embs = dual_encoder.encode_canonical(batch).cpu().numpy()
        vector_store.add_batch(record_keys[i:end], embs, metadatas[i:end])

    vector_store.save(index_output_dir)
    print(f"Indexed {len(vector_store)} canonical glyph vectors to {index_output_dir}")
    print("Training and catalog indexing completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GlyphCLIP on verified local fonts.")
    parser.add_argument("--max-fonts", type=int, default=500, help="Number of verified fonts to include")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    font_directories = [
        Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts"),
        Path(r"C:\Windows\Fonts"),
    ]
    run_training(
        font_dirs=font_directories,
        max_fonts=args.max_fonts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
