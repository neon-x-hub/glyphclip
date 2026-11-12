"""Contrastive training harness for GlyphCLIP dual-encoder visual alignment."""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.renderer import DeterministicCanvasRenderer
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.models.loss import InfoNCELoss
from glyphclip.pipeline.degradation import TypographicDegradationPipeline


class FontSyntheticDataset(Dataset):
    """Generates synthetic (canonical, degraded) pair batches for contrastive training."""

    def __init__(self, font_paths, iterations_per_epoch: int = 1000):
        self.font_paths = [str(p) for p in font_paths]
        self.iterations = iterations_per_epoch
        self.renderer = DeterministicCanvasRenderer()
        self.charset = get_charset()

    def __len__(self) -> int:
        return self.iterations

    def __getitem__(self, idx: int):
        # Sample random font and random character
        font_idx = torch.randint(0, len(self.font_paths), (1,)).item()
        char_idx = torch.randint(0, len(self.charset), (1,)).item()

        font_path = self.font_paths[font_idx]
        char = self.charset[char_idx]

        canonical = self.renderer.render_glyph_tensor(font_path, char=char)
        return canonical, font_path, char


def train(
    fonts_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    output_dir: Path = Path("./checkpoints"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    font_files = list(Path(fonts_dir).glob("**/*.ttf")) + list(Path(fonts_dir).glob("**/*.otf"))
    if not font_files:
        raise ValueError(f"No fonts found in {fonts_dir}")

    dataset = FontSyntheticDataset(font_files, iterations_per_epoch=batch_size * 20)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    dual_encoder = DualEncoder(embed_dim=256, freeze_canonical=True).to(device)
    degradation = TypographicDegradationPipeline().to(device)
    criterion = InfoNCELoss(init_temperature=0.07, learnable=True).to(device)

    optimizer = torch.optim.AdamW(
        list(dual_encoder.visual_encoder.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    print(f"Starting contrastive alignment training across {len(font_files)} typefaces...")
    for epoch in range(1, epochs + 1):
        dual_encoder.visual_encoder.train()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        for canonical_batch, _, _ in dataloader:
            canonical_batch = canonical_batch.squeeze(1).unsqueeze(1).to(device)
            # Generate degraded visual observations
            with torch.no_grad():
                degraded_batch = degradation(canonical_batch)

            optimizer.zero_grad()
            z_v, z_c = dual_encoder(degraded_batch, canonical_batch)
            loss, metrics = criterion(z_v, z_c)

            loss.backward()
            optimizer.step()

            total_loss += metrics["loss"]
            total_acc += metrics["in_batch_top1_acc"]
            steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0.0
        avg_acc = (total_acc / steps * 100.0) if steps > 0 else 0.0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | In-Batch Top-1 Acc: {avg_acc:.1f}% | Tau: {criterion.temperature:.4f}")

        # Save checkpoint
        torch.save(dual_encoder.state_dict(), output_dir / f"glyphclip_epoch_{epoch:02d}.pt")

    print(f"Training complete. Weights saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GlyphCLIP visual encoder via InfoNCE.")
    parser.add_argument("--fonts-dir", type=str, required=True, help="Directory containing training fonts")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    train(Path(args.fonts_dir), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
