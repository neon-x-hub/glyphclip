"""Train 2-channel Vector-SDF ConvNeXt Dual-Encoder (~20.5M params) and compare against 1-ch raster."""

import argparse
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.sdf import VectorSDFRenderer, batch_occupancy_to_dual
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.models.loss import InfoNCELoss
from glyphclip.pipeline.dataset import FontDatasetSplitter
from glyphclip.pipeline.degradation import TypographicDegradationPipeline
from glyphclip.pipeline.evaluator import IndependentTestHarness


def pre_render_sdf_catalog(
    font_paths: List[Path],
    charset: List[str],
    renderer: VectorSDFRenderer,
    desc: str = "Rendering Vector-SDF catalog",
    max_weight_gap: int = 400,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, str], Dict[str, List[str]]]:
    """Pre-render canonical [Occupancy, Vector-TSDF] tensors plus family and adjacent-weight maps."""
    catalog: Dict[str, Dict[str, torch.Tensor]] = {}
    fid_to_family: Dict[str, str] = {}
    fid_meta: Dict[str, Tuple[str, bool, int]] = {}
    t0 = time.perf_counter()

    for fpath in tqdm(font_paths, desc=desc):
        try:
            meta = FontParser.parse(fpath)
            font_id = meta.postscript_name
            font_key = f"{font_id}_{fpath.stem}"
            fam_key = FontDatasetSplitter.normalize_family_key(fpath, meta.family)
            font_dict: Dict[str, torch.Tensor] = {}

            for ch in charset:
                arr_u8 = renderer.render_dual_uint8(fpath, ch)
                if arr_u8[0].max() < 60:
                    continue
                font_dict[ch] = torch.from_numpy(arr_u8)

            if len(font_dict) >= len(charset) * 0.8:
                catalog[font_key] = font_dict
                fid_to_family[font_key] = fam_key
                fid_meta[font_key] = (fam_key, bool(meta.is_italic), int(meta.weight))
        except Exception:
            continue

    # Build cross-weight peer lists within same (family, italic) and |w1 - w2| <= max_weight_gap
    cross_weight_peers: Dict[str, List[str]] = {}
    by_fam_style: Dict[Tuple[str, bool], List[Tuple[str, int]]] = {}
    for fid, (fam, italic, weight) in fid_meta.items():
        by_fam_style.setdefault((fam, italic), []).append((fid, weight))
    for (_, _), members in by_fam_style.items():
        if len(members) < 2:
            continue
        for fid, w in members:
            peers = [pfid for pfid, pw in members if pfid != fid and abs(pw - w) <= max_weight_gap]
            if peers:
                cross_weight_peers[fid] = peers

    elapsed = time.perf_counter() - t0
    total_glyphs = sum(len(d) for d in catalog.values())
    print(f"Rendered {total_glyphs} Vector-SDF glyphs across {len(catalog)} fonts in {elapsed:.1f}s.")
    return catalog, fid_to_family, cross_weight_peers


def _sample_distinct_families(
    eligible_fids: List[str],
    fid_to_family: Dict[str, str],
    k: int,
) -> List[str]:
    """Sample at most one font variant per family so InfoNCE batches have no same-family false negatives."""
    by_fam: Dict[str, List[str]] = {}
    for fid in eligible_fids:
        fam = fid_to_family.get(fid, fid)
        by_fam.setdefault(fam, []).append(fid)
    fams = list(by_fam.keys())
    if len(fams) < k:
        return [random.choice(by_fam[f]) for f in fams]
    chosen_fams = random.sample(fams, k)
    return [random.choice(by_fam[f]) for f in chosen_fams]


def sample_hard_negative_pair(
    catalog: Dict[str, Dict[str, torch.Tensor]],
    font_ids: List[str],
    charset: List[str],
    batch_size: int,
    degradation: TypographicDegradationPipeline,
    device: torch.device,
    num_chars: int = 1,
    fid_to_family: Dict[str, str] = None,
    cross_weight_peers: Dict[str, List[str]] = None,
    p_cross_weight: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample distinct-family (font, char) pairs with optional adjacent-weight positives."""
    char_to_idx = {ch: idx for idx, ch in enumerate(charset)}
    fmap = fid_to_family or {}
    peers_map = cross_weight_peers or {}

    def pick_pair(fid: str, ch: str) -> Tuple[torch.Tensor, torch.Tensor]:
        can_t = catalog[fid][ch]
        vis_src_t = can_t
        if peers_map and random.random() < p_cross_weight:
            peers = [p for p in peers_map.get(fid, ()) if ch in catalog.get(p, ())]
            if peers:
                vis_src_t = catalog[random.choice(peers)][ch]
        return vis_src_t, can_t

    vis_src_tensors: List[torch.Tensor] = []
    can_tensors: List[torch.Tensor] = []
    char_labels: List[int] = []

    if num_chars <= 1:
        ch = random.choice(charset)
        eligible = [fid for fid in font_ids if ch in catalog[fid]]
        sampled = _sample_distinct_families(eligible, fmap, batch_size)
        if len(sampled) < 4:
            return None, None, None
        for fid in sampled:
            v_t, c_t = pick_pair(fid, ch)
            vis_src_tensors.append(v_t)
            can_tensors.append(c_t)
            char_labels.append(char_to_idx[ch])
    else:
        chosen_chars = random.sample(charset, min(num_chars, len(charset)))
        per_char = max(2, batch_size // len(chosen_chars))
        for ch in chosen_chars:
            eligible = [fid for fid in font_ids if ch in catalog[fid]]
            sampled_fids = _sample_distinct_families(eligible, fmap, per_char)
            if len(sampled_fids) < 2:
                continue
            for fid in sampled_fids:
                v_t, c_t = pick_pair(fid, ch)
                vis_src_tensors.append(v_t)
                can_tensors.append(c_t)
                char_labels.append(char_to_idx[ch])
        if len(can_tensors) < 4:
            return None, None, None

    canonical = torch.stack(can_tensors, dim=0).to(device).float() / 255.0
    vis_base = torch.stack(vis_src_tensors, dim=0).to(device).float() / 255.0
    char_indices = torch.tensor(char_labels, dtype=torch.long, device=device)

    with torch.no_grad():
        degraded_occ = degradation(vis_base[:, 0:1])
        visual = batch_occupancy_to_dual(degraded_occ, sigma=8.0, thr_jitter=0.06, fast_mask=True)

    return visual, canonical, char_indices


def evaluate_val_loss(
    dual_encoder: DualEncoder,
    criterion: InfoNCELoss,
    val_catalog: Dict[str, Dict[str, torch.Tensor]],
    val_fids: List[str],
    charset: List[str],
    degradation: TypographicDegradationPipeline,
    device: torch.device,
    steps: int = 24,
    batch_size: int = 24,
    val_fid_to_family: Dict[str, str] = None,
) -> Tuple[float, float, float, float]:
    """Compute held-out validation joint loss, contrastive Top-1, and auxiliary OCR Top-1/Top-3."""
    dual_encoder.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_ocr1 = 0.0
    val_ocr3 = 0.0
    count = 0

    with torch.no_grad():
        for step_idx in range(steps):
            n_ch = 1 if (step_idx % 2 == 0) else 4
            visual, canonical, char_indices = sample_hard_negative_pair(
                val_catalog,
                val_fids,
                charset,
                batch_size,
                degradation,
                device,
                num_chars=n_ch,
                fid_to_family=val_fid_to_family,
                p_cross_weight=0.0,
            )
            if visual is None:
                continue
            z_v, logits_v = dual_encoder.encode_visual_with_chars(visual)
            z_c, logits_c = dual_encoder.encode_canonical_with_chars(canonical)
            loss_nce, metrics = criterion(z_v, z_c)
            joint_loss = float(metrics["loss"])
            if logits_v is not None and logits_c is not None:
                ce_v = F.cross_entropy(logits_v, char_indices)
                ce_c = F.cross_entropy(logits_c, char_indices)
                joint_loss += float(0.4 * ce_v + 0.15 * ce_c)
                ocr1 = (logits_v.argmax(dim=-1) == char_indices).float().mean().item()
                _, top3 = logits_v.topk(min(3, logits_v.shape[-1]), dim=-1)
                ocr3 = top3.eq(char_indices.unsqueeze(1)).any(dim=1).float().mean().item()
                val_ocr1 += ocr1
                val_ocr3 += ocr3
            val_loss += joint_loss
            val_acc += metrics["in_batch_top1_acc"]
            count += 1

    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        val_loss / count,
        (val_acc / count) * 100.0,
        (val_ocr1 / count) * 100.0,
        (val_ocr3 / count) * 100.0,
    )


def train_sdf(
    font_dirs: List[Path],
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 2e-4,
    embed_dim: int = 512,
    max_fonts: Optional[int] = 200,
    max_per_family: Optional[int] = None,
    steps_per_epoch: int = 0,
    max_weight_gap: int = 400,
    p_cross_weight: float = 0.45,
    output_dir: Path = Path("./checkpoints"),
    ckpt_name: str = "glyphclip_sdf.pt",
    resume: bool = False,
    scratch: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode_tag = "v3 Scratch" if scratch else "v2.2 Resume/Warmstart"
    print("\n=======================================================")
    print(f"GlyphCLIP 2-Channel Vector-SDF ConvNeXt Training ({mode_tag})")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Input Channels: [0] Ink Occupancy + [1] Continuous Vector TSDF | Weight Gap <= {max_weight_gap}")
    print("=======================================================\n")

    # 1. Family-disjoint split cleanly across train / val / test
    budget = max_fonts if (max_fonts is not None and max_fonts > 0) else None
    per_fam_cap = max_per_family if (max_per_family is not None and max_per_family > 0) else None
    train_fonts, val_fonts, test_fonts = FontDatasetSplitter.get_font_splits(
        font_dirs,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        max_total_fonts=budget,
        max_per_family=per_fam_cap,
    )
    print(f"Dataset Split (Family-Disjoint): {len(train_fonts)} Train | {len(val_fonts)} Val | {len(test_fonts)} Test")

    charset = get_charset("alpha")
    sdf_renderer = VectorSDFRenderer(supersample=3, tsdf_sigma=8.0)

    train_catalog, train_fid_to_fam, train_peers = pre_render_sdf_catalog(
        train_fonts, charset, sdf_renderer, desc="Train Vector-SDF", max_weight_gap=max_weight_gap
    )
    val_catalog, val_fid_to_fam, _ = pre_render_sdf_catalog(
        val_fonts, charset, sdf_renderer, desc="Val Vector-SDF", max_weight_gap=max_weight_gap
    )

    train_fids = list(train_catalog.keys())
    val_fids = list(val_catalog.keys())
    if len(train_fids) < 16:
        raise RuntimeError(f"Not enough training fonts: {len(train_fids)}")
    if steps_per_epoch <= 0:
        steps_per_epoch = max(140, min(360, len(train_fids) // 2))
    print(f"Unique Training Families: {len(set(train_fid_to_fam.values()))} | Cross-Weight Peers (gap<={max_weight_gap}): {len(train_peers)} fonts")

    # 2. Initialize 2-channel ConvNeXt Dual-Encoder (+52-class auxiliary OCR head)
    dual_encoder = DualEncoder(
        embed_dim=embed_dim,
        backbone_type="convnext",
        shared_backbone=True,
        freeze_canonical=False,
        in_channels=2,
        num_chars=len(charset),
    ).to(device)

    ckpt_out = output_dir / ckpt_name
    warmstart_ckpt = output_dir / "glyphclip_scaled.pt"
    if scratch:
        print("Training cleanly FROM SCRATCH (random initialization; zero warm-start/resume leakage).")
    elif resume and ckpt_out.exists():
        print(f"Resuming 2-channel Vector-SDF weights from {ckpt_out} (strict=False)...")
        missing, unexpected = dual_encoder.load_state_dict(
            torch.load(ckpt_out, map_location=device, weights_only=True), strict=False
        )
        if missing:
            print(f"  Initialized new parameters: {missing}")
    elif warmstart_ckpt.exists():
        print(f"Warm-starting 2-channel backbone from {warmstart_ckpt}...")
        state_1ch = torch.load(warmstart_ckpt, map_location=device, weights_only=True)
        dual_encoder.load_warmstart_from_1ch(state_1ch)

    degradation = TypographicDegradationPipeline().to(device)
    criterion = InfoNCELoss(init_temperature=0.05, learnable=True).to(device)

    optimizer = torch.optim.AdamW(
        list(dual_encoder.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    if scratch and epochs >= 4:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.25, total_iters=2)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 2), eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[2])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"\nTraining for {epochs} epochs ({steps_per_epoch} mixed hard-negative/cross-char steps/epoch, bs={batch_size})...")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        dual_encoder.train()
        running_loss = 0.0
        running_acc = 0.0
        running_ocr1 = 0.0
        running_pos = 0.0
        running_neg = 0.0
        valid_steps = 0

        for step_idx in range(steps_per_epoch):
            # Alternate between pure same-char hard negatives (50%) and 4-char mixed negatives (50%)
            n_ch = 1 if (step_idx % 2 == 0) else 4
            visual, canonical, char_indices = sample_hard_negative_pair(
                train_catalog,
                train_fids,
                charset,
                batch_size,
                degradation,
                device,
                num_chars=n_ch,
                fid_to_family=train_fid_to_fam,
                cross_weight_peers=train_peers,
                p_cross_weight=p_cross_weight,
            )
            if visual is None:
                continue

            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    z_v, logits_v = dual_encoder.encode_visual_with_chars(visual)
                    z_c, logits_c = dual_encoder.encode_canonical_with_chars(canonical)
                    loss_nce, metrics = criterion(z_v, z_c)
                    loss = loss_nce
                    ocr_acc1 = 0.0
                    if logits_v is not None and logits_c is not None:
                        ce_v = F.cross_entropy(logits_v, char_indices)
                        ce_c = F.cross_entropy(logits_c, char_indices)
                        loss = loss_nce + 0.4 * ce_v + 0.15 * ce_c
                        ocr_acc1 = (logits_v.argmax(dim=-1) == char_indices).float().mean().item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z_v, logits_v = dual_encoder.encode_visual_with_chars(visual)
                z_c, logits_c = dual_encoder.encode_canonical_with_chars(canonical)
                loss_nce, metrics = criterion(z_v, z_c)
                loss = loss_nce
                ocr_acc1 = 0.0
                if logits_v is not None and logits_c is not None:
                    ce_v = F.cross_entropy(logits_v, char_indices)
                    ce_c = F.cross_entropy(logits_c, char_indices)
                    loss = loss_nce + 0.4 * ce_v + 0.15 * ce_c
                    ocr_acc1 = (logits_v.argmax(dim=-1) == char_indices).float().mean().item()
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            running_acc += metrics["in_batch_top1_acc"]
            running_ocr1 += ocr_acc1
            running_pos += metrics["pos_sim"]
            running_neg += metrics["neg_sim"]
            valid_steps += 1

        scheduler.step()
        elapsed = time.perf_counter() - t0
        tr_loss = running_loss / max(1, valid_steps)
        tr_acc = (running_acc / max(1, valid_steps)) * 100.0
        tr_ocr1 = (running_ocr1 / max(1, valid_steps)) * 100.0
        pos_s = running_pos / max(1, valid_steps)
        neg_s = running_neg / max(1, valid_steps)

        val_loss, val_acc, val_ocr1, val_ocr3 = evaluate_val_loss(
            dual_encoder,
            criterion,
            val_catalog,
            val_fids,
            charset,
            degradation,
            device,
            val_fid_to_family=val_fid_to_fam,
        )

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {tr_loss:.4f} Top1: {tr_acc:.1f}% OCR1: {tr_ocr1:.1f}% (Pos:{pos_s:.3f} Neg:{neg_s:.3f}) | "
            f"Val Loss: {val_loss:.4f} Top1: {val_acc:.1f}% OCR1/3: {val_ocr1:.1f}%/{val_ocr3:.1f}% | "
            f"Tau: {criterion.temperature:.4f}"
        )

        if val_loss > 0 and val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(dual_encoder.state_dict(), ckpt_out)
        elif val_loss == 0 and tr_loss <= best_val_loss:
            best_val_loss = tr_loss
            torch.save(dual_encoder.state_dict(), ckpt_out)

    print(f"\nBest Vector-SDF model saved to {ckpt_out}")

    # Load best checkpoint for held-out independent test comparison
    dual_encoder.load_state_dict(torch.load(ckpt_out, map_location=device, weights_only=True))
    dual_encoder.eval()

    print("\n-------------------------------------------------------")
    print("Zero-Shot Independent Test Evaluation (Vector-SDF vs 1-ch Raster)")
    print("-------------------------------------------------------")
    sdf_harness = IndependentTestHarness(
        dual_encoder=dual_encoder,
        test_font_paths=test_fonts,
        device=device,
        embed_dim=embed_dim,
        charset_subset="alpha",
    )
    m_aligned = sdf_harness.run_benchmark(
        test_words=["Funnel", "Typography", "Sample", "Geometric"], use_text_hint=True
    )
    m_blind = sdf_harness.run_benchmark(
        test_words=["Funnel", "Typography", "Sample", "Geometric"], use_text_hint=False
    )
    print(f"[2-ch Vector-SDF] Text-Aligned ({m_aligned.total_queries} queries): "
          f"Top-1={m_aligned.top1_accuracy:.1f}% | Top-3={m_aligned.top3_accuracy:.1f}% | MRR={m_aligned.mrr:.4f}")
    print(f"[2-ch Vector-SDF] Blind Auto   ({m_blind.total_queries} queries): "
          f"Top-1={m_blind.top1_accuracy:.1f}% | Top-3={m_blind.top3_accuracy:.1f}% | MRR={m_blind.mrr:.4f}")

    # Baseline 1-ch comparison on exact same test split
    if warmstart_ckpt.exists():
        base_encoder = DualEncoder(
            embed_dim=embed_dim, backbone_type="convnext", shared_backbone=True, in_channels=1
        ).to(device)
        base_encoder.load_state_dict(torch.load(warmstart_ckpt, map_location=device, weights_only=True))
        base_encoder.eval()
        base_harness = IndependentTestHarness(
            dual_encoder=base_encoder,
            test_font_paths=test_fonts,
            device=device,
            embed_dim=embed_dim,
            charset_subset="alpha",
        )
        b_aligned = base_harness.run_benchmark(
            test_words=["Funnel", "Typography", "Sample", "Geometric"], use_text_hint=True
        )
        b_blind = base_harness.run_benchmark(
            test_words=["Funnel", "Typography", "Sample", "Geometric"], use_text_hint=False
        )
        print(f"[1-ch Baseline  ] Text-Aligned ({b_aligned.total_queries} queries): "
              f"Top-1={b_aligned.top1_accuracy:.1f}% | Top-3={b_aligned.top3_accuracy:.1f}% | MRR={b_aligned.mrr:.4f}")
        print(f"[1-ch Baseline  ] Blind Auto   ({b_blind.total_queries} queries): "
              f"Top-1={b_blind.top1_accuracy:.1f}% | Top-3={b_blind.top3_accuracy:.1f}% | MRR={b_blind.mrr:.4f}")

    # Test funnel.png on 2-ch Vector-SDF
    funnel_path = Path("funnel.png")
    if funnel_path.exists():
        funnel_img = cv2.imread(str(funnel_path))
        rank_h, preds_h = sdf_harness.evaluate_word_query(
            funnel_img, "Funnel Display", top_k=5, query_text="Funnel"
        )
        rank_b, preds_b = sdf_harness.evaluate_word_query(
            funnel_img, "Funnel Display", top_k=5, query_text=None
        )
        print(f"\nfunnel.png (Hint='Funnel') -> Funnel Display Rank: #{rank_h}")
        for p in preds_h[:3]:
            print(f"   #{p.rank}: {p.family} ({p.confidence*100:.1f}%)")
        print(f"funnel.png (Blind Auto)    -> Funnel Display Rank: #{rank_b}")
        for p in preds_b[:3]:
            print(f"   #{p.rank}: {p.family} ({p.confidence*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2-Channel Vector-SDF GlyphCLIP.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-fonts", type=int, default=200, help="<=0 uses all available font families")
    parser.add_argument("--max-per-family", type=int, default=0, help="Cap files per family (0 = uncapped)")
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 auto-scales with catalog size")
    parser.add_argument("--max-weight-gap", type=int, default=400, help="Max OS/2 weight distance for cross-weight positives")
    parser.add_argument("--p-cross-weight", type=float, default=0.45, help="Probability of pairing distinct weights of same family")
    parser.add_argument("--ckpt-name", type=str, default="glyphclip_sdf.pt", help="Output checkpoint filename")
    parser.add_argument("--resume", action="store_true", help="Resume from output checkpoint")
    parser.add_argument("--scratch", action="store_true", help="Train cleanly from random initialization (v3)")
    args = parser.parse_args()

    font_directories = [
        Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts"),
        Path(r"C:\Windows\Fonts"),
    ]
    train_sdf(
        font_dirs=font_directories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_fonts=args.max_fonts,
        max_per_family=args.max_per_family,
        steps_per_epoch=args.steps_per_epoch,
        max_weight_gap=args.max_weight_gap,
        p_cross_weight=args.p_cross_weight,
        ckpt_name=args.ckpt_name,
        resume=args.resume,
        scratch=args.scratch,
    )
