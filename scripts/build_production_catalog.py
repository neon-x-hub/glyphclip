"""Build production font catalog indexing Windows fonts and user library."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import torch
import numpy as np
from typing import List, Set
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.canvas.renderer import DeterministicCanvasRenderer
from glyphclip.canvas.sdf import VectorSDFRenderer, compute_edge_orthogonality
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.charset import get_charset
from glyphclip.index.vector_store import GlyphVectorStore

def main():
    parser = argparse.ArgumentParser(description="Build production font catalog.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    elif Path("checkpoints/glyphclip_sdf.pt").exists():
        ckpt_path = Path("checkpoints/glyphclip_sdf.pt")
    elif Path("checkpoints/glyphclip_scaled.pt").exists():
        ckpt_path = Path("checkpoints/glyphclip_scaled.pt")
    else:
        ckpt_path = Path("checkpoints/glyphclip_dev.pt")

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    stem_w = state.get("visual_encoder.downsample_layers.0.0.weight")
    in_channels = int(stem_w.shape[1]) if (stem_w is not None and stem_w.ndim == 4) else 1
    char_head_w = state.get("visual_encoder.char_head.weight")
    num_chars = int(char_head_w.shape[0]) if (char_head_w is not None and char_head_w.ndim == 2) else 0
    use_sdf = (in_channels == 2)
    print(f"Detected architecture: in_channels={in_channels} ({'Vector-SDF' if use_sdf else 'Raster'}), num_chars={num_chars}")

    dual = DualEncoder(
        embed_dim=512,
        backbone_type="convnext",
        shared_backbone=True,
        in_channels=in_channels,
        num_chars=num_chars,
    ).to(device)
    dual.load_state_dict(state)
    dual.eval()

    renderer = DeterministicCanvasRenderer()
    sdf_renderer = VectorSDFRenderer() if use_sdf else None
    charset = get_charset("alpha")  # 52 upper + lower Latin characters

    # 1. Windows core fonts
    win_fonts_dir = Path(r"C:\Windows\Fonts")
    core_win_files = [
        "arial.ttf", "arialbd.ttf", "ariblk.ttf",
        "calibri.ttf", "calibrib.ttf",
        "times.ttf", "timesbd.ttf",
        "segoeui.ttf", "segoeuib.ttf",
        "georgia.ttf", "georgiab.ttf",
        "verdana.ttf", "verdanab.ttf",
        "tahoma.ttf", "tahomabd.ttf",
        "trebuc.ttf", "trebucbd.ttf",
        "comic.ttf", "comicbd.ttf",
        "impact.ttf",
        "consola.ttf", "consolab.ttf",
        "cour.ttf", "courbd.ttf",
        "pala.ttf", "palab.ttf",
        "candara.ttf", "corbel.ttf", "constant.ttf",
    ]

    candidate_paths: List[Path] = []
    for fname in core_win_files:
        p = win_fonts_dir / fname
        if p.exists():
            candidate_paths.append(p)

    # 2. User fonts directory: curated distinct families
    user_fonts_dir = Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts")
    if user_fonts_dir.exists():
        user_patterns = [
            "FunnelDisplay-VariableFont_wght.ttf",
            "Montserrat-Regular.ttf", "Montserrat-Bold.ttf",
            "Poppins-Regular.ttf", "Poppins-Bold.ttf",
            "RobotoMono-Regular.ttf", "RobotoMono-Bold.ttf",
            "RobotoMono-VariableFont_wght.ttf",
            "BeVietnamPro-Regular.ttf", "BeVietnamPro-Bold.ttf",
            "BodoniModa-VariableFont_opsz,wght.ttf",
            "Balgin Regular.ttf", "Balgin Bold.ttf",
            "Neon-Regular.ttf", "Neon-Bold.ttf",
            "Archivo-Regular.ttf", "Archivo-Bold.ttf",
            "Arsenica-Arabic-Antiqua-Regular-TRIAL.ttf",
            "SignateGrotesk-Black.ttf",
            "Silkscreen-Regular.ttf",
        ]
        seen_user = set()
        for pat in user_patterns:
            for p in user_fonts_dir.glob(pat):
                if p.suffix.lower() in [".ttf", ".otf"] and p.name not in seen_user:
                    seen_user.add(p.name)
                    candidate_paths.append(p)

    print(f"Total curated font files: {len(candidate_paths)}")

    vector_store = GlyphVectorStore(embed_dim=512)
    indexed_families: Set[str] = set()

    batch_tensors = []
    batch_ids = []
    batch_metas = []

    def flush_batch():
        nonlocal batch_tensors, batch_ids, batch_metas
        if not batch_tensors:
            return
        # Ensure lengths are perfectly aligned
        n = min(len(batch_tensors), len(batch_ids), len(batch_metas))
        b = torch.stack(batch_tensors[:n], dim=0).to(device)
        with torch.no_grad():
            embs = dual.encode_canonical(b).cpu().numpy()
        vector_store.add_batch(batch_ids[:n], embs, batch_metas[:n])
        batch_tensors.clear()
        batch_ids.clear()
        batch_metas.clear()

    total_glyphs_indexed = 0
    for font_path in candidate_paths:
        try:
            meta = FontParser.parse(font_path)
            # Quick check if font renders Latin 'A'
            test_arr = renderer.render_glyph_numpy(font_path, "A", weight=meta.weight)
            if test_arr.max() < 50:
                continue

            weights = [300, 400, 600, 700] if meta.is_variable else [meta.weight]
            for w in weights:
                for ch in charset:
                    if use_sdf and sdf_renderer is not None:
                        t = sdf_renderer.render_dual_tensor(font_path, ch, weight=w)
                        if t[0].max().item() < 0.2:
                            continue
                    else:
                        arr = renderer.render_glyph_numpy(font_path, ch, weight=w)
                        if arr.max() < 50:
                            continue
                        t = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
                    m_dict = {
                        "family": meta.family,
                        "postscript_name": meta.postscript_name,
                        "glyph_char": ch,
                        "weight": w,
                        "italic": bool(getattr(meta, "is_italic", False)),
                        "is_variable": bool(getattr(meta, "is_variable", False)),
                        "ortho_ratio": round(compute_edge_orthogonality(t[0]), 4),
                    }
                    gid = f"{meta.postscript_name}_w{w}_{ch}"
                    batch_tensors.append(t)
                    batch_ids.append(gid)
                    batch_metas.append(m_dict)
                    total_glyphs_indexed += 1

                    if len(batch_tensors) >= 256:
                        flush_batch()

            indexed_families.add(meta.family)
            print(f"  + Indexed: {meta.family} ({len(weights)} weights)")
        except Exception as e:
            print(f"Error on {font_path.name}: {e}")
            # Sync lists if exception occurred
            n = min(len(batch_tensors), len(batch_ids), len(batch_metas))
            batch_tensors = batch_tensors[:n]
            batch_ids = batch_ids[:n]
            batch_metas = batch_metas[:n]
            continue

    flush_batch()
    print(f"\nIndexed {total_glyphs_indexed} glyph vectors across {len(indexed_families)} font families.")
    print(f"Families: {sorted(list(indexed_families))}")

    output_dir = Path("font_index")
    vector_store.save(output_dir)
    print(f"Vector store saved to {output_dir.resolve()}.")

if __name__ == "__main__":
    main()
