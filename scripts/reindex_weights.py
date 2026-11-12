"""Re-index font catalog with discrete weights and full alphabet coverage."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm
from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.renderer import DeterministicCanvasRenderer
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.models.dual_encoder import DualEncoder


def reindex(
    output_index_dir: Path = Path("font_index"),
    weights_path: Path = Path("checkpoints/glyphclip_scaled.pt"),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading scaled ConvNeXt model on {device}...")
    dual = DualEncoder(embed_dim=512, backbone_type="convnext", shared_backbone=True).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    dual.load_state_dict(state)
    dual.eval()

    font_dirs = [
        Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts"),
        Path(r"C:\Windows\Fonts"),
    ]
    charset = get_charset("alpha")  # A-Z, a-z (52 letters)
    renderer = DeterministicCanvasRenderer()

    # Discover candidate fonts
    all_fonts = []
    for d in font_dirs:
        if d.exists():
            all_fonts.extend(list(d.glob("*.ttf")) + list(d.glob("*.otf")))

    # Select verified fonts (ensure Funnel Display is included)
    from fontTools.ttLib import TTFont
    check_set = set(charset)
    verified = []
    funnel_path = None
    for f in all_fonts:
        try:
            tt = TTFont(f)
            cmap = tt.getBestCmap()
            if cmap and all(ord(c) in cmap for c in check_set):
                if "funnel" in f.name.lower():
                    funnel_path = f
                else:
                    verified.append(f)
        except Exception:
            continue

    # Take 80 representative fonts plus Funnel Display
    selected_fonts = verified[:80]
    if funnel_path and funnel_path not in selected_fonts:
        selected_fonts.append(funnel_path)

    print(f"Indexing {len(selected_fonts)} typefaces across standard weights and {len(charset)} glyphs...")

    tensors = []
    ids = []
    metadatas = []

    for fpath in tqdm(selected_fonts, desc="Generating canonical embeddings"):
        try:
            meta = FontParser.parse(fpath)
            # Sample discrete weights for variable fonts: e.g. [300, 400, 500, 600, 700]
            if meta.is_variable:
                weights = [w for w in [300, 400, 500, 600, 700, 800] if "wght" in meta.variable_axes and meta.variable_axes["wght"].min_value <= w <= meta.variable_axes["wght"].max_value]
                if not weights:
                    weights = [meta.weight]
            else:
                weights = [meta.weight]

            for w in weights:
                for ch in charset:
                    arr = renderer.render_glyph_numpy(fpath, ch, weight=w)
                    if arr.max() < 80:
                        continue
                    t = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
                    tensors.append(t)
                    ids.append(f"{meta.postscript_name}_w{w}_{ch}")
                    metadatas.append({
                        "family": meta.family,
                        "postscript_name": meta.postscript_name,
                        "weight": w,
                        "italic": meta.is_italic,
                        "glyph_char": ch,
                        "source_path": str(fpath),
                    })
        except Exception:
            continue

    print(f"Encoding {len(tensors)} canonical vectors on GPU...")
    vector_store = GlyphVectorStore(embed_dim=512)

    batch_size = 128
    for i in range(0, len(tensors), batch_size):
        end = min(i + batch_size, len(tensors))
        batch = torch.stack(tensors[i:end], dim=0).to(device)
        with torch.no_grad():
            embs = dual.encode_canonical(batch).cpu().numpy()
        vector_store.add_batch(ids[i:end], embs, metadatas[i:end])

    output_index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save(output_index_dir)
    print(f"Successfully saved {len(vector_store)} vectors to {output_index_dir}!")


if __name__ == "__main__":
    reindex()
