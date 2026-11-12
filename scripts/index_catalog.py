"""Batch index an entire directory of font files into GlyphVectorStore."""

import argparse
from pathlib import Path
from tqdm import tqdm
import torch

from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.renderer import DeterministicCanvasRenderer
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.models.dual_encoder import DualEncoder


def index_directory(
    fonts_dir: Path,
    output_index_dir: Path,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    fonts_dir = Path(fonts_dir)
    output_index_dir = Path(output_index_dir)
    output_index_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)
    dual_encoder = DualEncoder(embed_dim=256, freeze_canonical=True).to(device)
    dual_encoder.eval()

    renderer = DeterministicCanvasRenderer()
    vector_store = GlyphVectorStore(embed_dim=256)
    charset = get_charset()

    font_files = list(fonts_dir.glob("**/*.ttf")) + list(fonts_dir.glob("**/*.otf"))
    print(f"Discovered {len(font_files)} fonts in {fonts_dir}...")

    total_indexed = 0
    for font_path in tqdm(font_files, desc="Indexing fonts"):
        try:
            meta = FontParser.parse(font_path)
            weights = FontParser.get_supported_weights(meta)

            tensors = []
            record_ids = []
            metadatas = []

            for w in weights:
                for char in charset:
                    rec_id = f"{meta.postscript_name}_w{w}_{char}"
                    t = renderer.render_glyph_tensor(font_path, char=char, weight=w)
                    tensors.append(t)
                    record_ids.append(rec_id)
                    metadatas.append({
                        "family": meta.family,
                        "postscript_name": meta.postscript_name,
                        "weight": w,
                        "italic": meta.is_italic,
                        "glyph_char": char,
                        "license": meta.license,
                        "source_path": str(font_path),
                    })

            if tensors:
                batch = torch.stack(tensors, dim=0).to(device)
                with torch.no_grad():
                    embs = dual_encoder.encode_canonical(batch).cpu().numpy()
                vector_store.add_batch(record_ids, embs, metadatas)
                total_indexed += len(record_ids)

        except Exception as e:
            print(f"Skipping {font_path.name}: {e}")
            continue

    vector_store.save(output_index_dir)
    print(f"Successfully indexed {total_indexed} glyph vectors to {output_index_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch index font catalog into GlyphCLIP vector store.")
    parser.add_argument("--fonts-dir", type=str, required=True, help="Directory containing .ttf/.otf fonts")
    parser.add_argument("--output-dir", type=str, default="./font_index", help="Output directory for vector store")
    args = parser.parse_args()
    index_directory(Path(args.fonts_dir), Path(args.output_dir))
