"""Verify production 2-ch Vector-SDF font_index across diverse typefaces and funnel.png."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from glyphclip.canvas.charset import get_charset
from glyphclip.index.consensus import BayesianConsensusEngine
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.pipeline.segmentation import VisualSegmentationPipeline


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vs = GlyphVectorStore.load("font_index")
    print(f"Loaded Vector-SDF store: {len(vs)} vectors")

    ckpt_state = torch.load("checkpoints/glyphclip_sdf.pt", map_location=device, weights_only=True)
    char_head_w = ckpt_state.get("visual_encoder.char_head.weight")
    num_chars = int(char_head_w.shape[0]) if (char_head_w is not None and char_head_w.ndim == 2) else 0
    print(f"Loaded checkpoint with auxiliary char_head num_chars={num_chars}")

    dual = DualEncoder(
        embed_dim=512,
        backbone_type="convnext",
        shared_backbone=True,
        in_channels=2,
        num_chars=num_chars,
    ).to(device)
    dual.load_state_dict(ckpt_state)
    dual.eval()

    charset = get_charset("alpha")
    seg = VisualSegmentationPipeline(include_sdf=True)
    eng = BayesianConsensusEngine(vs, top_k=50, missing_penalty_lambda=0.5)

    def run_query(gt_name: str, word: str, img_arr: np.ndarray, use_hint: bool = True):
        glyphs = seg.process(img_arr)
        crops = [g.canvas_tensor for g in glyphs]
        b = torch.stack(crops, dim=0).to(device)
        with torch.no_grad():
            embs = dual.encode_visual(b).cpu().numpy()
        q_chars = list(word[:len(crops)]) if use_hint else None
        q_cands = None if use_hint else dual.predict_char_candidates(b, charset, top_k=3, min_prob=0.08)
        preds = eng.aggregate(
            list(embs), query_chars=q_chars, max_results=5, query_char_candidates=q_cands
        )
        rank = -1
        for p in preds:
            if gt_name.lower() in p.family.lower() or p.family.lower() in gt_name.lower():
                rank = p.rank
                break
        mode = "Hint" if use_hint else "Blind"
        top_str = ", ".join(f"#{p.rank} {p.family} ({p.confidence*100:.1f}%)" for p in preds[:3])
        print(f"[{mode}] {gt_name:<18} '{word:<7}' -> GT Rank: {rank if rank > 0 else '>5':<2} | Top3: {top_str}")
        return rank == 1

    def render_sample(font_path: str, word: str, size: int = 64) -> np.ndarray:
        img = Image.new("RGB", (len(word) * size + 80, size + 60), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        f = ImageFont.truetype(font_path, size=size)
        draw.text((30, 20), word, fill=(0, 0, 0), font=f)
        return np.array(img)

    tests = [
        ("Times New Roman", r"C:\Windows\Fonts\times.ttf", "Typo"),
        ("Times New Roman", r"C:\Windows\Fonts\times.ttf", "Sample"),
        ("Georgia", r"C:\Windows\Fonts\georgia.ttf", "Sample"),
        ("Arial", r"C:\Windows\Fonts\arial.ttf", "Sample"),
        ("Calibri", r"C:\Windows\Fonts\calibri.ttf", "Sample"),
        ("Impact", r"C:\Windows\Fonts\impact.ttf", "Sample"),
        ("Comic Sans MS", r"C:\Windows\Fonts\comic.ttf", "Sample"),
        ("Consolas", r"C:\Windows\Fonts\consola.ttf", "Sample"),
        ("Montserrat", r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\Montserrat-Regular.ttf", "Design"),
        ("Poppins", r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\Poppins-Regular.ttf", "Sign"),
    ]

    hint_wins = 0
    blind_wins = 0
    total = 0

    for gt, fp, word in tests:
        if not Path(fp).exists():
            continue
        arr = render_sample(fp, word)
        if run_query(gt, word, arr, use_hint=True):
            hint_wins += 1
        if run_query(gt, word, arr, use_hint=False):
            blind_wins += 1
        total += 1

    # Also test funnel.png
    funnel_img = cv2.imread("funnel.png")
    if funnel_img is not None:
        if run_query("Funnel Display", "Funnel", funnel_img, use_hint=True):
            hint_wins += 1
        if run_query("Funnel Display", "Funnel", funnel_img, use_hint=False):
            blind_wins += 1
        total += 1

    print(f"\nSummary: Hint Top-1 = {hint_wins}/{total} ({hint_wins/total*100:.1f}%) | Blind Top-1 = {blind_wins}/{total} ({blind_wins/total*100:.1f}%)")


if __name__ == "__main__":
    main()
