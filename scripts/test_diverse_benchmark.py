import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2, torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.canvas.renderer import DeterministicCanvasRenderer
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.charset import get_charset
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.pipeline.segmentation import VisualSegmentationPipeline

device = torch.device('cuda')
dual = DualEncoder(embed_dim=512, backbone_type='convnext', shared_backbone=True).to(device)
state = torch.load('checkpoints/glyphclip_scaled.pt', map_location=device, weights_only=True)
dual.load_state_dict(state)
dual.eval()

renderer = DeterministicCanvasRenderer()
charset = get_charset('alpha')

fonts = [
    r'C:\Windows\Fonts\arial.ttf',
    r'C:\Windows\Fonts\times.ttf',
    r'C:\Windows\Fonts\calibri.ttf',
    r'C:\Windows\Fonts\segoeui.ttf',
    r'C:\Windows\Fonts\georgia.ttf',
    r'C:\Windows\Fonts\comic.ttf',
    r'C:\Windows\Fonts\impact.ttf',
    r'C:\Windows\Fonts\consola.ttf',
    r'C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\FunnelDisplay-VariableFont_wght.ttf',
]

test_store = GlyphVectorStore(embed_dim=512)
tensors, ids, metas = [], [], []

for fpath in fonts:
    meta = FontParser.parse(fpath)
    weights = [300, 500, 600, 700] if meta.is_variable else [meta.weight]
    for w in weights:
        for ch in charset:
            arr = renderer.render_glyph_numpy(fpath, ch, weight=w)
            if arr.max() < 80: continue
            t = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
            tensors.append(t)
            ids.append(f'{meta.postscript_name}_w{w}_{ch}')
            metas.append({'family': meta.family, 'glyph_char': ch, 'weight': w})

batch = torch.stack(tensors, dim=0).to(device)
with torch.no_grad():
    embs = dual.encode_canonical(batch).cpu().numpy()
test_store.add_batch(ids, embs, metas)
print(f'Test store built with {len(test_store)} vectors across {len(fonts)} distinct families.')

def evaluate(font_path, word, query_img=None, ground_truth=None):
    if query_img is None:
        img = Image.new('RGB', (len(word)*60 + 80, 140), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        f = ImageFont.truetype(str(font_path), size=64)
        draw.text((30, 30), word, fill=(0, 0, 0), font=f)
        arr = np.array(img)
    else:
        arr = query_img

    seg = VisualSegmentationPipeline()
    glyphs = seg.process(arr)
    crops = [g.canvas_tensor for g in glyphs]
    b = torch.stack(crops, dim=0).to(device)
    with torch.no_grad():
        crop_embs = dual.encode_visual(b).cpu().numpy()

    # Character-constrained scoring
    fams = set(m['family'] for m in test_store.metadata)
    fam_scores = {}
    for fam in fams:
        glyph_scores = []
        for i, ch in enumerate(word[:len(crops)]):
            matching = [idx for idx, m in enumerate(test_store.metadata) if m['family'] == fam and m['glyph_char'] == ch]
            if not matching:
                glyph_scores.append(0.0)
            else:
                sim = max(float(test_store.embeddings[idx] @ crop_embs[i]) for idx in matching)
                glyph_scores.append(sim)
        fam_scores[fam] = np.mean(glyph_scores)

    ranked = sorted(fam_scores.items(), key=lambda x: x[1], reverse=True)
    gt = ground_truth or Path(font_path).name
    top_fam, top_score = ranked[0]
    is_correct = (top_fam.lower() in gt.lower() or gt.lower() in top_fam.lower())
    print(f'Query "{word}" in {gt}: Rank #1 -> {top_fam} ({top_score:.4f}) | Correct: {is_correct}')
    for r, (fam, sc) in enumerate(ranked[:3], 1):
        print(f'    #{r}: {fam} ({sc:.4f})')

evaluate(r'C:\Windows\Fonts\times.ttf', 'Sample', ground_truth='Times New Roman')
evaluate(r'C:\Windows\Fonts\comic.ttf', 'Sample', ground_truth='Comic Sans MS')
evaluate(r'C:\Windows\Fonts\impact.ttf', 'Sample', ground_truth='Impact')
evaluate(r'C:\Windows\Fonts\consola.ttf', 'Sample', ground_truth='Consolas')
funnel_img = cv2.imread('funnel.png')
evaluate(r'C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts\FunnelDisplay-VariableFont_wght.ttf', 'Funnel', query_img=funnel_img, ground_truth='Funnel Display')
