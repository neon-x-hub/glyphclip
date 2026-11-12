import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2, torch
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.pipeline.segmentation import VisualSegmentationPipeline
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.index.consensus import BayesianConsensusEngine

device = torch.device('cuda')
dual = DualEncoder(embed_dim=512, backbone_type='convnext', shared_backbone=True).to(device)
state = torch.load('checkpoints/glyphclip_scaled.pt', map_location=device, weights_only=True)
dual.load_state_dict(state)
dual.eval()

store = GlyphVectorStore.load('font_index')
consensus = BayesianConsensusEngine(store, top_k=len(store))

img = cv2.imread('funnel.png')
seg = VisualSegmentationPipeline()
glyphs = seg.process(img)
tensors = [g.canvas_tensor for g in glyphs]
batch = torch.stack(tensors, dim=0).to(device)

with torch.no_grad():
    embs = dual.encode_visual(batch).cpu().numpy()

q_vecs = [embs[i] for i in range(len(embs))]
preds = consensus.aggregate(q_vecs, max_results=len(store))

print("Top 10 Predictions:")
for p in preds[:10]:
    print(f"Rank {p.rank}: {p.family} (conf: {p.confidence:.4f})")

print("\nFunnel Display Search:")
found = False
for p in preds:
    if 'funnel' in p.family.lower():
        found = True
        print(f"Funnel Display Rank: {p.rank}/{len(preds)} | Confidence: {p.confidence:.4f}")
        for k, v in p.per_glyph_scores.items():
            print(f"   {k}: char {v['char']} sim {v['similarity']}")
        break
if not found:
    print("Funnel Display NOT FOUND in predictions list!")
