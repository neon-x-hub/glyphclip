import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import torch
import numpy as np
from glyphclip.server.app import create_app

app = create_app(device='cuda')
img = cv2.imread('funnel.png')
segmented = app.state.segmentation_pipeline.process(img)
print('Number of segmented crops:', len(segmented))

tensors = [g.canvas_tensor for g in segmented]
batch = torch.stack(tensors, dim=0).to('cuda')
with torch.no_grad():
    embs = app.state.dual_encoder.encode_visual(batch).cpu().numpy()

query_vectors = [embs[i] for i in range(len(embs))]

for i, q in enumerate(query_vectors):
    hits = app.state.vector_store.search(q, top_k=3)
    fam = hits[0].metadata["family"]
    ch = hits[0].metadata["glyph_char"]
    sim = hits[0].similarity
    print(f"Crop {i} top hit: {fam} (char: {ch}) sim: {sim:.4f}")

# Check where Funnel Display ranks for each crop
for i, q in enumerate(query_vectors):
    all_hits = app.state.vector_store.search(q, top_k=len(app.state.vector_store))
    funnel_hits = [(idx, h) for idx, h in enumerate(all_hits) if "funnel" in h.metadata.get("family", "").lower()]
    if funnel_hits:
        best_rank, best_hit = funnel_hits[0]
        print(f"Crop {i}: Funnel Display best rank={best_rank+1}/{len(all_hits)}, char={best_hit.metadata['glyph_char']}, sim={best_hit.similarity:.4f}")

preds = app.state.consensus_engine.aggregate(query_vectors, max_results=5)
print('\nTop consensus predictions:')
for p in preds:
    print(f"Rank {p.rank}: {p.family} (conf: {p.confidence:.4f})")
    for k, v in p.per_glyph_scores.items():
        print(f"   {k}: char {v['char']} sim {v['similarity']}")
