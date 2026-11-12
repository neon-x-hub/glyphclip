"""Run reproducible ablation experiments for the GlyphCLIP technical report.

Saves structured metrics to docs/ablation_results.json:
  1. Held-out 96-font zero-shot test split: 1-ch vs 2-ch v2.2 vs 2-ch v3.0, plus char_head ON vs OFF.
  2. Cross-weight jump sensitivity (delta_w in {0, 200, 300, 400}): v1.1 vs v2.2 vs v3.0.
  3. Edge-orthogonality trait scoring (ortho_penalty_weight=0.0 vs 1.4) on real pixel & smooth queries.
  4. IVF-Cosine ANN vs exact search: recall@k and latency across index sizes & nprobe ratios.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.sdf import VectorSDFRenderer, compute_edge_orthogonality
from glyphclip.index.consensus import BayesianConsensusEngine
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.pipeline.dataset import FontDatasetSplitter
from glyphclip.pipeline.evaluator import IndependentTestHarness
from glyphclip.pipeline.segmentation import VisualSegmentationPipeline


FONT_DIRS = [
    Path(r"C:\Users\ASUS\AppData\Local\Microsoft\Windows\Fonts"),
    Path(r"C:\Windows\Fonts"),
]


def load_encoder(ckpt_path: Path, in_channels: int, num_chars: int, device: torch.device) -> DualEncoder:
    model = DualEncoder(
        embed_dim=512,
        backbone_type="convnext",
        shared_backbone=True,
        in_channels=in_channels,
        num_chars=num_chars,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def eval_harness_mode(harness: IndependentTestHarness, use_text_hint: bool, disable_char_head: bool = False) -> Dict:
    orig_num_chars = harness.dual_encoder.num_chars
    if disable_char_head:
        harness.dual_encoder.num_chars = 0
    try:
        m = harness.run_benchmark(
            test_words=["Funnel", "Typography", "Sample", "Geometric"],
            use_text_hint=use_text_hint,
        )
    finally:
        harness.dual_encoder.num_chars = orig_num_chars
    return {
        "total_queries": m.total_queries,
        "top1": round(m.top1_accuracy, 2),
        "top3": round(m.top3_accuracy, 2),
        "top5": round(m.top5_accuracy, 2),
        "mrr": round(m.mrr, 4),
    }


def run_zero_shot_ablations(device: torch.device) -> Dict:
    print("\n=== Ablation 1: 96-Font Held-Out Zero-Shot & Auxiliary OCR Head ===")
    _, _, test_fonts = FontDatasetSplitter.get_font_splits(
        FONT_DIRS,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        max_total_fonts=None,
        max_per_family=4,
        seed=42,
    )
    print(f"Test split fonts: {len(test_fonts)}")
    results = {"num_test_fonts": len(test_fonts), "models": {}}

    configs = [
        ("v1_1_raster_1ch", Path("checkpoints/glyphclip_scaled.pt"), 1, 0),
        ("v2_2_vectorsdf_2ch", Path("checkpoints/glyphclip_v2_2.pt"), 2, 52),
        ("v3_0_vectorsdf_scratch", Path("checkpoints/glyphclip_v3.pt"), 2, 52),
    ]

    for name, ckpt, in_ch, n_chars in configs:
        if not ckpt.exists():
            print(f"Skipping {name}: {ckpt} not found")
            continue
        print(f"Evaluating {name} ({ckpt})...")
        enc = load_encoder(ckpt, in_ch, n_chars, device)
        harness = IndependentTestHarness(
            dual_encoder=enc,
            test_font_paths=test_fonts,
            device=device,
            embed_dim=512,
            charset_subset="alpha",
        )
        res = {
            "num_vectors": len(harness.vector_store),
            "text_aligned": eval_harness_mode(harness, use_text_hint=True),
            "blind_auto": eval_harness_mode(harness, use_text_hint=False),
        }
        if name == "v3_0_vectorsdf_scratch":
            res["blind_no_char_head"] = eval_harness_mode(
                harness, use_text_hint=False, disable_char_head=True
            )
        results["models"][name] = res
        print(f"  {name}: {json.dumps(res)}")

    return results


def _render_word_from_path(font_path: Path, word: str = "Sample", size: int = 72) -> np.ndarray:
    img = Image.new("RGB", (len(word) * size + 80, size + 80), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(font_path), size=size)
    draw.text((40, 40), word, fill=(20, 20, 20), font=font)
    arr = np.array(img)
    return cv2.GaussianBlur(arr, (3, 3), 0.5)


def run_cross_weight_ablations(device: torch.device) -> Dict:
    print("\n=== Ablation 2: Cross-Weight Invariance Across Delta-W Jumps ===")
    by_fam: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for d in FONT_DIRS:
        if not d.exists():
            continue
        for p in list(d.glob("*.ttf")) + list(d.glob("*.otf")):
            try:
                m = FontParser.parse(p)
                if not m.is_italic and not m.is_variable:
                    by_fam[FontDatasetSplitter.normalize_family_key(p, m.family)][m.weight] = p
            except Exception:
                continue

    # Define weight-jump buckets
    # delta 0: 400 -> 400 (30 families)
    # delta 200: 400 -> 600 or 500 -> 700 or 300 -> 500
    # delta 300: 400 -> 700 (30 families)
    # delta 400: 300 -> 700 or 400 -> 800
    buckets = {
        "delta_0": [(k, dw[400], dw[400]) for k, dw in by_fam.items() if 400 in dw and 700 in dw][:30],
        "delta_200": [],
        "delta_300": [(k, dw[400], dw[700]) for k, dw in by_fam.items() if 400 in dw and 700 in dw][:30],
        "delta_400": [],
    }
    for k, dw in by_fam.items():
        if 400 in dw and 600 in dw:
            buckets["delta_200"].append((k, dw[400], dw[600]))
        elif 500 in dw and 700 in dw:
            buckets["delta_200"].append((k, dw[500], dw[700]))
        elif 300 in dw and 500 in dw:
            buckets["delta_200"].append((k, dw[300], dw[500]))
    buckets["delta_200"] = buckets["delta_200"][:30]

    for k, dw in by_fam.items():
        if 300 in dw and 700 in dw:
            buckets["delta_400"].append((k, dw[300], dw[700]))
        elif 400 in dw and 800 in dw:
            buckets["delta_400"].append((k, dw[400], dw[800]))
    buckets["delta_400"] = buckets["delta_400"][:30]

    for bname, pairs in buckets.items():
        print(f"  Bucket {bname}: {len(pairs)} family pairs")

    charset = get_charset("alpha")
    sdf = VectorSDFRenderer()
    from glyphclip.canvas.renderer import DeterministicCanvasRenderer
    raster = DeterministicCanvasRenderer()

    results = {"buckets": {k: len(v) for k, v in buckets.items()}, "models": {}}
    configs = [
        ("v1_1_raster_1ch", Path("checkpoints/glyphclip_scaled.pt"), 1, 0),
        ("v2_2_vectorsdf_2ch", Path("checkpoints/glyphclip_v2_2.pt"), 2, 52),
        ("v3_0_vectorsdf_scratch", Path("checkpoints/glyphclip_v3.pt"), 2, 52),
    ]

    for mname, ckpt, in_ch, n_chars in configs:
        if not ckpt.exists():
            continue
        enc = load_encoder(ckpt, in_ch, n_chars, device)
        seg = VisualSegmentationPipeline(include_sdf=(in_ch == 2), min_aspect=0.06)
        m_res = {}

        for bname, pairs in buckets.items():
            if not pairs:
                continue
            store = GlyphVectorStore(embed_dim=512)
            ids, embs_list, metas = [], [], []
            with torch.no_grad():
                for fk, idx_p, _ in pairs:
                    meta = FontParser.parse(idx_p)
                    ts, vc = [], []
                    for ch in set("Sample"):
                        if in_ch == 2:
                            t = sdf.render_dual_tensor(idx_p, ch)
                            valid = t[0].max().item() > 0.2
                        else:
                            arr = raster.render_glyph_numpy(idx_p, ch)
                            t = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
                            valid = arr.max() >= 80
                        if valid:
                            ts.append(t)
                            vc.append(ch)
                    if ts:
                        b = torch.stack(ts, dim=0).to(device)
                        vecs = enc.encode_canonical(b).cpu().numpy()
                        for ch, v, tensor in zip(vc, vecs, ts):
                            ids.append(f"{fk}_{ch}")
                            embs_list.append(v)
                            metas.append({
                                "family": meta.family,
                                "postscript_name": meta.postscript_name,
                                "weight": meta.weight,
                                "glyph_char": ch,
                                "norm_key": fk,
                                "ortho_ratio": compute_edge_orthogonality(tensor[0]) if in_ch == 2 else 0.5,
                            })
            store.add_batch(ids, np.stack(embs_list, axis=0), metas)
            ce = BayesianConsensusEngine(store, top_k=50)

            hint_top1, hint_top3 = 0, 0
            blind_top1, blind_top3 = 0, 0
            hint_mrr, blind_mrr = 0.0, 0.0

            for fk, _, qry_p in pairs:
                gt_meta = FontParser.parse(qry_p)
                gt_norm = FontDatasetSplitter.normalize_family_key(qry_p, gt_meta.family)
                word_img = _render_word_from_path(qry_p, "Sample")
                crops = seg.process(word_img)
                if not crops:
                    continue
                qb = torch.stack([c.canvas_tensor for c in crops], dim=0).to(device)
                with torch.no_grad():
                    if n_chars > 0:
                        q_embs, logits = enc.encode_visual_with_chars(qb)
                        probs = torch.softmax(logits, dim=-1)
                        topv, topi = probs.topk(3, dim=-1)
                        cands = [
                            [charset[j] for j in topi[i].tolist() if probs[i, j] >= 0.08]
                            for i in range(len(crops))
                        ]
                    else:
                        q_embs = enc.encode_visual(qb)
                        cands = None
                qvecs = [e.cpu().numpy() for e in q_embs]
                qorthos = [compute_edge_orthogonality(c.canvas_tensor[0]) for c in crops]

                def get_rank(preds):
                    for pr in preds:
                        p_norm = FontDatasetSplitter.normalize_family_key(Path(pr.family), pr.family)
                        if p_norm == gt_norm or gt_meta.family.lower() in pr.family.lower() or pr.family.lower() in gt_meta.family.lower():
                            return pr.rank
                    return -1

                p_hint = ce.aggregate(
                    qvecs, query_chars=list("Sample")[:len(qvecs)], max_results=5, query_ortho_ratios=qorthos
                )
                r_h = get_rank(p_hint)
                if r_h == 1:
                    hint_top1 += 1
                if 1 <= r_h <= 3:
                    hint_top3 += 1
                if r_h > 0:
                    hint_mrr += 1.0 / r_h

                p_blind = ce.aggregate(
                    qvecs, query_chars=None, query_char_candidates=cands, max_results=5, query_ortho_ratios=qorthos
                )
                r_b = get_rank(p_blind)
                if r_b == 1:
                    blind_top1 += 1
                if 1 <= r_b <= 3:
                    blind_top3 += 1
                if r_b > 0:
                    blind_mrr += 1.0 / r_b

            n = len(pairs)
            m_res[bname] = {
                "n": n,
                "hint_top1": round(hint_top1 / n * 100.0, 1),
                "hint_top3": round(hint_top3 / n * 100.0, 1),
                "hint_mrr": round(hint_mrr / n, 4),
                "blind_top1": round(blind_top1 / n * 100.0, 1),
                "blind_top3": round(blind_top3 / n * 100.0, 1),
                "blind_mrr": round(blind_mrr / n, 4),
            }
            print(f"  [{mname}][{bname}] Hint Top1={m_res[bname]['hint_top1']}% | Blind Top1={m_res[bname]['blind_top1']}%")

        results["models"][mname] = m_res
    return results


def run_ortho_ablations(device: torch.device) -> Dict:
    print("\n=== Ablation 3: Edge-Orthogonality Trait (Pixel vs Smooth Queries) ===")
    store = GlyphVectorStore.load("font_index")
    enc = load_encoder(Path("checkpoints/glyphclip_v3.pt"), 2, 52, device)
    seg = VisualSegmentationPipeline(include_sdf=True, min_aspect=0.06)
    charset = get_charset("alpha")

    queries = [
        ("ENSIA@PixelOperator8.png", "ENSIA", "Silkscreen"),
        ("ENSIA@Minecraft.png", "ENSIA", "Silkscreen"),
        ("ENSIA@Slikscreen.png", "ENSIA", "Silkscreen"),
        ("Ram-Altman@Funnel.png", "RamAltman", "Funnel Display"),
        ("funnel.png", "Funnel", "Funnel Display"),
    ]
    out = {}
    for fname, hint, target in queries:
        fpath = Path(fname)
        if not fpath.exists():
            continue
        crops = seg.process(cv2.imread(str(fpath)))
        qb = torch.stack([c.canvas_tensor for c in crops], dim=0).to(device)
        with torch.no_grad():
            q_embs, logits = enc.encode_visual_with_chars(qb)
        probs = torch.softmax(logits, dim=-1)
        topv, topi = probs.topk(3, dim=-1)
        cands = [
            [charset[j] for j in topi[i].tolist() if probs[i, j] >= 0.08]
            for i in range(len(crops))
        ]
        qvecs = [e.cpu().numpy() for e in q_embs]
        orthos = [compute_edge_orthogonality(c.canvas_tensor[0]) for c in crops]

        q_entry = {"mean_ortho_ratio": round(float(np.mean(orthos)), 4), "num_glyphs": len(crops), "weights": {}}
        for w in [0.0, 1.4]:
            ce = BayesianConsensusEngine(store, ortho_penalty_weight=w, ortho_deadband=0.18)
            preds_h = ce.aggregate(qvecs, query_chars=list(hint)[:len(qvecs)], max_results=5, query_ortho_ratios=orthos)
            preds_b = ce.aggregate(qvecs, query_chars=None, query_char_candidates=cands, max_results=5, query_ortho_ratios=orthos)
            q_entry["weights"][f"w_{w}"] = {
                "hint_top3": [{"rank": p.rank, "family": p.family, "conf": round(p.confidence, 4)} for p in preds_h[:3]],
                "blind_top3": [{"rank": p.rank, "family": p.family, "conf": round(p.confidence, 4)} for p in preds_b[:3]],
            }
        out[fname] = q_entry
        print(f"  {fname} (mean_ortho={q_entry['mean_ortho_ratio']}):")
        print(f"    w=0.0 Blind #1: {q_entry['weights']['w_0.0']['blind_top3'][0]} | Hint #1: {q_entry['weights']['w_0.0']['hint_top3'][0]}")
        print(f"    w=1.4 Blind #1: {q_entry['weights']['w_1.4']['blind_top3'][0]} | Hint #1: {q_entry['weights']['w_1.4']['hint_top3'][0]}")
    return out


def run_ann_ablations() -> Dict:
    print("\n=== Ablation 4: IVF-Cosine ANN Recall & Latency Scaling ===")
    prod_store = GlyphVectorStore.load("font_index")
    rng = np.random.default_rng(42)

    sizes = [len(prod_store), 10000, 30000, 60000]
    nprobe_ratios = [0.10, 0.25, 0.50]
    charset = get_charset("alpha")
    out = {"sizes": sizes, "benchmarks": []}

    for n_vecs in sizes:
        if n_vecs == len(prod_store):
            store = prod_store
            label = f"real_prod_{n_vecs}"
        else:
            store = GlyphVectorStore(embed_dim=512, ann_threshold=512, nprobe_ratio=0.25)
            # Generate clustered unit embeddings mimicking real character & style structure
            n_styles = max(50, n_vecs // 52)
            base_styles = rng.normal(size=(n_styles, 512)).astype(np.float32)
            char_offsets = rng.normal(scale=0.6, size=(52, 512)).astype(np.float32)
            ids, vecs, metas = [], [], []
            for idx in range(n_vecs):
                s_id = idx % n_styles
                c_id = (idx // n_styles) % 52
                v = base_styles[s_id] + char_offsets[c_id] + rng.normal(scale=0.2, size=512).astype(np.float32)
                v = v / (np.linalg.norm(v) + 1e-8)
                ids.append(f"syn_{idx}")
                vecs.append(v)
                metas.append({"family": f"Fam_{s_id}", "glyph_char": charset[c_id], "weight": 400})
            store.add_batch(ids, np.stack(vecs, axis=0), metas)
            label = f"synthetic_{n_vecs}"

        # Ensure ANN is built cleanly
        store.build_ann_index()

        # Sample 150 noisy query vectors from existing vectors
        sample_indices = rng.choice(len(store), size=min(150, len(store)), replace=False)
        queries = []
        for idx in sample_indices:
            v = store.embeddings[idx] + rng.normal(scale=0.12, size=store.embed_dim).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            ch = store.metadata[idx].get("glyph_char", "a")
            queries.append((v, ch))

        # Exact linear unfiltered AND char-filtered timings
        exact_top5_unfiltered = []
        t0 = time.perf_counter()
        for qv, _ in queries:
            res = store.search(qv, top_k=10, filter_char=None, use_ann=False)
            exact_top5_unfiltered.append([r.id for r in res[:5]])
        exact_ms_unfiltered = (time.perf_counter() - t0) / len(queries) * 1000.0

        exact_top5_filtered = []
        t0 = time.perf_counter()
        for qv, ch in queries:
            res = store.search(qv, top_k=10, filter_char=ch, use_ann=False)
            exact_top5_filtered.append([r.id for r in res[:5]])
        exact_ms_filtered = (time.perf_counter() - t0) / len(queries) * 1000.0

        entry = {
            "label": label,
            "n_vecs": len(store),
            "n_cells": int(len(store._centroids)) if store._centroids is not None else 0,
            "exact_unfiltered_ms": round(exact_ms_unfiltered, 3),
            "exact_char_filtered_ms": round(exact_ms_filtered, 3),
            "ann_configs": {},
        }

        for ratio in nprobe_ratios:
            k_cells = len(store._centroids)
            nprobe = max(1, min(k_cells, int(np.ceil(k_cells * ratio))))

            # Unfiltered ANN recall & latency
            t0 = time.perf_counter()
            r1_un, r5_un = 0, 0
            for (qv, _), gt_ids in zip(queries, exact_top5_unfiltered):
                res = store.search(qv, top_k=10, filter_char=None, use_ann=True, nprobe=nprobe)
                pred_ids = [r.id for r in res]
                if pred_ids and pred_ids[0] == gt_ids[0]:
                    r1_un += 1
                if gt_ids[0] in pred_ids[:5]:
                    r5_un += 1
            ann_ms_unfiltered = (time.perf_counter() - t0) / len(queries) * 1000.0

            # Char-filtered ANN recall & latency
            t0 = time.perf_counter()
            r1_fi, r5_fi = 0, 0
            for (qv, ch), gt_ids in zip(queries, exact_top5_filtered):
                res = store.search(qv, top_k=10, filter_char=ch, use_ann=True, nprobe=nprobe)
                pred_ids = [r.id for r in res]
                if pred_ids and pred_ids[0] == gt_ids[0]:
                    r1_fi += 1
                if gt_ids[0] in pred_ids[:5]:
                    r5_fi += 1
            ann_ms_filtered = (time.perf_counter() - t0) / len(queries) * 1000.0

            nq = len(queries)
            entry["ann_configs"][f"nprobe_{ratio}"] = {
                "nprobe": nprobe,
                "unfiltered_ms": round(ann_ms_unfiltered, 3),
                "unfiltered_recall1": round(r1_un / nq * 100.0, 1),
                "unfiltered_recall5": round(r5_un / nq * 100.0, 1),
                "char_filtered_ms": round(ann_ms_filtered, 3),
                "char_filtered_recall1": round(r1_fi / nq * 100.0, 1),
                "char_filtered_recall5": round(r5_fi / nq * 100.0, 1),
            }

        out["benchmarks"].append(entry)
        print(f"  N={n_vecs} (cells={entry['n_cells']}): Exact={entry['exact_unfiltered_ms']:.3f}ms | "
              f"ANN(0.25)={entry['ann_configs']['nprobe_0.25']['unfiltered_ms']:.3f}ms "
              f"(R@1={entry['ann_configs']['nprobe_0.25']['unfiltered_recall1']}%, "
              f"CharFiltered={entry['ann_configs']['nprobe_0.25']['char_filtered_ms']:.3f}ms)")
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running GlyphCLIP Ablation Suite on {device}...")

    all_results = {
        "device": str(device),
        "ablation_zero_shot": run_zero_shot_ablations(device),
        "ablation_cross_weight": run_cross_weight_ablations(device),
        "ablation_edge_orthogonality": run_ortho_ablations(device),
        "ablation_ann_scaling": run_ann_ablations(),
    }

    out_path = Path("docs/ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved complete ablation measurements to {out_path}")


if __name__ == "__main__":
    main()
