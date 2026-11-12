"""Bayesian Multi-Glyph Consensus Scoring Engine.

Aggregates individual character-level retrieval scores into high-confidence
word-level font predictions using typographic entropy weighting and
missing-candidate penalization.
"""

import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from glyphclip.canvas.charset import get_entropy_weight
from glyphclip.index.vector_store import GlyphVectorStore, SearchResult


@dataclass
class FontPrediction:
    """Ranked font match prediction for a query image."""
    rank: int
    family: str
    postscript_name: str
    weight: int
    style: str
    confidence: float
    per_glyph_scores: Dict[str, Dict[str, Any]]
    license: Optional[str] = None
    download_url: Optional[str] = None


class BayesianConsensusEngine:
    """Aggregates multi-glyph k-NN candidate hits into unified typeface rankings."""

    def __init__(
        self,
        vector_store: GlyphVectorStore,
        top_k: int = 50,
        missing_penalty_lambda: float = 0.5,
        ortho_penalty_weight: float = 1.4,
        ortho_deadband: float = 0.18,
    ):
        """Initialize the consensus scoring engine.

        Args:
            vector_store: Indexed canonical glyph vector database.
            top_k: Candidate pool depth retrieved per glyph (default: 50).
            missing_penalty_lambda: Penalty weight lambda for missing glyph candidates in [0, 1].
            ortho_penalty_weight: Penalty scale for query vs. candidate edge-orthogonality mismatch.
            ortho_deadband: Tolerance margin on |query_ortho - cand_ortho| before penalizing.
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.missing_penalty_lambda = missing_penalty_lambda
        self.ortho_penalty_weight = ortho_penalty_weight
        self.ortho_deadband = ortho_deadband

    def aggregate(
        self,
        query_embeddings: List[np.ndarray],
        query_chars: Optional[List[Optional[str]]] = None,
        max_results: int = 5,
        query_char_candidates: Optional[List[Optional[List[str]]]] = None,
        query_ortho_ratios: Optional[List[float]] = None,
    ) -> List[FontPrediction]:
        """Execute multi-glyph consensus scoring across query embeddings.

        Args:
            query_embeddings: List of visual query vectors from segmented word glyphs.
            query_chars: Optional list of identified/guessed character symbols for each crop.
            max_results: Number of top ranked font predictions to return.
            query_char_candidates: Optional list of top-k predicted character candidates per glyph
                                   from the auxiliary character head (used when query_chars is None).

        Returns:
            List of FontPrediction objects sorted by final composite confidence.
        """
        num_glyphs = len(query_embeddings)
        if num_glyphs == 0:
            return []

        # Default query characters if unknown
        if query_chars is None or len(query_chars) != num_glyphs:
            query_chars = [None] * num_glyphs
        else:
            query_chars = list(query_chars)

        if query_char_candidates is not None and len(query_char_candidates) != num_glyphs:
            query_char_candidates = None

        # Resolve per-glyph filters and primary character labels before computing weights
        per_glyph_filters = []
        for g_idx, q_emb in enumerate(query_embeddings):
            char_filter = query_chars[g_idx]
            if char_filter is not None:
                per_glyph_filters.append(char_filter)
            elif (
                query_char_candidates is not None
                and query_char_candidates[g_idx]
                and len(query_char_candidates[g_idx]) > 0
            ):
                cands = list(query_char_candidates[g_idx])
                query_chars[g_idx] = cands[0]
                per_glyph_filters.append(cands)
            else:
                # Fallback: auto-detect via nearest canonical votes
                vote_results = self.vector_store.search(q_emb, top_k=20)
                char_votes: Dict[str, float] = collections.defaultdict(float)
                for res in vote_results:
                    c = res.metadata.get("glyph_char")
                    if c:
                        char_votes[c] += max(0.0, res.similarity)
                if char_votes:
                    best_c = max(char_votes.items(), key=lambda x: x[1])[0]
                    query_chars[g_idx] = best_c
                    per_glyph_filters.append(best_c)
                else:
                    per_glyph_filters.append(None)

        # Assign typographic entropy weights
        glyph_weights = [
            get_entropy_weight(char) if char is not None else 1.0
            for char in query_chars
        ]
        total_weight = sum(glyph_weights) if sum(glyph_weights) > 0 else float(num_glyphs)

        if query_ortho_ratios is not None and len(query_ortho_ratios) != num_glyphs:
            query_ortho_ratios = None

        # 1. Retrieve candidate hits for each glyph crop with character consistency
        # Store per-glyph candidate similarities: glyph_hits[glyph_idx][family] = (max_sim, best_meta)
        glyph_hits: List[Dict[str, Dict[str, Any]]] = []
        all_candidate_families = set()

        for g_idx, q_emb in enumerate(query_embeddings):
            char_filter = per_glyph_filters[g_idx]
            num_filter_chars = len(char_filter) if isinstance(char_filter, list) else 1
            q_ortho = query_ortho_ratios[g_idx] if query_ortho_ratios is not None else None

            # Search canonical representations constrained to character / candidate chars
            results: List[SearchResult] = self.vector_store.search(
                q_emb,
                top_k=max(self.top_k, 250 * max(1, num_filter_chars)),
                filter_char=char_filter,
            )

            hits_for_glyph: Dict[str, Dict[str, Any]] = {}
            for res in results:
                fam = res.metadata.get("family", "Unknown")
                all_candidate_families.add(fam)
                sim = max(0.0, res.similarity)
                cand_ortho = res.metadata.get("ortho_ratio")
                if q_ortho is not None and cand_ortho is not None:
                    diff = abs(float(q_ortho) - float(cand_ortho))
                    if diff > self.ortho_deadband:
                        sim = max(0.0, sim - self.ortho_penalty_weight * (diff - self.ortho_deadband))

                if fam not in hits_for_glyph or sim > hits_for_glyph[fam]["similarity"]:
                    hits_for_glyph[fam] = {
                        "similarity": sim,
                        "metadata": res.metadata,
                        "char": res.metadata.get("glyph_char", query_chars[g_idx] or "?"),
                    }
            glyph_hits.append(hits_for_glyph)

        # 2. Compute composite Bayesian score C(f) and penalize missing candidates
        ranked_candidates = []
        for fam in all_candidate_families:
            weighted_sim_sum = 0.0
            missing_count = 0
            per_glyph_detail: Dict[str, Dict[str, Any]] = {}
            best_metadata: Dict[str, Any] = {}

            for g_idx in range(num_glyphs):
                key = f"c_{g_idx}"
                w_i = glyph_weights[g_idx]

                if fam in glyph_hits[g_idx]:
                    match_info = glyph_hits[g_idx][fam]
                    sim = match_info["similarity"]
                    weighted_sim_sum += w_i * sim
                    per_glyph_detail[key] = {
                        "char": match_info["char"],
                        "similarity": round(float(sim), 4),
                    }
                    if not best_metadata:
                        best_metadata = match_info["metadata"]
                else:
                    missing_count += 1
                    per_glyph_detail[key] = {
                        "char": query_chars[g_idx] or "?",
                        "similarity": 0.0,
                    }

            # Entropy-weighted geometric consensus
            raw_confidence = weighted_sim_sum / total_weight

            # Penalization factor for missing candidate sets
            missing_ratio = float(missing_count) / float(num_glyphs)
            final_confidence = raw_confidence * (1.0 - (self.missing_penalty_lambda * missing_ratio))

            ranked_candidates.append({
                "family": fam,
                "confidence": max(0.0, min(1.0, final_confidence)),
                "metadata": best_metadata,
                "per_glyph_scores": per_glyph_detail,
            })

        # 3. Sort descending by confidence
        ranked_candidates.sort(key=lambda c: c["confidence"], reverse=True)

        # Format output predictions
        predictions: List[FontPrediction] = []
        for rank, cand in enumerate(ranked_candidates[:max_results], start=1):
            meta = cand["metadata"]
            predictions.append(
                FontPrediction(
                    rank=rank,
                    family=cand["family"],
                    postscript_name=meta.get("postscript_name", cand["family"]),
                    weight=meta.get("weight", 400),
                    style="italic" if meta.get("italic") else "normal",
                    confidence=round(float(cand["confidence"]), 4),
                    per_glyph_scores=cand["per_glyph_scores"],
                    license=meta.get("license"),
                    download_url=meta.get("download_url"),
                )
            )

        return predictions
