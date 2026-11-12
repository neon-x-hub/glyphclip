"""Vector store for canonical glyph embeddings with character buckets and IVF-Cosine ANN.

Provides fast O(1) character-bucket lookup, Spherical K-Means Inverted File (IVF)
Approximate Nearest Neighbor search, and persistent storage for zero-shot font indexing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np
import torch


@dataclass
class SearchResult:
    """Individual vector search match result."""
    id: str
    similarity: float
    metadata: Dict[str, Any]


class GlyphVectorStore:
    """High-performance vector index for canonical glyph embeddings with IVF ANN backbone."""

    def __init__(
        self,
        embed_dim: int = 256,
        ann_threshold: int = 512,
        nprobe_ratio: float = 0.25,
    ):
        self.embed_dim = embed_dim
        self.ann_threshold = ann_threshold
        self.nprobe_ratio = nprobe_ratio
        self.ann_backend = "ivf-numpy"

        self.ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # [N, embed_dim] float32
        self.metadata: List[Dict[str, Any]] = []
        self.id_to_index: Dict[str, int] = {}

        # O(1) character-partitioned inverted buckets mapping glyph_char -> row indices
        self._char_buckets: Dict[str, List[int]] = {}

        # IVF-Cosine ANN structures
        self._centroids: Optional[np.ndarray] = None          # [K, embed_dim] unit float32
        self._assignments: Optional[np.ndarray] = None        # [N] int32 cluster IDs
        self._cell_buckets: Optional[List[List[int]]] = None  # K lists of global row indices
        self._built_n: int = 0

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def has_ann(self) -> bool:
        """Whether IVF-Cosine ANN centroids and inverted cell lists are built."""
        return (
            self._centroids is not None
            and self._assignments is not None
            and self._cell_buckets is not None
            and len(self._assignments) == len(self.ids)
        )

    def _register_char_bucket(self, idx: int, char: Optional[str]) -> None:
        if char:
            self._char_buckets.setdefault(char, []).append(idx)

    def _rebuild_char_buckets(self) -> None:
        buckets: Dict[str, List[int]] = {}
        for idx, meta in enumerate(self.metadata):
            ch = meta.get("glyph_char")
            if ch:
                buckets.setdefault(ch, []).append(idx)
        self._char_buckets = buckets

    def _rebuild_cell_buckets(self) -> None:
        if self._centroids is None or self._assignments is None:
            self._cell_buckets = None
            return
        k = len(self._centroids)
        buckets: List[List[int]] = [[] for _ in range(k)]
        for idx, cid in enumerate(self._assignments.tolist()):
            if 0 <= cid < k:
                buckets[cid].append(idx)
        self._cell_buckets = buckets

    @staticmethod
    def _normalize_rows(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        return (mat / norms).astype(np.float32, copy=False)

    @staticmethod
    def _top_k_order(sims: np.ndarray, k: int) -> np.ndarray:
        """Return indices of top-k descending similarities in O(M + k log k)."""
        if k <= 0 or len(sims) == 0:
            return np.empty(0, dtype=np.int64)
        k = min(k, len(sims))
        if k >= len(sims):
            return np.argsort(-sims)
        part = np.argpartition(-sims, k - 1)[:k]
        return part[np.argsort(-sims[part])]

    def build_ann_index(
        self,
        nlist: Optional[int] = None,
        max_iter: int = 12,
        seed: int = 42,
    ) -> bool:
        """Build Spherical K-Means Voronoi centroids and inverted lists over stored embeddings."""
        n = len(self.ids)
        if self.embeddings is None or n < 64:
            self._centroids = None
            self._assignments = None
            self._cell_buckets = None
            self._built_n = 0
            return False

        k = nlist if nlist is not None else max(16, min(256, int(round(np.sqrt(n)))))
        k = max(2, min(k, n))

        rng = np.random.default_rng(seed)
        if n > 24000:
            sample_idx = rng.choice(n, size=24000, replace=False)
            train_pts = self.embeddings[sample_idx]
        else:
            train_pts = self.embeddings

        init_idx = rng.choice(len(train_pts), size=k, replace=False)
        centroids = train_pts[init_idx].copy()
        centroids = self._normalize_rows(centroids)

        # Spherical Lloyd clustering on S^(d-1)
        for _ in range(max_iter):
            sims = np.dot(train_pts, centroids.T)
            labels = np.argmax(sims, axis=1)
            new_c = np.zeros_like(centroids)
            np.add.at(new_c, labels, train_pts)
            # Re-seed empty clusters cleanly if any occurred
            norms = np.linalg.norm(new_c, axis=1)
            empty = norms < 1e-6
            if np.any(empty):
                refill = rng.choice(len(train_pts), size=int(np.sum(empty)), replace=False)
                new_c[empty] = train_pts[refill]
            centroids = self._normalize_rows(new_c)

        # Assign all stored vectors in bounded chunks
        assignments = np.empty(n, dtype=np.int32)
        chunk = 16384
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            assignments[start:end] = np.argmax(np.dot(self.embeddings[start:end], centroids.T), axis=1)

        self._centroids = centroids
        self._assignments = assignments
        self._built_n = n
        self._rebuild_cell_buckets()
        return True

    def _assign_new_to_ann(self, new_vecs: np.ndarray, start_idx: int) -> None:
        """Incrementally assign newly appended vectors into existing IVF Voronoi cells."""
        if (
            self._centroids is None
            or self._assignments is None
            or self._cell_buckets is None
            or len(self._assignments) != start_idx
        ):
            return
        new_cids = np.argmax(np.dot(new_vecs, self._centroids.T), axis=1).astype(np.int32)
        self._assignments = np.concatenate([self._assignments, new_cids])
        for offset, cid in enumerate(new_cids.tolist()):
            self._cell_buckets[cid].append(start_idx + offset)

    def add(
        self,
        record_id: str,
        embedding: Union[np.ndarray, torch.Tensor, List[float]],
        metadata: Dict[str, Any],
    ) -> None:
        """Add or update a single canonical glyph vector in the store."""
        if isinstance(embedding, torch.Tensor):
            vec = embedding.detach().cpu().numpy().flatten().astype(np.float32)
        elif isinstance(embedding, list):
            vec = np.array(embedding, dtype=np.float32)
        else:
            vec = np.asarray(embedding, dtype=np.float32).flatten()

        if vec.shape[0] != self.embed_dim:
            raise ValueError(f"Expected embedding dimension {self.embed_dim}, got {vec.shape[0]}")

        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec = vec / norm

        if record_id in self.id_to_index:
            idx = self.id_to_index[record_id]
            old_char = self.metadata[idx].get("glyph_char")
            new_char = metadata.get("glyph_char")
            self.embeddings[idx] = vec
            self.metadata[idx] = metadata
            if old_char != new_char:
                if old_char and old_char in self._char_buckets:
                    self._char_buckets[old_char] = [i for i in self._char_buckets[old_char] if i != idx]
                self._register_char_bucket(idx, new_char)
            if self.has_ann:
                old_cid = int(self._assignments[idx])
                new_cid = int(np.argmax(np.dot(self._centroids, vec)))
                if old_cid != new_cid:
                    self._cell_buckets[old_cid] = [i for i in self._cell_buckets[old_cid] if i != idx]
                    self._cell_buckets[new_cid].append(idx)
                    self._assignments[idx] = new_cid
            return

        idx = len(self.ids)
        self.ids.append(record_id)
        self.metadata.append(metadata)
        self.id_to_index[record_id] = idx
        self._register_char_bucket(idx, metadata.get("glyph_char"))

        row = np.expand_dims(vec, axis=0)
        if self.embeddings is None:
            self.embeddings = row
        else:
            self.embeddings = np.vstack([self.embeddings, row])

        self._assign_new_to_ann(row, idx)

    def add_batch(
        self,
        ids: List[str],
        embeddings: Union[np.ndarray, torch.Tensor],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add or update a batch of canonical glyph vectors in O(B) vectorized chunks."""
        if len(ids) != len(metadatas):
            raise ValueError("ids and metadatas lists must have identical lengths.")
        if not ids:
            return

        if isinstance(embeddings, torch.Tensor):
            mat = embeddings.detach().cpu().numpy().astype(np.float32)
        else:
            mat = np.array(embeddings, dtype=np.float32)

        if mat.ndim == 1:
            mat = np.expand_dims(mat, axis=0)
        if mat.shape[1] != self.embed_dim:
            raise ValueError(f"Expected embedding dimension {self.embed_dim}, got {mat.shape[1]}")

        mat = self._normalize_rows(mat)

        pending_ids: List[str] = []
        pending_vecs: List[np.ndarray] = []
        pending_metas: List[Dict[str, Any]] = []
        pending_pos: Dict[str, int] = {}

        for rec_id, vec, meta in zip(ids, mat, metadatas):
            if rec_id in self.id_to_index:
                self.add(rec_id, vec, meta)
            elif rec_id in pending_pos:
                pos = pending_pos[rec_id]
                pending_vecs[pos] = vec
                pending_metas[pos] = meta
            else:
                pending_pos[rec_id] = len(pending_ids)
                pending_ids.append(rec_id)
                pending_vecs.append(vec)
                pending_metas.append(meta)

        if pending_ids:
            new_start_idx = len(self.ids)
            for offset, (rec_id, meta) in enumerate(zip(pending_ids, pending_metas)):
                idx = new_start_idx + offset
                self.ids.append(rec_id)
                self.metadata.append(meta)
                self.id_to_index[rec_id] = idx
                self._register_char_bucket(idx, meta.get("glyph_char"))

            new_mat = np.vstack(pending_vecs).astype(np.float32, copy=False)
            if self.embeddings is None:
                self.embeddings = new_mat
            else:
                self.embeddings = np.vstack([self.embeddings, new_mat])
            if self.has_ann:
                self._assign_new_to_ann(new_mat, new_start_idx)

        # Auto-build or refresh ANN when scaling past threshold / doubling catalog
        n = len(self.ids)
        if n >= self.ann_threshold and (not self.has_ann or n >= max(self._built_n * 2, self._built_n + 2048)):
            self.build_ann_index()

    def search(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[float]],
        top_k: int = 50,
        filter_char: Optional[Union[str, List[str], set[str]]] = None,
        use_ann: bool = True,
        nprobe: Optional[int] = None,
    ) -> List[SearchResult]:
        """Perform cosine k-NN search using O(1) char buckets and IVF-Cosine ANN when scaled."""
        if self.embeddings is None or len(self.ids) == 0:
            return []

        if isinstance(query_embedding, torch.Tensor):
            q = query_embedding.detach().cpu().numpy().flatten().astype(np.float32)
        elif isinstance(query_embedding, list):
            q = np.array(query_embedding, dtype=np.float32)
        else:
            q = np.asarray(query_embedding, dtype=np.float32).flatten()

        q_norm = float(np.linalg.norm(q))
        if q_norm > 1e-6:
            q = q / q_norm

        # 1. Character-filtered lookup via O(1) inverted char buckets
        has_char_filter = isinstance(filter_char, str) or (
            filter_char is not None and len(filter_char) > 0
        )
        if has_char_filter:
            if isinstance(filter_char, str):
                raw_indices = self._char_buckets.get(filter_char, [])
            else:
                raw_indices = []
                for ch in filter_char:
                    raw_indices.extend(self._char_buckets.get(ch, []))
            if not raw_indices:
                return []
            indices = np.asarray(raw_indices, dtype=np.int64)

            if use_ann and self.has_ann and len(indices) > self.ann_threshold:
                k_cells = len(self._centroids)
                probe_k = nprobe or max(8, min(k_cells, int(np.ceil(k_cells * self.nprobe_ratio))))
                c_sims = np.dot(self._centroids, q)
                probed_cells = self._top_k_order(c_sims, probe_k)
                cell_mask = np.zeros(k_cells, dtype=bool)
                cell_mask[probed_cells] = True
                ann_indices = indices[cell_mask[self._assignments[indices]]]
                if len(ann_indices) >= min(top_k, len(indices)):
                    indices = ann_indices

            sims = np.dot(self.embeddings[indices], q)
            sorted_local = self._top_k_order(sims, top_k)
            return [
                SearchResult(
                    id=self.ids[int(indices[loc])],
                    similarity=float(sims[loc]),
                    metadata=self.metadata[int(indices[loc])],
                )
                for loc in sorted_local
            ]

        # 2. Global unfiltered search (ANN IVF-Cosine when above threshold, exact otherwise)
        n = len(self.ids)
        if use_ann and self.has_ann and n > self.ann_threshold:
            k_cells = len(self._centroids)
            probe_k = nprobe or max(8, min(k_cells, int(np.ceil(k_cells * self.nprobe_ratio))))
            c_sims = np.dot(self._centroids, q)
            probed_cells = self._top_k_order(c_sims, probe_k)
            parts = [
                np.asarray(self._cell_buckets[int(c)], dtype=np.int64)
                for c in probed_cells
                if self._cell_buckets[int(c)]
            ]
            if parts:
                cand_indices = np.concatenate(parts)
                if len(cand_indices) >= min(top_k, n):
                    sims = np.dot(self.embeddings[cand_indices], q)
                    sorted_local = self._top_k_order(sims, top_k)
                    return [
                        SearchResult(
                            id=self.ids[int(cand_indices[loc])],
                            similarity=float(sims[loc]),
                            metadata=self.metadata[int(cand_indices[loc])],
                        )
                        for loc in sorted_local
                    ]

        sims = np.dot(self.embeddings, q)
        top_indices = self._top_k_order(sims, top_k)
        return [
            SearchResult(
                id=self.ids[int(idx)],
                similarity=float(sims[idx]),
                metadata=self.metadata[int(idx)],
            )
            for idx in top_indices
        ]

    def save(self, directory: Union[str, Path]) -> None:
        """Persist embeddings, metadata, and IVF ANN index to target directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.embeddings is not None:
            if len(self.ids) >= self.ann_threshold and not self.has_ann:
                self.build_ann_index()
            np.save(dir_path / "embeddings.npy", self.embeddings)

        if self.has_ann:
            np.savez(
                dir_path / "ann_index.npz",
                centroids=self._centroids,
                assignments=self._assignments,
                built_n=np.array([self._built_n], dtype=np.int64),
            )

        payload = {
            "embed_dim": self.embed_dim,
            "ann_threshold": self.ann_threshold,
            "nprobe_ratio": self.nprobe_ratio,
            "ids": self.ids,
            "metadata": self.metadata,
        }
        with open(dir_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "GlyphVectorStore":
        """Load vector index, metadata, character buckets, and ANN Voronoi cells."""
        dir_path = Path(directory)
        meta_file = dir_path / "metadata.json"
        emb_file = dir_path / "embeddings.npy"
        ann_file = dir_path / "ann_index.npz"

        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        store = cls(
            embed_dim=payload.get("embed_dim", 256),
            ann_threshold=payload.get("ann_threshold", 512),
            nprobe_ratio=payload.get("nprobe_ratio", 0.25),
        )
        store.ids = payload.get("ids", [])
        store.metadata = payload.get("metadata", [])
        store.id_to_index = {rec_id: idx for idx, rec_id in enumerate(store.ids)}
        store._rebuild_char_buckets()

        if emb_file.exists():
            store.embeddings = np.load(emb_file)

        if ann_file.exists() and store.embeddings is not None:
            ann_data = np.load(ann_file)
            centroids = ann_data["centroids"]
            assignments = ann_data["assignments"]
            if len(assignments) == len(store.ids):
                store._centroids = centroids.astype(np.float32, copy=False)
                store._assignments = assignments.astype(np.int32, copy=False)
                store._built_n = int(ann_data["built_n"][0]) if "built_n" in ann_data else len(store.ids)
                store._rebuild_cell_buckets()

        if not store.has_ann and store.embeddings is not None and len(store.ids) >= store.ann_threshold:
            store.build_ann_index()

        return store
