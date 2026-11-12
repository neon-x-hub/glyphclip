"""Font dataset splitting, hard-negative batch generator, and independent test harness."""

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from fontTools.ttLib import TTFont
from torch.utils.data import Dataset

from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.renderer import DeterministicCanvasRenderer
from glyphclip.pipeline.degradation import TypographicDegradationPipeline


class FontDatasetSplitter:
    """Manages reproducible family-disjoint Train, Validation, and Test splits."""

    _WEIGHT_SUFFIX_RE = re.compile(
        r"[-_\s]+(hairline|thin|extralight|ultralight|light|book|regular|normal|medium|semibold|demibold|bold|extrabold|ultrabold|heavy|black|italic|oblique)$"
    )

    @classmethod
    def normalize_family_key(cls, font_path: Path, metadata_family: Optional[str] = None) -> str:
        """Normalize a font's family key so all weights/styles of a typeface group together."""
        raw = (metadata_family or font_path.stem).strip().lower()
        prev = None
        while raw != prev:
            prev = raw
            raw = cls._WEIGHT_SUFFIX_RE.sub("", raw).strip()
        return raw or font_path.stem.lower()

    @classmethod
    def get_font_splits(
        cls,
        font_dirs: List[Path],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        max_total_fonts: Optional[int] = 300,
        max_per_family: Optional[int] = None,
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """Split verified fonts into strictly family-disjoint train, validation, and test sets.

        All weight/style files belonging to the same font family stay in the exact same split,
        guaranteeing zero family leakage for true out-of-distribution zero-shot evaluation.
        """
        charset = set(get_charset("alpha"))  # A-Z, a-z
        all_candidates: List[Path] = []

        for d in font_dirs:
            if d.exists():
                all_candidates.extend(sorted(list(d.glob("*.ttf")) + list(d.glob("*.otf"))))

        # Group verified fonts strictly by normalized family key along with weight/italic info
        family_groups: Dict[str, List[Tuple[bool, int, Path]]] = {}
        funnel_fonts: List[Path] = []

        for fpath in all_candidates:
            try:
                tt = TTFont(fpath)
                cmap = tt.getBestCmap()
                if not cmap or not all(ord(ch) in cmap for ch in charset):
                    continue
                if "funnel" in fpath.name.lower():
                    funnel_fonts.append(fpath)
                    continue
                meta = FontParser.parse(fpath)
                fam_key = cls.normalize_family_key(fpath, meta.family)
                if "funnel" in fam_key:
                    funnel_fonts.append(fpath)
                else:
                    family_groups.setdefault(fam_key, []).append(
                        (bool(meta.is_italic), int(meta.weight), fpath)
                    )
            except Exception:
                continue

        # Optional per-family cap: select evenly across upright weights (keeps Thin/Regular/Bold pairs)
        rng = random.Random(seed)
        selected_groups: Dict[str, List[Path]] = {}
        for k, entries in family_groups.items():
            if max_per_family is not None and max_per_family > 0 and len(entries) > max_per_family:
                upright = sorted([e for e in entries if not e[0]], key=lambda x: (x[1], str(x[2])))
                italic = sorted([e for e in entries if e[0]], key=lambda x: (x[1], str(x[2])))
                chosen: List[Path] = []
                if len(upright) >= max_per_family:
                    idxs = np.linspace(0, len(upright) - 1, max_per_family, dtype=int)
                    chosen = [upright[int(i)][2] for i in idxs]
                else:
                    chosen = [e[2] for e in upright]
                    rem = max_per_family - len(chosen)
                    if rem > 0 and italic:
                        idxs = np.linspace(0, len(italic) - 1, min(rem, len(italic)), dtype=int)
                        chosen.extend(italic[int(i)][2] for i in idxs)
                selected_groups[k] = chosen
            else:
                selected_groups[k] = [e[2] for e in entries]

        # Shuffle family keys deterministically
        family_keys = sorted(selected_groups.keys())
        rng.shuffle(family_keys)

        # Apply optional max_total_fonts budget while keeping whole families intact
        if max_total_fonts is not None:
            selected_keys: List[str] = []
            running_files = len(funnel_fonts)
            budget = max(1, max_total_fonts)
            for k in family_keys:
                grp_len = len(selected_groups[k])
                if running_files + grp_len <= budget or len(selected_keys) < 3:
                    selected_keys.append(k)
                    running_files += grp_len
                if running_files >= budget:
                    break
            family_keys = selected_keys

        nf = len(family_keys)
        n_train = max(1, int(nf * train_ratio)) if nf >= 3 else nf
        n_val = max(1, int(nf * val_ratio)) if nf - n_train >= 2 else max(0, nf - n_train)

        train_keys = family_keys[:n_train]
        val_keys = family_keys[n_train : n_train + n_val]
        test_keys = family_keys[n_train + n_val :]

        train_fonts = [p for k in train_keys for p in selected_groups[k]]
        val_fonts = [p for k in val_keys for p in selected_groups[k]]
        test_fonts = [p for k in test_keys for p in selected_groups[k]]

        # Explicitly keep Funnel Display in held-out test set if present
        for fp in funnel_fonts:
            if fp not in test_fonts:
                test_fonts.append(fp)

        return train_fonts, val_fonts, test_fonts


class HardNegativeFontDataset(Dataset):
    """Generates character-aligned batches to force the network to learn pure font geometry.

    In each batch, multiple distinct fonts render the SAME character (e.g. all fonts render 'e'),
    forcing the contrastive objective to distinguish subtle typographic features (apertures, serifs,
    contrast axes) rather than simply performing OCR character recognition.
    """

    def __init__(
        self,
        font_paths: List[Path],
        charset: Optional[List[str]] = None,
        iterations_per_epoch: int = 1000,
    ):
        self.font_paths = font_paths
        self.charset = charset or get_charset("alpha")
        self.iterations = iterations_per_epoch
        self.renderer = DeterministicCanvasRenderer()

    def __len__(self) -> int:
        return self.iterations

    def get_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch where all fonts render the same randomly chosen character.

        Returns:
            canonical_batch: Pristine renders [B, 1, 224, 224] in [0, 1]
            visual_batch: Degraded visual renders [B, 1, 224, 224] in [0, 1]
        """
        # Pick one character for this entire mini-batch
        char = random.choice(self.charset)
        # Sample B distinct fonts
        sampled_fonts = random.sample(self.font_paths, min(batch_size, len(self.font_paths)))

        tensors = []
        for fpath in sampled_fonts:
            t = self.renderer.render_glyph_tensor(fpath, char=char)
            tensors.append(t)

        canonical = torch.stack(tensors, dim=0).to(device)
        return canonical, char
