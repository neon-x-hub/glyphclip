"""Typographic character set specifications and entropy-based discriminative weights."""

import string
from typing import Dict, List

# Full standard alphanumeric character set (62 glyphs)
# Covers all lowercase, uppercase, and digits so words like "Funnel" are completely represented
ALPHANUMERIC_CHARSET: List[str] = list(string.ascii_lowercase + string.ascii_uppercase + string.digits)

# Core subset of topologically discriminative glyphs from DESIGN.md
CORE_DISCRIMINATIVE_SET: List[str] = [
    "a", "g", "e", "f", "Q", "R", "S", "O", "M", "t"
]

# Canonical character set defaults to full alphabet
CANONICAL_CHARSET: List[str] = ALPHANUMERIC_CHARSET

# Typographic discriminative weights reflecting geometric entropy
# (e.g. two-story 'g', calligraphic 'Q' tail vs simple geometric 'l')
DISCRIMINATIVE_WEIGHTS: Dict[str, float] = {
    # High entropy lowercase
    "g": 1.4,
    "a": 1.3,
    "e": 1.2,
    "f": 1.2,
    "t": 1.1,
    "s": 1.1,
    "r": 1.1,
    "k": 1.1,
    "x": 1.1,
    # High entropy uppercase
    "Q": 1.4,
    "R": 1.3,
    "S": 1.2,
    "G": 1.2,
    "B": 1.1,
    "M": 1.1,
    "W": 1.1,
    "A": 1.1,
    # Neutral weights
    "O": 0.9,
    "o": 0.9,
    "c": 0.9,
    "C": 0.9,
    # Low-entropy glyphs
    "l": 0.6,
    "i": 0.6,
    "I": 0.6,
    "1": 0.6,
}

DEFAULT_WEIGHT: float = 1.0


def get_charset(subset: str = "full") -> List[str]:
    """Return character set: 'full' (62 glyphs), 'alpha' (52 glyphs), or 'core' (10 glyphs)."""
    if subset == "core":
        return list(CORE_DISCRIMINATIVE_SET)
    elif subset == "alpha":
        return list(string.ascii_letters)
    return list(ALPHANUMERIC_CHARSET)


def get_entropy_weight(char: str) -> float:
    """Return the discriminative weight w_i for a given character symbol."""
    return DISCRIMINATIVE_WEIGHTS.get(char, DEFAULT_WEIGHT)
