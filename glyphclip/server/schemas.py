"""Pydantic schemas for GlyphCLIP REST microservice API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GlyphScore(BaseModel):
    """Similarity score detail for an individual detected glyph."""
    char: str
    similarity: float


class PredictionItem(BaseModel):
    """Ranked typeface prediction match."""
    rank: int
    family: str
    postscript_name: str
    weight: int
    style: str
    confidence: float
    per_glyph_scores: Dict[str, GlyphScore]
    license: Optional[str] = None
    download_url: Optional[str] = None


class RecognizeResponse(BaseModel):
    """Response payload for /v1/recognize endpoint."""
    status: str = "success"
    latency_ms: float
    characters_detected: int
    predictions: List[PredictionItem]


class IndexFontResponse(BaseModel):
    """Response payload for /v1/index/font endpoint."""
    status: str = "success"
    family: str
    postscript_name: str
    indexed_glyphs: int
    weights_indexed: List[int]
    total_vectors_in_store: int
    persisted: bool = False


class BatchIndexFontResponse(BaseModel):
    """Response payload for multi-file /v1/index/fonts endpoint."""
    status: str = "success"
    files_indexed: int
    results: List[IndexFontResponse]
    total_vectors_in_store: int
    persisted: bool = False


class HealthResponse(BaseModel):
    """Service health and operational status."""
    status: str = "healthy"
    version: str
    device: str
    total_vectors_indexed: int
