"""Normalized output schema shared across all stack adapters.

Every adapter must return a `PageResult` so downstream metrics and the parquet
writer can treat every stack uniformly. Stacks that don't natively produce one
of the fields (e.g. Docling has no KV head) leave it as an empty list — the
metric layer treats absence as a zero score, not as a parse error.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class BBox:
    """Pixel-space bounding box on the rendered page image."""
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class TextBlock:
    text: str
    bbox: BBox | None = None
    confidence: float | None = None
    reading_order: int | None = None


@dataclass
class TableCell:
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""
    is_header: bool = False


@dataclass
class Table:
    cells: list[TableCell] = field(default_factory=list)
    bbox: BBox | None = None
    html: str | None = None  # if the stack emits HTML directly
    n_rows: int = 0
    n_cols: int = 0


@dataclass
class KVPair:
    key: str
    value: str
    key_bbox: BBox | None = None
    value_bbox: BBox | None = None
    confidence: float | None = None


@dataclass
class LayoutBlock:
    """Layout-detection output: regions of the page (text / table / figure / title / etc.)."""
    label: str
    bbox: BBox
    confidence: float | None = None


@dataclass
class PageResult:
    page_id: str
    stack_id: str
    model_revision: str  # HF revision SHA or vendor model ID + version
    raw_text: str = ""           # full reading-order text dump
    text_blocks: list[TextBlock] = field(default_factory=list)
    layout: list[LayoutBlock] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    kv_pairs: list[KVPair] = field(default_factory=list)
    latency_ms: float | None = None
    raw_response_path: str | None = None  # path to the on-disk dump of the stack's raw output
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
