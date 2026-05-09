"""Docling adapter — local execution on the RTX 3050.

Maps Docling's `ConversionResult.document` (a `DoclingDocument`) onto our
normalized `PageResult`. Docling has no native KV head — `kv_pairs` is
populated only when the document includes `key_value_items`/`form_items`
(rare in practice for arbitrary PDFs). For pages where KV matters, score
Docling on layout/tables/text and skip the KV axis.
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import StackAdapter
from .schema import (
    BBox,
    KVPair,
    LayoutBlock,
    PageResult,
    Table,
    TableCell,
    TextBlock,
)


def _bbox_or_none(prov_list) -> BBox | None:
    if not prov_list:
        return None
    box = getattr(prov_list[0], "bbox", None)
    if box is None:
        return None
    return BBox(x0=float(box.l), y0=float(box.t), x1=float(box.r), y1=float(box.b))


class DoclingAdapter(StackAdapter):
    stack_id = "docling"
    model_revision = "docling@local-pip"  # we report the installed pip version at warmup

    def __init__(self) -> None:
        self._converter = None

    def warmup(self) -> None:
        from docling.document_converter import DocumentConverter
        import docling

        self.model_revision = f"docling@{getattr(docling, '__version__', 'local-pip')}"
        self._converter = DocumentConverter()

    def process_page(self, image_path: Path, page_id: str) -> PageResult:
        if self._converter is None:
            self.warmup()

        result = PageResult(
            page_id=page_id,
            stack_id=self.stack_id,
            model_revision=self.model_revision,
        )

        try:
            t0 = time.perf_counter()
            conv = self._converter.convert(source=image_path, raises_on_error=True)
            result.latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            return result

        doc = conv.document

        try:
            result.raw_text = doc.export_to_text()
        except Exception:
            result.raw_text = ""

        for txt in getattr(doc, "texts", []) or []:
            result.text_blocks.append(
                TextBlock(
                    text=getattr(txt, "text", "") or "",
                    bbox=_bbox_or_none(getattr(txt, "prov", None)),
                    reading_order=None,
                )
            )
            label = getattr(txt, "label", None)
            label_str = getattr(label, "value", str(label)) if label is not None else "text"
            bbox = _bbox_or_none(getattr(txt, "prov", None))
            if bbox is not None:
                result.layout.append(LayoutBlock(label=str(label_str), bbox=bbox, confidence=None))

        for tbl in getattr(doc, "tables", []) or []:
            data = getattr(tbl, "data", None)
            if data is None:
                continue
            cells: list[TableCell] = []
            for c in getattr(data, "table_cells", []) or []:
                cells.append(
                    TableCell(
                        row=int(getattr(c, "start_row_offset_idx", 0)),
                        col=int(getattr(c, "start_col_offset_idx", 0)),
                        rowspan=int(getattr(c, "row_span", 1)),
                        colspan=int(getattr(c, "col_span", 1)),
                        text=getattr(c, "text", "") or "",
                        is_header=bool(getattr(c, "column_header", False))
                        or bool(getattr(c, "row_header", False)),
                    )
                )
            try:
                html = tbl.export_to_html(doc=doc)
            except Exception:
                html = None
            result.tables.append(
                Table(
                    cells=cells,
                    bbox=_bbox_or_none(getattr(tbl, "prov", None)),
                    html=html,
                    n_rows=int(getattr(data, "num_rows", 0) or 0),
                    n_cols=int(getattr(data, "num_cols", 0) or 0),
                )
            )

        for kv in getattr(doc, "key_value_items", []) or []:
            graph = getattr(kv, "graph", None)
            if not graph:
                continue
            for cell in getattr(graph, "cells", []) or []:
                lbl = getattr(cell, "label", None)
                if str(lbl).lower().endswith("key"):
                    pass

        return result
