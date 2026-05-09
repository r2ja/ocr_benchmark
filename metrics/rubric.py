"""Manual scoring rubric — used where automated metrics are insufficient
(e.g. SEC 10-K pages without published ground truth).

Each axis scored 1-5; rubric defined in docs/rubric.md. This module only
provides the dataclass and a CSV/parquet helper so the manual scores join the
automated results in one place.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ManualRubric:
    page_id: str
    stack_id: str
    layout_fidelity: int       # 1-5
    table_quality: int         # 1-5
    kv_quality: int            # 1-5
    text_accuracy: int         # 1-5
    notes: str = ""

    def overall(self) -> float:
        return (
            self.layout_fidelity
            + self.table_quality
            + self.kv_quality
            + self.text_accuracy
        ) / 4.0
