"""Table structure / content scoring.

We do NOT pull in the full TEDS reference implementation (apted-tree-edit-
distance is heavy and depends on lxml). For the 8-page mini-corpus we use a
lighter approach:

  - structure_score: cell-grid IoU (do row/col/rowspan/colspan match the gold
    grid?) — captures whether the table topology is right
  - content_score: row-wise normalized Levenshtein similarity averaged over
    rows — captures whether the cell text is right

Headline number = harmonic mean of structure_score and content_score, written
to parquet as `table_score`. Reproducible, transparent, and tractable to debug
when a row scores low.

If we later want canonical TEDS, we can add it as an optional dependency and
run it offline on the same per-page outputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import rapidfuzz


@dataclass
class TableCellGold:
    row: int
    col: int
    rowspan: int
    colspan: int
    text: str


@dataclass
class TableScore:
    structure_score: float
    content_score: float
    overall: float
    n_gold_cells: int
    n_pred_cells: int


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def score_table(
    predicted: list[TableCellGold],
    gold: list[TableCellGold],
) -> TableScore:
    pred_by_pos = {(c.row, c.col): c for c in predicted}
    gold_by_pos = {(c.row, c.col): c for c in gold}

    if not gold_by_pos:
        return TableScore(0.0, 0.0, 0.0, 0, len(pred_by_pos))

    struct_matches = 0
    content_sims: list[float] = []
    for pos, g in gold_by_pos.items():
        p = pred_by_pos.get(pos)
        if p is None:
            content_sims.append(0.0)
            continue
        if p.rowspan == g.rowspan and p.colspan == g.colspan:
            struct_matches += 1
        sim = rapidfuzz.fuzz.ratio(_norm(p.text), _norm(g.text)) / 100.0
        content_sims.append(sim)

    structure_score = struct_matches / len(gold_by_pos)
    content_score = sum(content_sims) / len(content_sims)
    if structure_score + content_score == 0:
        overall = 0.0
    else:
        overall = 2 * structure_score * content_score / (structure_score + content_score)

    return TableScore(
        structure_score=structure_score,
        content_score=content_score,
        overall=overall,
        n_gold_cells=len(gold_by_pos),
        n_pred_cells=len(pred_by_pos),
    )
