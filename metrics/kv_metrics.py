"""Key-value extraction F1.

Two flavors:
  - exact: key + value must match exactly (after whitespace normalization)
  - fuzzy: value match uses normalized Levenshtein with a configurable threshold

We score per-page. Aggregation across the mini-corpus happens in the report layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import rapidfuzz


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


@dataclass
class KVScore:
    precision: float
    recall: float
    f1: float
    matched: int
    n_pred: int
    n_gold: int


def kv_f1(
    predicted: list[tuple[str, str]],
    gold: list[tuple[str, str]],
    fuzzy_threshold: float | None = None,
) -> KVScore:
    """Compute KV F1.

    fuzzy_threshold: if set (e.g. 0.85), value matches use normalized Levenshtein
    similarity >= threshold instead of exact equality. Keys are always exact.
    """
    pred_norm = [(_norm(k), _norm(v)) for k, v in predicted]
    gold_norm = [(_norm(k), _norm(v)) for k, v in gold]

    used_gold: set[int] = set()
    matched = 0
    for pk, pv in pred_norm:
        for i, (gk, gv) in enumerate(gold_norm):
            if i in used_gold or pk != gk:
                continue
            if fuzzy_threshold is None:
                if pv == gv:
                    used_gold.add(i)
                    matched += 1
                    break
            else:
                sim = rapidfuzz.fuzz.ratio(pv, gv) / 100.0
                if sim >= fuzzy_threshold:
                    used_gold.add(i)
                    matched += 1
                    break

    n_pred = len(pred_norm)
    n_gold = len(gold_norm)
    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_gold if n_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return KVScore(precision, recall, f1, matched, n_pred, n_gold)
