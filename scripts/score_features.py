"""Score per-axis feature detection across all stacks.

Each axis has its own scoring rubric:

  checkboxes — detection F1 (label fuzzy-match) + state accuracy
  signatures — detection F1 (signer name fuzzy-match)
  formulas   — fuzzy-similarity of best-matching predicted line per gold formula
  codes      — exact-match decode of payload

Output: results/feature_scores.parquet + a printed matrix.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import rapidfuzz

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FEATURES_DIR = REPO_ROOT / "corpus" / "features"

STACKS = [
    "docling", "qwen-8b", "qwen-30b-a3b", "qwen-32b", "qwen-235b-a22b",
    "qianfan-ocr", "deepseek-ocr",
]
AXES = ["checkboxes", "signatures", "formulas", "codes"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _load_gold(axis: str) -> dict:
    fmap = {
        "checkboxes": "checkboxes_w9.gold.json",
        "signatures": "signatures_jpm.gold.json",
        "formulas":   "formulas_arxiv.gold.json",
        "codes":      "codes_synthetic.gold.json",
    }
    return json.loads((FEATURES_DIR / fmap[axis]).read_text(encoding="utf-8"))


def _load_pred(stack: str, axis: str) -> dict | None:
    p = RESULTS_ROOT / "features" / stack / f"{axis}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_kv_lines(text: str) -> list[tuple[str, str]]:
    """Pull every `KEY :: VALUE` line out of model output."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if "::" in line and not line.upper().startswith(("NO_CHECKBOXES", "NO_SIGNATURES", "NO_CODES")):
            k, v = line.split("::", 1)
            out.append((k.strip(), v.strip()))
    return out


def score_checkboxes(gold: dict, pred_text: str) -> dict:
    expected = gold["expected_checkboxes"]
    pred_pairs = _parse_kv_lines(pred_text)

    matched = 0
    state_correct = 0
    used_pred: set[int] = set()
    for g in expected:
        g_label = _norm(g["label"])
        best_i = -1
        best_sim = 0.0
        for i, (pk, _pv) in enumerate(pred_pairs):
            if i in used_pred:
                continue
            sim = rapidfuzz.fuzz.partial_ratio(_norm(pk), g_label) / 100.0
            if sim > best_sim:
                best_sim = sim
                best_i = i
        if best_sim >= 0.65 and best_i >= 0:
            used_pred.add(best_i)
            matched += 1
            pred_state = pred_pairs[best_i][1].strip().upper()
            if g["state"].upper() in pred_state:
                state_correct += 1

    n_gold = len(expected)
    n_pred = len(pred_pairs)
    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "axis": "checkboxes",
        "detection_f1": round(f1, 3),
        "state_accuracy": round(state_correct / max(matched, 1), 3) if matched else 0.0,
        "n_gold": n_gold, "n_pred": n_pred, "matched": matched,
    }


def score_signatures(gold: dict, pred_text: str) -> dict:
    expected = gold["expected_signatures"]
    pred_pairs = _parse_kv_lines(pred_text)

    matched = 0
    used: set[int] = set()
    for g in expected:
        g_name = _norm(g["signer"])
        best_i, best_sim = -1, 0.0
        for i, (pk, _pv) in enumerate(pred_pairs):
            if i in used:
                continue
            sim = rapidfuzz.fuzz.partial_ratio(_norm(pk), g_name) / 100.0
            if sim > best_sim:
                best_sim, best_i = sim, i
        if best_sim >= 0.65 and best_i >= 0:
            used.add(best_i)
            matched += 1

    n_gold = len(expected)
    n_pred = len(pred_pairs)
    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "axis": "signatures",
        "detection_f1": round(f1, 3),
        "n_gold": n_gold, "n_pred": n_pred, "matched": matched,
    }


def score_formulas(gold: dict, pred_text: str) -> dict:
    expected = gold["expected_formulas"]
    pred_lines = [ln.strip() for ln in (pred_text or "").splitlines() if ln.strip() and "NO_FORMULAS" not in ln.upper()]

    sims = []
    for g in expected:
        g_latex = _norm(g["latex"])
        best = 0.0
        for line in pred_lines:
            best = max(best, rapidfuzz.fuzz.partial_ratio(_norm(line), g_latex) / 100.0)
        sims.append(best)

    return {
        "axis": "formulas",
        "mean_similarity": round(sum(sims) / len(sims), 3) if sims else 0.0,
        "n_gold": len(expected),
        "n_pred": len(pred_lines),
        "matched": sum(1 for s in sims if s >= 0.6),
    }


def score_codes(gold: dict, pred_text: str) -> dict:
    expected = gold["expected_codes"]
    pred_pairs = _parse_kv_lines(pred_text)
    pred_payloads = [pv for (_pk, pv) in pred_pairs]

    matched = 0
    for g in expected:
        gp = g["payload"].strip()
        # Exact-match (code payloads should be decoded verbatim)
        if any(gp in pv or pv in gp for pv in pred_payloads):
            matched += 1

    n_gold = len(expected)
    n_pred = len(pred_pairs)
    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "axis": "codes",
        "exact_match_f1": round(f1, 3),
        "n_gold": n_gold, "n_pred": n_pred, "matched": matched,
    }


SCORERS = {
    "checkboxes": score_checkboxes,
    "signatures": score_signatures,
    "formulas":   score_formulas,
    "codes":      score_codes,
}


def main() -> None:
    rows: list[dict] = []
    for stack in STACKS:
        for axis in AXES:
            pred = _load_pred(stack, axis)
            if pred is None:
                continue
            gold = _load_gold(axis)
            if pred.get("error"):
                rows.append({"stack": stack, "axis": axis, "error": pred["error"]})
                continue
            raw_text = pred.get("raw_text") or ""
            scored = SCORERS[axis](gold, raw_text)
            scored["stack"] = stack
            scored["raw_text_chars"] = len(raw_text)
            scored["latency_ms"] = round(pred.get("latency_ms") or 0.0, 1)
            rows.append(scored)

    if not rows:
        print("no scoreable predictions")
        return

    df = pd.DataFrame(rows)
    out = RESULTS_ROOT / "feature_scores.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)\n")

    # Pivot per stack × axis with each axis's headline metric
    headline = {
        "checkboxes": "detection_f1",
        "signatures": "detection_f1",
        "formulas":   "mean_similarity",
        "codes":      "exact_match_f1",
    }
    print("=" * 78)
    print("FEATURE-AXIS MATRIX (each cell is the headline score for that axis)")
    print("=" * 78)
    pivot_cols = []
    for axis in AXES:
        col = f"{axis}_{headline[axis]}"
        sub = df[df["axis"] == axis][["stack", headline[axis]]].rename(columns={headline[axis]: col})
        pivot_cols.append(sub.set_index("stack"))
    pivot = pd.concat(pivot_cols, axis=1).reindex(STACKS)
    print(pivot.to_string())
    print()

    print("=" * 78)
    print("DETAIL — per row")
    print("=" * 78)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
