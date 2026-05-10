"""Compute per-axis scores by joining raw stack outputs to gold truth.

Mirrors the Azure Document Intelligence feature surface as much as possible
given what we can score automatically. Per-page per-stack we report:

  - text  : CER vs gold transcription (Read API)
  - kv    : F1 vs structured key-value gold (General Document API)
  - tables: count, total cells, max rows/cols (Layout API — counts have no
            gold dependency); plus structure-vs-Docling agreement on SEC pages
            (Docling treated as a high-quality structural reference)
  - structure_richness: total layout/text blocks per page (proxy for whether
            the stack produces structured output at all)

Output: results/scores_local.parquet + a printed multi-axis matrix.

Usage:
    python -m scripts.score_results
    python -m scripts.score_results --stacks docling,qwen-32b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import rapidfuzz

from metrics.kv_metrics import kv_f1
from metrics.ocr_metrics import cer

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
CORPUS_ROOT = REPO_ROOT / "corpus"

GOLD_KV_PAGES = {
    "funsd-form-01": CORPUS_ROOT / "funsd" / "form_01.gold.json",
    "cord-receipt-01": CORPUS_ROOT / "cord" / "receipt_01.gold.json",
}
GOLD_TEXT_PAGES = {
    "iam-handwriting-01": CORPUS_ROOT / "iam" / "line_01.gold.json",
}


def _load_gold_kv_funsd(gold: dict) -> list[tuple[str, str]]:
    return [(p["key"].rstrip(":").strip(), p["value"].strip()) for p in gold.get("kv_pairs", [])]


def _load_gold_kv_cord(gold: dict) -> list[tuple[str, str]]:
    """Flatten CORD-v2 gt_parse top-level into KV pairs."""
    parse = gold.get("gt_parse") or {}
    out: list[tuple[str, str]] = []
    for item in parse.get("menu", []) or []:
        if isinstance(item, dict):
            if item.get("nm"):
                out.append(("item_name", str(item["nm"]).strip()))
            if item.get("price"):
                out.append(("item_price", str(item["price"]).strip()))
            if item.get("cnt"):
                out.append(("item_count", str(item["cnt"]).strip()))
    for section in ("sub_total", "total"):
        sec = parse.get(section)
        if isinstance(sec, dict):
            for k, v in sec.items():
                out.append((f"{section}_{k}", str(v).strip()))
    return out


def _predicted_kv(raw: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in raw.get("kv_pairs") or []:
        if isinstance(item, dict):
            out.append((str(item.get("key", "")), str(item.get("value", ""))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
    return out


def _table_summary(raw: dict) -> dict:
    """Per-page table summary that requires no gold."""
    tables = raw.get("tables") or []
    n_cells = sum(len(t.get("cells") or []) for t in tables)
    max_rows = max((t.get("n_rows") or 0 for t in tables), default=0)
    max_cols = max((t.get("n_cols") or 0 for t in tables), default=0)
    return {
        "n_tables": len(tables),
        "n_table_cells": n_cells,
        "max_table_rows": max_rows,
        "max_table_cols": max_cols,
    }


def _structure_summary(raw: dict) -> dict:
    """Counts of structured artifacts produced (no gold needed)."""
    return {
        "n_text_blocks": len(raw.get("text_blocks") or []),
        "n_layout_blocks": len(raw.get("layout") or []),
        "n_kv_pairs": len(raw.get("kv_pairs") or []),
        "raw_text_chars": len(raw.get("raw_text") or ""),
    }


def _table_agreement_with_reference(stack_tables: list[dict], ref_tables: list[dict]) -> dict:
    """Coarse structural agreement: do count/dimensions match a reference stack?

    We use Docling's tables as the structural reference for SEC pages where
    no published gold exists. Output:
      ref_n_tables, table_count_match, dim_jaccard
    Not a true TEDS — just a sanity check on whether a stack reconstructs
    the same number/shape of tables Docling did.
    """
    ref_n = len(ref_tables)
    pred_n = len(stack_tables)
    count_match = 1.0 if ref_n == pred_n else (1.0 - abs(ref_n - pred_n) / max(ref_n, pred_n, 1))

    def shapes(ts):
        return {(t.get("n_rows") or 0, t.get("n_cols") or 0) for t in ts}

    ref_shapes = shapes(ref_tables)
    pred_shapes = shapes(stack_tables)
    if not ref_shapes and not pred_shapes:
        jaccard = 1.0
    elif not ref_shapes or not pred_shapes:
        jaccard = 0.0
    else:
        inter = len(ref_shapes & pred_shapes)
        union = len(ref_shapes | pred_shapes)
        jaccard = inter / union
    return {
        "ref_n_tables": ref_n,
        "pred_n_tables": pred_n,
        "table_count_match": round(count_match, 3),
        "shape_jaccard": round(jaccard, 3),
    }


def _table_content_vs_reference(stack_tables: list[dict], ref_tables: list[dict]) -> float | None:
    """Average fuzzy similarity of table cell content against the reference.

    Aligns the largest table from each side and compares cell-by-cell.
    Returns None if either side has no tables.
    """
    if not stack_tables or not ref_tables:
        return None
    pred_t = max(stack_tables, key=lambda t: len(t.get("cells") or []))
    ref_t = max(ref_tables, key=lambda t: len(t.get("cells") or []))

    def by_pos(t):
        return {(c.get("row", 0), c.get("col", 0)): (c.get("text") or "").strip() for c in t.get("cells") or []}

    pred = by_pos(pred_t)
    ref = by_pos(ref_t)
    if not ref:
        return None
    sims = []
    for pos, ref_text in ref.items():
        pred_text = pred.get(pos, "")
        if not ref_text and not pred_text:
            continue
        sims.append(rapidfuzz.fuzz.ratio(pred_text.lower(), ref_text.lower()) / 100.0)
    return round(sum(sims) / len(sims), 3) if sims else None


def score_page(stack: str, page: str, raw: dict, docling_raw: dict | None) -> dict:
    row: dict = {
        "stack": stack,
        "page": page,
        "latency_ms": round(raw.get("latency_ms") or 0.0, 1),
        "error": raw.get("error"),
    }
    row.update(_structure_summary(raw))
    row.update(_table_summary(raw))

    # KV — exact / fuzzy F1 vs structured gold
    if page in GOLD_KV_PAGES:
        gold = json.loads(GOLD_KV_PAGES[page].read_text(encoding="utf-8"))
        gold_kv = _load_gold_kv_funsd(gold) if page == "funsd-form-01" else _load_gold_kv_cord(gold)
        s = kv_f1(_predicted_kv(raw), gold_kv, fuzzy_threshold=0.85)
        row.update(
            kv_f1=round(s.f1, 3),
            kv_precision=round(s.precision, 3),
            kv_recall=round(s.recall, 3),
            n_gold_kv=s.n_gold,
        )

    # Text — CER vs transcription gold
    if page in GOLD_TEXT_PAGES:
        gold = json.loads(GOLD_TEXT_PAGES[page].read_text(encoding="utf-8"))
        ref = (gold.get("text") or "").strip()
        hyp = (raw.get("raw_text") or "").strip()
        row.update(
            cer=round(cer(ref, hyp), 3),
            ref_chars=len(ref),
            hyp_chars=len(hyp),
        )

    # Tables — structural agreement with Docling on pages where Docling is
    # the de-facto reference (SEC 10-Ks, fintabnet). Skip self-comparison.
    if docling_raw is not None and stack != "docling":
        agree = _table_agreement_with_reference(
            raw.get("tables") or [], docling_raw.get("tables") or []
        )
        row.update(agree)
        content_sim = _table_content_vs_reference(
            raw.get("tables") or [], docling_raw.get("tables") or []
        )
        if content_sim is not None:
            row["table_content_vs_docling"] = content_sim

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacks", default=None, help="comma-separated stack names")
    args = parser.parse_args()

    stacks = (
        args.stacks.split(",")
        if args.stacks
        else ["docling", "qwen-8b", "qwen-30b-a3b", "qwen-32b", "qwen-235b-a22b",
              "qianfan-ocr", "deepseek-ocr"]
    )

    # Pre-load Docling raw outputs as the structural reference
    docling_dir = RESULTS_ROOT / "docling" / "raw"
    docling_raws: dict[str, dict] = {}
    if docling_dir.exists():
        for f in docling_dir.glob("*.json"):
            docling_raws[f.stem] = json.loads(f.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for stack in stacks:
        raw_dir = RESULTS_ROOT / stack / "raw"
        if not raw_dir.exists():
            print(f"[skip] no raw dumps for stack={stack}")
            continue
        for raw_file in sorted(raw_dir.glob("*.json")):
            page = raw_file.stem
            raw = json.loads(raw_file.read_text(encoding="utf-8"))
            rows.append(score_page(stack, page, raw, docling_raws.get(page)))

    if not rows:
        print("no scoreable rows")
        return

    df = pd.DataFrame(rows)
    out = RESULTS_ROOT / "scores_local.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    print()

    # Per-stack aggregates across the multi-axis matrix.
    print("=" * 78)
    print("MULTI-AXIS BENCHMARK MATRIX (per-stack means)")
    print("=" * 78)

    agg_cols: dict[str, str] = {}
    if "kv_f1" in df.columns:
        agg_cols["kv_f1"] = "mean"
    if "cer" in df.columns:
        agg_cols["cer"] = "mean"
    if "n_tables" in df.columns:
        agg_cols["n_tables"] = "sum"
    if "n_table_cells" in df.columns:
        agg_cols["n_table_cells"] = "sum"
    if "shape_jaccard" in df.columns:
        agg_cols["shape_jaccard"] = "mean"
    if "table_content_vs_docling" in df.columns:
        agg_cols["table_content_vs_docling"] = "mean"
    if "latency_ms" in df.columns:
        agg_cols["latency_ms"] = "median"

    agg = df.groupby("stack").agg(agg_cols).round(3)
    print(agg.to_string())
    print()

    # Per-page-per-stack detailed table
    cols_to_show = [
        "stack", "page", "latency_ms",
        "n_tables", "n_table_cells", "n_kv_pairs", "raw_text_chars",
    ]
    if "kv_f1" in df.columns:
        cols_to_show.append("kv_f1")
    if "cer" in df.columns:
        cols_to_show.append("cer")
    if "shape_jaccard" in df.columns:
        cols_to_show.append("shape_jaccard")
    if "table_content_vs_docling" in df.columns:
        cols_to_show.append("table_content_vs_docling")

    cols_present = [c for c in cols_to_show if c in df.columns]
    print("=" * 78)
    print("PER-PAGE-PER-STACK")
    print("=" * 78)
    print(df[cols_present].to_string(index=False))


if __name__ == "__main__":
    main()
