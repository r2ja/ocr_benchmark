"""Compute per-axis scores by joining raw stack outputs to gold truth.

For each (stack, page) we load the raw JSON dump in `results/<stack>/raw/`
and the gold-truth file at `corpus/<source>/<page>.gold.json`. We compute:

  - KV F1   on funsd (clean kv_pairs gold) and cord (gt_parse flattened)
  - CER     on iam (clean transcription gold) and where text-gold is available
  - Manual rubric flag on the rest (sec-10k, fintabnet, omnidocbench, docile)

Output: results/scores_local.parquet + a printed summary table.

Usage:
    python -m scripts.score_results
    python -m scripts.score_results --stacks docling,qwen-32b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from metrics.kv_metrics import kv_f1
from metrics.ocr_metrics import cer

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
CORPUS_ROOT = REPO_ROOT / "corpus"

PAGE_TO_GOLD = {
    "funsd-form-01": CORPUS_ROOT / "funsd" / "form_01.gold.json",
    "iam-handwriting-01": CORPUS_ROOT / "iam" / "line_01.gold.json",
    "cord-receipt-01": CORPUS_ROOT / "cord" / "receipt_01.gold.json",
}


def _load_gold_kv_funsd(gold: dict) -> list[tuple[str, str]]:
    return [(p["key"].rstrip(":").strip(), p["value"].strip()) for p in gold.get("kv_pairs", [])]


def _load_gold_kv_cord(gold: dict) -> list[tuple[str, str]]:
    """Flatten CORD-v2 gt_parse top-level into KV pairs.

    gt_parse contains menu (list of items with nm/price/num), sub_total, total.
    We extract: per-item name/price, plus top-level subtotal/total fields.
    """
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


def _load_gold_text_iam(gold: dict) -> str:
    return (gold.get("text") or "").strip()


def _predicted_kv(raw: dict) -> list[tuple[str, str]]:
    """Adapter outputs kv_pairs as either list[KVPair-dict] or list[tuple]."""
    out: list[tuple[str, str]] = []
    for item in raw.get("kv_pairs") or []:
        if isinstance(item, dict):
            out.append((str(item.get("key", "")), str(item.get("value", ""))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
    return out


def score_one(stack: str, page: str, raw: dict) -> dict | None:
    gold_path = PAGE_TO_GOLD.get(page)
    if not gold_path or not gold_path.exists():
        return None
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    pred_kv = _predicted_kv(raw)
    out = {"stack": stack, "page": page}

    if page == "funsd-form-01":
        gold_kv = _load_gold_kv_funsd(gold)
        s = kv_f1(pred_kv, gold_kv, fuzzy_threshold=0.85)
        out.update(
            axis="kv",
            kv_f1=s.f1,
            kv_precision=s.precision,
            kv_recall=s.recall,
            n_pred_kv=s.n_pred,
            n_gold_kv=s.n_gold,
        )
        return out

    if page == "cord-receipt-01":
        gold_kv = _load_gold_kv_cord(gold)
        s = kv_f1(pred_kv, gold_kv, fuzzy_threshold=0.85)
        out.update(
            axis="kv",
            kv_f1=s.f1,
            kv_precision=s.precision,
            kv_recall=s.recall,
            n_pred_kv=s.n_pred,
            n_gold_kv=s.n_gold,
        )
        return out

    if page == "iam-handwriting-01":
        ref = _load_gold_text_iam(gold)
        hyp = (raw.get("raw_text") or "").strip()
        out.update(axis="text", cer=cer(ref, hyp), ref_chars=len(ref), hyp_chars=len(hyp))
        return out

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacks", default=None, help="comma-separated stack names")
    args = parser.parse_args()

    stacks = (args.stacks.split(",") if args.stacks else
              ["docling", "qwen-8b", "qwen-30b-a3b", "qwen-32b", "qwen-235b-a22b"])

    rows: list[dict] = []
    for stack in stacks:
        raw_dir = RESULTS_ROOT / stack / "raw"
        if not raw_dir.exists():
            print(f"[skip] no raw dumps for stack={stack}")
            continue
        for raw_file in sorted(raw_dir.glob("*.json")):
            page = raw_file.stem
            raw = json.loads(raw_file.read_text(encoding="utf-8"))
            scored = score_one(stack, page, raw)
            if scored is not None:
                rows.append(scored)

    if not rows:
        print("no scoreable rows")
        return

    df = pd.DataFrame(rows)
    out = RESULTS_ROOT / "scores_local.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    print()
    print("=" * 70)
    print("PER-PAGE-PER-STACK SCORES")
    print("=" * 70)
    print(df.to_string(index=False))
    print()

    print("=" * 70)
    print("PER-STACK AGGREGATES (where applicable)")
    print("=" * 70)
    if "kv_f1" in df.columns:
        kv = df[df["axis"] == "kv"].groupby("stack")[["kv_f1", "kv_precision", "kv_recall"]].mean().round(3)
        print("\nKV (mean across {} pages):".format(df[df["axis"] == "kv"]["page"].nunique()))
        print(kv.to_string())
    if "cer" in df.columns:
        text = df[df["axis"] == "text"].groupby("stack")[["cer"]].mean().round(3)
        print("\nText CER (mean across {} pages):".format(df[df["axis"] == "text"]["page"].nunique()))
        print(text.to_string())


if __name__ == "__main__":
    main()
