"""Run one stack across the mini-corpus, write a parquet of per-page results.

Usage:
    python -m scripts.run_eval --stack docling
    python -m scripts.run_eval --stack qwen-32b --pages sec-10k-tech-01,docile-invoice-01

STATUS: skeleton. Wires together adapters + corpus manifest + parquet writer.
Per-axis scoring is invoked from `scripts/score_results.py` after this writes
the raw outputs — keeps the run loop pure I/O so we don't re-run the model on
metric bugs.
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from corpus_meta.corpus_manifest import MINI_CORPUS, CorpusPage

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

ADAPTER_REGISTRY = {
    "docling":   ("adapters.docling_adapter",  "DoclingAdapter"),
    "paddle-vl": ("adapters.paddle_vl_adapter", "PaddleVLAdapter"),
    "dots":      ("adapters.dots_adapter",      "DotsAdapter"),
    "deepseek":  ("adapters.deepseek_adapter",  "DeepSeekOCRAdapter"),
    "qwen-32b":  ("adapters.qwen_adapter",      "QwenVLAdapter"),
    # Optional Qwen size variants (same adapter, different OpenRouter slug)
    "qwen-8b":   ("adapters.qwen_adapter",      "QwenVLAdapter"),
    "qwen-30b-a3b":  ("adapters.qwen_adapter",  "QwenVLAdapter"),
    "qwen-235b-a22b":("adapters.qwen_adapter",  "QwenVLAdapter"),
}

# For the Qwen-* variants, override the slug at instantiation time.
QWEN_SLUG_OVERRIDES = {
    "qwen-32b":       "qwen/qwen3-vl-32b-instruct",
    "qwen-8b":        "qwen/qwen3-vl-8b-instruct",
    "qwen-30b-a3b":   "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen-235b-a22b": "qwen/qwen3-vl-235b-a22b-instruct",
}


def _load_adapter(stack_id: str):
    if stack_id not in ADAPTER_REGISTRY:
        raise SystemExit(f"unknown stack: {stack_id}. options: {list(ADAPTER_REGISTRY)}")
    module_path, cls_name = ADAPTER_REGISTRY[stack_id]
    cls = getattr(importlib.import_module(module_path), cls_name)
    if stack_id in QWEN_SLUG_OVERRIDES:
        return cls(model_slug=QWEN_SLUG_OVERRIDES[stack_id])
    return cls() if callable(cls) else cls


def _filter_corpus(page_filter: str | None) -> list[CorpusPage]:
    if not page_filter:
        return MINI_CORPUS
    wanted = set(page_filter.split(","))
    return [p for p in MINI_CORPUS if p.page_id in wanted]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", required=True, choices=list(ADAPTER_REGISTRY))
    parser.add_argument("--pages", default=None, help="comma-separated page_ids; default = all")
    parser.add_argument("--dry-run", action="store_true", help="list what would run, don't call the model")
    args = parser.parse_args()

    pages = _filter_corpus(args.pages)
    print(f"[run_eval] stack={args.stack}  pages={[p.page_id for p in pages]}")
    if args.dry_run:
        return

    adapter = _load_adapter(args.stack)
    adapter.warmup()

    rows: list[dict] = []
    raw_dir = RESULTS_ROOT / args.stack / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for page in pages:
        image_path = REPO_ROOT / page.local_path
        if not image_path.exists():
            print(f"  [SKIP] {page.page_id}: corpus file not built yet ({image_path})")
            continue

        t0 = time.perf_counter()
        result = adapter.process_page(image_path, page.page_id)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result.latency_ms = result.latency_ms or latency_ms

        raw_path = raw_dir / f"{page.page_id}.json"
        raw_path.write_text(json.dumps(result.to_dict(), default=str, indent=2))
        result.raw_response_path = str(raw_path.relative_to(REPO_ROOT))

        rows.append({
            "stack_id": result.stack_id,
            "model_revision": result.model_revision,
            "page_id": result.page_id,
            "latency_ms": result.latency_ms,
            "raw_response_path": result.raw_response_path,
            "error": result.error,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })

    if rows:
        out = RESULTS_ROOT / f"{args.stack}_run_{int(time.time())}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"[run_eval] wrote {out} ({len(rows)} pages)")
    else:
        print("[run_eval] no pages produced output")


if __name__ == "__main__":
    main()
