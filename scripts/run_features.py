"""Run every stack against every feature axis × its test page.

Each axis uses a tight, axis-specific prompt so the model output is easy to
parse into structured detections. We do NOT reuse the generic Markdown
prompt — the goal is to test whether each stack can detect the feature at
all, not whether it produces nice Markdown.

Output: one raw JSON per (stack, axis) at
    results/features/<stack>/<axis>.json
plus a parquet index at results/features_run.parquet.

Usage:
    python -m scripts.run_features                  # all stacks × all axes
    python -m scripts.run_features --stacks docling,qwen-32b
    python -m scripts.run_features --axes checkboxes,codes
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
FEATURES_DIR = REPO_ROOT / "corpus" / "features"

_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

ADAPTER_REGISTRY = {
    "docling":      ("adapters.docling_adapter",  "DoclingAdapter"),
    "qwen-8b":      ("adapters.qwen_adapter",     "QwenVLAdapter"),
    "qwen-30b-a3b": ("adapters.qwen_adapter",     "QwenVLAdapter"),
    "qwen-32b":     ("adapters.qwen_adapter",     "QwenVLAdapter"),
    "qwen-235b-a22b":("adapters.qwen_adapter",    "QwenVLAdapter"),
    "qianfan-ocr":  ("adapters.baidu_adapter",    "BaiduQianfanOCRAdapter"),
    "deepseek-ocr": ("adapters.deepseek_adapter", "DeepSeekOCRAdapter"),
    "pyzbar":       ("adapters.pyzbar_adapter",   "PyzbarSpecialistAdapter"),
}

QWEN_SLUG_OVERRIDES = {
    "qwen-32b":       "qwen/qwen3-vl-32b-instruct",
    "qwen-8b":        "qwen/qwen3-vl-8b-instruct",
    "qwen-30b-a3b":   "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen-235b-a22b": "qwen/qwen3-vl-235b-a22b-instruct",
}

AXIS_PROMPTS = {
    "checkboxes": (
        "List every checkbox visible in this image. For each, output exactly one "
        "line in the format:\n  LABEL :: CHECKED\n  LABEL :: UNCHECKED\n"
        "Where LABEL is the text next to the checkbox and the state is whether "
        "the box appears filled. Output nothing else — no preamble, no Markdown, "
        "no commentary. If there are no checkboxes, output the single line: "
        "NO_CHECKBOXES_FOUND"
    ),
    "signatures": (
        "List every signature, stamp, or seal visible in this image. For each, "
        "output exactly one line in the format:\n  SIGNER_NAME :: ROLE_OR_TITLE\n"
        "Use the printed name below the signature line if present, otherwise "
        "describe the signature glyph. Output nothing else — no preamble, no "
        "Markdown. If there are no signatures, output: NO_SIGNATURES_FOUND"
    ),
    "formulas": (
        "Extract every mathematical formula or equation visible in this image as "
        "LaTeX. Output one formula per line, no other text. Do not include the "
        "$ or \\[ delimiters — just the LaTeX. If there are no formulas, output: "
        "NO_FORMULAS_FOUND"
    ),
    "codes": (
        "Detect every QR code, barcode, or other machine-readable code visible "
        "in this image. For each, output one line in the format:\n  TYPE :: DECODED_VALUE\n"
        "TYPE should be one of qr, ean13, ean8, code128, datamatrix, etc. "
        "DECODED_VALUE should be the exact text/URL the code encodes. If you "
        "cannot decode the value, output the type and `UNDECODABLE`. If there "
        "are no codes, output: NO_CODES_FOUND"
    ),
    "receipt_schema": (
        "This image is a receipt. Extract the following fields and return ONLY "
        "a single valid JSON object with these exact keys, no other text, no "
        "Markdown, no commentary:\n"
        "{\"items\": [{\"name\": \"\", \"quantity\": \"\", \"price\": \"\"}], "
        "\"subtotal\": \"\", \"tax\": \"\", \"total\": \"\"}\n"
        "Use null for any field that's not present. Numeric values stay as "
        "strings exactly as printed."
    ),
    "invoice_schema": (
        "This image is an invoice. Extract the following fields and return ONLY "
        "a single valid JSON object with these exact keys, no other text, no "
        "Markdown, no commentary:\n"
        "{\"invoice_number\": \"\", \"invoice_date\": \"\", \"seller_name\": \"\", "
        "\"client_name\": \"\", \"items\": [{\"description\": \"\", \"quantity\": \"\", "
        "\"price\": \"\"}], \"total\": \"\"}\n"
        "Use null for any field that's not present. Strings stay exactly as "
        "printed. Date in MM/DD/YYYY if visible."
    ),
}

# Schema-axes reuse pages from the general corpus (CORD receipt + DocILE invoice).
CORPUS_DIR = REPO_ROOT / "corpus"
AXIS_TO_PAGE = {
    "checkboxes":      ("checkboxes_w9",   FEATURES_DIR / "checkboxes_w9.png"),
    "signatures":      ("signatures_jpm",  FEATURES_DIR / "signatures_jpm.png"),
    "formulas":        ("formulas_arxiv",  FEATURES_DIR / "formulas_arxiv.png"),
    "codes":           ("codes_synthetic", FEATURES_DIR / "codes_synthetic.png"),
    "receipt_schema":  ("cord-receipt-01", CORPUS_DIR / "cord" / "receipt_01.png"),
    "invoice_schema":  ("docile-invoice-01", CORPUS_DIR / "docile" / "invoice_01.png"),
}


def _load_adapter(stack_id: str, prompt: str):
    module_path, cls_name = ADAPTER_REGISTRY[stack_id]
    cls = getattr(importlib.import_module(module_path), cls_name)
    if stack_id in QWEN_SLUG_OVERRIDES:
        return cls(model_slug=QWEN_SLUG_OVERRIDES[stack_id], prompt=prompt)
    if stack_id == "docling":
        # Docling has no prompt knob; it produces structured layout/tables/text.
        # We score against the raw text it returns, just like other axes.
        return cls()
    if stack_id == "pyzbar":
        # pyzbar specialist: no prompt either, just decodes machine-readable codes.
        return cls()
    return cls(prompt=prompt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacks", default=None)
    parser.add_argument("--axes", default=None)
    args = parser.parse_args()

    stacks = args.stacks.split(",") if args.stacks else list(ADAPTER_REGISTRY)
    axes = args.axes.split(",") if args.axes else list(AXIS_PROMPTS)

    rows: list[dict] = []
    for stack in stacks:
        if stack not in ADAPTER_REGISTRY:
            print(f"[skip] unknown stack: {stack}")
            continue
        out_dir = RESULTS_ROOT / "features" / stack
        out_dir.mkdir(parents=True, exist_ok=True)

        for axis in axes:
            if axis not in AXIS_PROMPTS:
                print(f"[skip] unknown axis: {axis}")
                continue
            page_id, image_path = AXIS_TO_PAGE[axis]
            if not image_path.exists():
                print(f"[skip] missing image: {image_path}")
                continue

            prompt = AXIS_PROMPTS[axis]
            print(f"[run] stack={stack:18s} axis={axis:12s} page={page_id}")
            try:
                adapter = _load_adapter(stack, prompt)
                adapter.warmup()
                t0 = time.perf_counter()
                result = adapter.process_page(image_path, page_id)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                result.latency_ms = result.latency_ms or latency_ms
            except Exception as e:
                print(f"  EXCEPTION {type(e).__name__}: {e}")
                rows.append({
                    "stack": stack, "axis": axis, "page_id": page_id,
                    "error": f"{type(e).__name__}: {e}",
                    "raw_text_chars": 0, "latency_ms": None,
                })
                continue

            raw_path = out_dir / f"{axis}.json"
            raw_path.write_text(json.dumps(result.to_dict(), default=str, indent=2))

            rows.append({
                "stack": stack,
                "axis": axis,
                "page_id": page_id,
                "raw_text_chars": len(result.raw_text or ""),
                "latency_ms": round(result.latency_ms or 0.0, 1),
                "error": result.error,
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })

    if rows:
        out = RESULTS_ROOT / f"features_run_{int(time.time())}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"\n[features] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
