"""Back-fill `tables` field in existing raw JSON dumps.

Earlier runs of the OpenAI-compatible VLM adapters stored only `raw_text` and
`kv_pairs`. Markdown tables in `raw_text` were not parsed into structured
`Table` objects, so `tables = []` even when the model output was rich. This
script re-runs the new `parse_markdown_tables` parser over every existing
raw JSON and writes the result back, so downstream scoring (`score_results.py`)
can compute table TEDS without re-paying OpenRouter calls.

Idempotent — safe to run repeatedly.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from dataclasses import asdict
from adapters.openai_compatible_base import parse_deepseek_grounding, parse_tables

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

# Stacks where we use the markdown adapter pattern. Docling builds tables
# directly via its own pipeline so we leave its raw JSONs alone.
MD_STACKS = ["qwen-8b", "qwen-30b-a3b", "qwen-32b", "qwen-235b-a22b", "qianfan-ocr", "deepseek-ocr"]


def reparse_one(raw_path: Path, stack: str) -> tuple[int, int]:
    """Return (tables_before, tables_after)."""
    d = json.loads(raw_path.read_text(encoding="utf-8"))
    before = len(d.get("tables") or [])
    text = d.get("raw_text") or ""
    if not text:
        return before, before

    if stack == "deepseek-ocr":
        grounded = parse_deepseek_grounding(text)
        if grounded["text_blocks"] or grounded["tables"] or grounded["layout"]:
            d["text_blocks"] = [asdict(b) for b in grounded["text_blocks"]]
            d["layout"] = [asdict(b) for b in grounded["layout"]]
            d["tables"] = [asdict(t) for t in grounded["tables"]]
            raw_path.write_text(json.dumps(d, default=str, indent=2))
            return before, len(grounded["tables"])
    parsed = parse_tables(text)
    d["tables"] = [asdict(t) for t in parsed]
    raw_path.write_text(json.dumps(d, default=str, indent=2))
    return before, len(parsed)


def main() -> None:
    total_before = total_after = 0
    for stack in MD_STACKS:
        raw_dir = RESULTS_ROOT / stack / "raw"
        if not raw_dir.exists():
            print(f"[skip] no raw dir for {stack}")
            continue
        for raw_path in sorted(raw_dir.glob("*.json")):
            before, after = reparse_one(raw_path, stack)
            total_before += before
            total_after += after
            delta = after - before
            sign = "+" if delta >= 0 else ""
            print(f"  {stack}/{raw_path.stem}: {before} -> {after} ({sign}{delta})")
    print()
    print(f"TOTAL tables: {total_before} -> {total_after} (delta {total_after - total_before:+d})")


if __name__ == "__main__":
    main()
