"""One-shot verification: confirm Replicate has model endpoints for the slugs
we plan to use, BEFORE the user deposits the $5.

Usage:
    set REPLICATE_API_TOKEN=... (or put it in .env)
    python -m scripts.verify_replicate

Lists each candidate slug and reports whether it resolves on Replicate. Does
NOT run any inference — just metadata fetches, which are free for token
holders.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


CANDIDATES = {
    "qwen3-vl-32b": [
        "qwen/qwen3-vl-32b-instruct",
        "lucataco/qwen3-vl-32b",
        "lucataco/qwen3-vl",
    ],
    "deepseek-ocr-2": [
        "deepseek-ai/deepseek-ocr-2",
        "lucataco/deepseek-ocr",
        "deepseek-ai/deepseek-ocr",
    ],
    "dots-ocr": [
        "rednote-hilab/dots.ocr",
        "lucataco/dots.ocr",
    ],
    "paddleocr-vl": [
        "paddlepaddle/paddleocr-vl",
        "lucataco/paddleocr-vl",
    ],
}


def main() -> int:
    # Load .env if it exists, but don't crash if it doesn't
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("ERROR: REPLICATE_API_TOKEN not set. Put it in .env or export it.")
        return 1

    import replicate
    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    any_missing = False
    for stack, slugs in CANDIDATES.items():
        print(f"\n[{stack}]")
        any_resolved = False
        for slug in slugs:
            try:
                m = client.models.get(slug)
                latest = getattr(m, "latest_version", None)
                ver = getattr(latest, "id", "(no version)") if latest else "(no version)"
                print(f"  OK  {slug:50s}  latest_version={ver[:12]}")
                any_resolved = True
            except Exception as e:
                cls = type(e).__name__
                msg = str(e).splitlines()[0][:120] if str(e) else "(no message)"
                print(f"  --  {slug:50s}  {cls}: {msg}")
        if not any_resolved:
            any_missing = True
            print(f"  ** NO RESOLVED SLUG for {stack} — needs alternate (HF Inference Endpoints, Modal, or self-host)")
    return 1 if any_missing else 0


if __name__ == "__main__":
    sys.exit(main())
