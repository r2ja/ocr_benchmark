"""Search Replicate's public catalog AND query HF Inference Providers for the
four model families we need. Reports actual resolvable model identifiers we
can wire into the adapters.

Run: python -m scripts.search_hosts
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)


def search_replicate(query: str, limit: int = 15) -> list[dict]:
    """Use the public Replicate search endpoint."""
    import requests

    token = os.environ.get("REPLICATE_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = "https://api.replicate.com/v1/models"
    params = {"q": query}
    out: list[dict] = []
    for _ in range(2):  # at most one page-of-next
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for m in data.get("results", []):
            out.append({
                "owner": m.get("owner"),
                "name": m.get("name"),
                "slug": f"{m.get('owner')}/{m.get('name')}",
                "description": (m.get("description") or "")[:120],
            })
            if len(out) >= limit:
                return out
        url = data.get("next")
        params = None
        if not url:
            break
    return out


def search_hf_models(query: str, limit: int = 10) -> list[dict]:
    """Hit the HF Hub model search."""
    import requests

    r = requests.get(
        "https://huggingface.co/api/models",
        params={"search": query, "limit": limit, "full": "true"},
        timeout=30,
    )
    r.raise_for_status()
    out = []
    for m in r.json():
        # Inference providers are listed as `inferenceProviderMapping` (newer)
        # or implied by the `pipeline_tag` / `inference` fields.
        providers = list((m.get("inferenceProviderMapping") or {}).keys())
        out.append({
            "id": m.get("id") or m.get("modelId"),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "pipeline_tag": m.get("pipeline_tag"),
            "providers": providers,
        })
    return out


def main() -> int:
    _load_env()
    queries = {
        "qwen3-vl": "qwen3-vl",
        "qwen2.5-vl": "qwen2.5-vl",
        "deepseek-ocr": "deepseek-ocr",
        "dots.ocr": "dots.ocr",
        "paddleocr-vl": "paddleocr-vl",
    }

    print("=" * 60)
    print("REPLICATE SEARCH")
    print("=" * 60)
    for label, q in queries.items():
        print(f"\n[{label}]  query='{q}'")
        try:
            hits = search_replicate(q, limit=10)
            if not hits:
                print("  (no results)")
            for h in hits:
                print(f"  {h['slug']:50s}  {h['description']}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    print()
    print("=" * 60)
    print("HUGGING FACE — checking inferenceProviderMapping")
    print("=" * 60)
    for label, q in queries.items():
        print(f"\n[{label}]  query='{q}'")
        try:
            hits = search_hf_models(q, limit=5)
            for h in hits:
                providers = ",".join(h["providers"]) if h["providers"] else "no-providers"
                print(f"  {h['id']:60s}  dl={h['downloads']:<8}  providers={providers}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
