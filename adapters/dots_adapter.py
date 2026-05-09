"""dots.ocr-1.5 — vLLM-hosted on H200 (OpenAI-compatible).

Local 4 GB GPU was infeasible: the vision encoder compute buffer alone is
3.65 GB at the model's native 1288×1288 input — see docs/findings.md for the
empirical OOM trace. On a single H200 (141 GB) BF16 weights + compute buffer
fit with ~100× headroom.

The H200 runbook brings up vLLM with:

    vllm serve rednote-hilab/dots.ocr --port 8000 --dtype bfloat16 \\
        --served-model-name rednote-hilab/dots.ocr

and exports DOTS_BASE_URL=http://localhost:8000/v1 before invoking
`run_eval --stack dots`. No further wiring needed.
"""
from __future__ import annotations

from .openai_compatible_base import OpenAICompatibleVLMAdapter


class DotsAdapter(OpenAICompatibleVLMAdapter):
    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    DEFAULT_MODEL_SLUG = "rednote-hilab/dots.ocr"
    DEFAULT_API_KEY_ENV = "DOTS_API_KEY"
    ENV_PREFIX = "DOTS"
