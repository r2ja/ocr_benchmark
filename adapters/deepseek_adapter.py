"""DeepSeek-OCR-2 — vLLM-hosted on H200 (OpenAI-compatible).

Q8 weights alone are ~3 GB (before any compute buffer), so a 4 GB consumer
GPU OOMs at warmup. On a single H200 the BF16 weights + buffer trivially fit
and the MoE active params (~570 M) make per-page latency competitive.

The H200 runbook brings up vLLM with:

    vllm serve deepseek-ai/DeepSeek-OCR --port 8000 --dtype bfloat16 \\
        --trust-remote-code --served-model-name deepseek-ai/DeepSeek-OCR

and exports DEEPSEEK_BASE_URL=http://localhost:8000/v1 before invoking
`run_eval --stack deepseek`. The trust-remote-code flag is required because
the OCR-2 family ships its custom modeling code in the repo.
"""
from __future__ import annotations

from .openai_compatible_base import OpenAICompatibleVLMAdapter


class DeepSeekOCRAdapter(OpenAICompatibleVLMAdapter):
    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    DEFAULT_MODEL_SLUG = "deepseek-ai/DeepSeek-OCR"
    DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
    ENV_PREFIX = "DEEPSEEK"
