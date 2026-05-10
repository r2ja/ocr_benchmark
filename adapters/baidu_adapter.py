"""Baidu Qianfan-OCR-Fast adapter — via OpenRouter, free tier.

OpenRouter routes `baidu/qianfan-ocr-fast:free` as an image+text OCR model
at zero cost. Different product from PaddleOCR-VL but same Baidu family —
gives us another Chinese-vendor OCR data point in the bench-off matrix at
no incremental cost.

Available on OpenRouter (verified 2026-05-10):
    baidu/qianfan-ocr-fast:free   modalities=[image, text]   pricing=0/0
"""
from __future__ import annotations

from .openai_compatible_base import OpenAICompatibleVLMAdapter


class BaiduQianfanOCRAdapter(OpenAICompatibleVLMAdapter):
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL_SLUG = "baidu/qianfan-ocr-fast:free"
    DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
    ENV_PREFIX = "QIANFAN"
    DEFAULT_PROMPT = (
        "Extract the full content of this document image as Markdown. Preserve "
        "tables (use Markdown table syntax), headers, lists, and reading order. "
        "If the document contains form fields or key-value pairs, list them at "
        "the end under a 'Key-Value Pairs:' section, one per line as 'KEY :: VALUE'."
    )
