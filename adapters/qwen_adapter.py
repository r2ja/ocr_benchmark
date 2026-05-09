"""Qwen3-VL adapter — OpenAI-compatible chat completions.

Defaults to OpenRouter (verified live 2026-05-09 across the size sweep). The
H200 runbook can override at runtime via env vars to point at a local vLLM
ServingRuntime instead, without code changes:

    QWEN_BASE_URL=http://localhost:8000/v1
    QWEN_API_KEY=EMPTY

Available OpenRouter slugs (verified 2026-05-09):
    qwen/qwen3-vl-8b-instruct
    qwen/qwen3-vl-30b-a3b-instruct
    qwen/qwen3-vl-32b-instruct
    qwen/qwen3-vl-235b-a22b-instruct
"""
from __future__ import annotations

from .openai_compatible_base import OpenAICompatibleVLMAdapter


class QwenVLAdapter(OpenAICompatibleVLMAdapter):
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL_SLUG = "qwen/qwen3-vl-32b-instruct"
    DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
    ENV_PREFIX = "QWEN"
    DEFAULT_PROMPT = (
        "You are a document AI. Output the full document content as structured "
        "Markdown. Preserve tables (use Markdown table syntax), headers, lists, "
        "and reading order. If the document contains explicit form fields or "
        "key-value pairs, list them at the end under a 'Key-Value Pairs:' "
        "section, one per line in the format 'KEY :: VALUE'."
    )
