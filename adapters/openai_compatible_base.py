"""Shared base for VLM adapters that speak OpenAI-compatible chat completions.

This is the wire format for OpenRouter, vLLM ServingRuntime (RHOAI / RunPod),
llama-server, Together, and most other inference hosts. By keeping every VLM
adapter on this base we get one path to test/debug, and the same code that
runs against OpenRouter today runs against an H200 vLLM pod tomorrow with
only `base_url` and `model_slug` changing.

Subclasses set defaults; environment variables override at runtime so the
H200 runbook can point all adapters at `localhost:8000` without code changes:

    <STACK>_BASE_URL    e.g. DOTS_BASE_URL=http://localhost:8000/v1
    <STACK>_MODEL_SLUG  e.g. DOTS_MODEL_SLUG=rednote-hilab/dots.ocr
    <STACK>_API_KEY     e.g. DOTS_API_KEY=EMPTY  (vLLM ignores it)
"""
from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from pathlib import Path

from .base import StackAdapter
from .schema import KVPair, PageResult


GENERIC_PROMPT = (
    "Extract the full content of this document image as Markdown. Preserve "
    "tables (use Markdown table syntax), headers, lists, and reading order. "
    "If the document contains form fields or key-value pairs, list them at "
    "the end under a 'Key-Value Pairs:' section, one per line as 'KEY :: VALUE'."
)


def _image_data_uri(image_path: Path) -> str:
    """Return a data:image/png;base64,... URI for the page.

    Rasterizes the first page of a PDF if needed.
    """
    suffix = image_path.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(image_path))
        try:
            page = pdf[0]
            pil = page.render(scale=200 / 72.0).to_pil()
            buf = BytesIO()
            pil.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        finally:
            pdf.close()
    if suffix == ".png":
        return f"data:image/png;base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
    return f"data:image/jpeg;base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


class OpenAICompatibleVLMAdapter(StackAdapter):
    """Subclasses set the four DEFAULT_* class attrs."""

    DEFAULT_BASE_URL: str = ""
    DEFAULT_MODEL_SLUG: str = ""
    DEFAULT_API_KEY_ENV: str = ""
    DEFAULT_PROMPT: str = GENERIC_PROMPT
    ENV_PREFIX: str = ""  # e.g. "DOTS" → reads DOTS_BASE_URL, DOTS_MODEL_SLUG

    HTTP_REFERER: str = "https://logarithmtech.example"
    APP_TITLE: str = "Logarithm DocIntel Benchmark"
    MAX_TOKENS: int = 4096
    REQUEST_TIMEOUT: int = 240

    def __init__(
        self,
        model_slug: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.model_slug = (
            model_slug
            or os.environ.get(f"{self.ENV_PREFIX}_MODEL_SLUG")
            or self.DEFAULT_MODEL_SLUG
        )
        self.base_url = (
            base_url
            or os.environ.get(f"{self.ENV_PREFIX}_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        self.api_key = (
            api_key
            or os.environ.get(f"{self.ENV_PREFIX}_API_KEY")
            or os.environ.get(self.DEFAULT_API_KEY_ENV, "")
        )
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.stack_id = self._derive_stack_id(self.model_slug)
        self.model_revision = self.model_slug

    @staticmethod
    def _derive_stack_id(slug: str) -> str:
        last = slug.split("/")[-1]
        if last.endswith("-instruct"):
            last = last[: -len("-instruct")]
        return last

    def warmup(self) -> None:
        if not self.base_url:
            raise RuntimeError(
                f"{self.__class__.__name__}: base_url not set. "
                f"Set {self.ENV_PREFIX}_BASE_URL or pass base_url=..."
            )

    def _build_messages(self, data_uri: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

    def _build_headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        # OpenRouter-only attribution headers; harmless on vLLM/llama-server.
        h["HTTP-Referer"] = self.HTTP_REFERER
        h["X-Title"] = self.APP_TITLE
        return h

    def process_page(self, image_path: Path, page_id: str) -> PageResult:
        import requests

        result = PageResult(
            page_id=page_id,
            stack_id=self.stack_id,
            model_revision=self.model_revision,
        )
        try:
            data_uri = _image_data_uri(image_path)
            payload = {
                "model": self.model_slug,
                "messages": self._build_messages(data_uri),
                "max_tokens": self.MAX_TOKENS,
                "temperature": 0.0,
            }
            url = self.base_url.rstrip("/") + "/chat/completions"
            t0 = time.perf_counter()
            r = requests.post(
                url,
                json=payload,
                headers=self._build_headers(),
                timeout=self.REQUEST_TIMEOUT,
            )
            result.latency_ms = (time.perf_counter() - t0) * 1000.0
            if r.status_code != 200:
                result.error = f"HTTP {r.status_code}: {r.text[:300]}"
                return result
            body = r.json()
            text = body["choices"][0]["message"]["content"]
            self._populate_from_text(text, result)
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        return result

    def _populate_from_text(self, text: str, result: PageResult) -> None:
        result.raw_text = text
        if "Key-Value Pairs:" in text:
            kv_block = text.split("Key-Value Pairs:", 1)[1].strip()
            for line in kv_block.splitlines():
                if "::" in line:
                    k, v = line.split("::", 1)
                    result.kv_pairs.append(KVPair(key=k.strip(), value=v.strip()))
