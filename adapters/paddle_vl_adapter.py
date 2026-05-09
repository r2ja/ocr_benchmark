"""PaddleOCR-VL-1.5 — H200 path with two architecture options.

PaddleOCR-VL is unique among the bench-off candidates: it's not a single VLM
on full pages, it's a two-tier pipeline:

    [paddleocr Python] → layout detect + region crop ─→ [VLM on each crop]

The vendor's own repo ships `paddleocr.PaddleOCRVL` as the orchestrator that
does steps 1 and 2 internally and calls the VLM (typically `llama-server` or
vLLM) for step 3. Calling the VLM directly on a full page (skipping the
orchestrator) returns token spam — confirmed empirically on RTX 3050 in
findings.md and reproduced by independent users on Hub.

This adapter supports two modes, picked by the `PADDLE_VL_MODE` env var:

  - "orchestrator" (default; production-faithful): import and run the
    paddleocr Python pipeline. Requires `paddleocr[doc-parser]` and
    `paddlepaddle-gpu` installed — Linux + Python 3.11 only. The pipeline
    points at a vLLM/llama-server endpoint (PADDLE_VLM_BASE_URL) for the
    actual VLM call on each cropped region.

  - "fullpage" (sanity check): just call the VLM endpoint with the full
    page image, OpenAI-chat-completions style. Confirms the architectural
    finding (expect garbage). Useful for the workshop "here's why the
    orchestrator matters" demo.

Both modes are documented; runbook picks one. Default in this code is
"orchestrator" so the production-correct path runs on the H200.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .base import StackAdapter
from .openai_compatible_base import OpenAICompatibleVLMAdapter, _image_data_uri
from .schema import KVPair, PageResult


class PaddleVLFullPageAdapter(OpenAICompatibleVLMAdapter):
    """Mode 'fullpage': bypass paddleocr orchestrator, send full page to VLM.

    Expected to fail (token spam) — included for the workshop demonstration
    that the orchestrator is load-bearing, not optional.
    """

    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    DEFAULT_MODEL_SLUG = "PaddlePaddle/PaddleOCR-VL"
    DEFAULT_API_KEY_ENV = "PADDLE_VL_API_KEY"
    ENV_PREFIX = "PADDLE_VL"


class PaddleVLAdapter(StackAdapter):
    """Default mode 'orchestrator' — paddleocr Python pipeline + remote VLM.

    On import the adapter does NOT load paddleocr (heavy). It loads on
    `warmup()` so a `--dry-run` of run_eval doesn't trip the import on
    machines without Paddle.
    """

    stack_id = "paddle-vl"
    model_revision = "PaddlePaddle/PaddleOCR-VL"

    def __init__(self) -> None:
        self._pipeline = None
        self._fullpage = None
        self.mode = os.environ.get("PADDLE_VL_MODE", "orchestrator")

    def warmup(self) -> None:
        if self.mode == "fullpage":
            self._fullpage = PaddleVLFullPageAdapter()
            self._fullpage.warmup()
            self.stack_id = self._fullpage.stack_id
            self.model_revision = self._fullpage.model_revision + " [fullpage-bypass]"
            return

        # orchestrator mode — import paddleocr lazily
        try:
            from paddleocr import PaddleOCRVL  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "paddleocr not installed. On the H200 pod, set up a Python 3.11 "
                "venv and run: pip install paddlepaddle-gpu paddleocr[doc-parser]. "
                "Or set PADDLE_VL_MODE=fullpage to use the sanity-check path "
                "instead. Original ImportError: " + str(e)
            )

        # The PaddleOCRVL pipeline takes a vlm_base_url for its VLM tier.
        # On the H200 runbook we point it at a co-resident llama-server / vLLM.
        vlm_base_url = os.environ.get("PADDLE_VLM_BASE_URL", "http://localhost:8000/v1")
        vlm_model = os.environ.get("PADDLE_VLM_MODEL", "PaddlePaddle/PaddleOCR-VL")
        self._pipeline = PaddleOCRVL(
            vlm_base_url=vlm_base_url,
            vlm_model=vlm_model,
        )
        self.model_revision = f"{vlm_model} [orchestrator]"

    def process_page(self, image_path: Path, page_id: str) -> PageResult:
        if self.mode == "fullpage":
            assert self._fullpage is not None
            return self._fullpage.process_page(image_path, page_id)

        if self._pipeline is None:
            self.warmup()

        result = PageResult(
            page_id=page_id,
            stack_id=self.stack_id,
            model_revision=self.model_revision,
        )
        try:
            t0 = time.perf_counter()
            # paddleocr-vl pipelines vary in API across releases. The 1.5
            # release exposes `predict` returning a list of region results
            # with cell-level structure. We dump that into raw_text +
            # parse KV markers from it; the on-disk JSON is the full record.
            output = self._pipeline.predict(str(image_path))
            result.latency_ms = (time.perf_counter() - t0) * 1000.0

            text_parts: list[str] = []
            for region in output if isinstance(output, list) else [output]:
                if hasattr(region, "markdown"):
                    text_parts.append(region.markdown)
                elif isinstance(region, dict):
                    text_parts.append(region.get("markdown") or region.get("text") or "")
                else:
                    text_parts.append(str(region))
            full_text = "\n\n".join(t for t in text_parts if t)
            result.raw_text = full_text
            if "Key-Value Pairs:" in full_text:
                kv_block = full_text.split("Key-Value Pairs:", 1)[1].strip()
                for line in kv_block.splitlines():
                    if "::" in line:
                        k, v = line.split("::", 1)
                        result.kv_pairs.append(KVPair(key=k.strip(), value=v.strip()))
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        return result
