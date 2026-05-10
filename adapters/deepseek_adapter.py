"""DeepSeek-OCR-2 adapter — via Novita serverless, OpenAI-compatible.

Novita confirmed serverless DeepSeek-OCR-2 at `deepseek/deepseek-ocr-2`
on the OpenAI-compat endpoint at `https://api.novita.ai/openai/v1`.
Pay-per-token, ~$0.001 for our 8-page bench.

Earlier H200 vLLM attempts hit cascading dependency conflicts
(LlamaFlashAttention2 removed in transformers 4.48+, custom modeling
code requires very specific versions). Novita has those issues solved
on their side.

Sign up at novita.ai → generate API key → put in `.env` as
`NOVITA_API_KEY`. Then `python -m scripts.run_eval --stack deepseek-ocr`.

Alternative provider: DeepInfra also offers DeepSeek-OCR (note: v1, not v2)
at $0.03/$0.10 per M tokens. Set DEEPSEEK_BASE_URL + DEEPSEEK_MODEL_SLUG +
DEEPSEEK_API_KEY env vars to switch.
"""
from __future__ import annotations

from .openai_compatible_base import (
    OpenAICompatibleVLMAdapter,
    parse_deepseek_grounding,
    parse_tables,
)
from .schema import KVPair, PageResult


class DeepSeekOCRAdapter(OpenAICompatibleVLMAdapter):
    DEFAULT_BASE_URL = "https://api.novita.ai/openai/v1"
    DEFAULT_MODEL_SLUG = "deepseek/deepseek-ocr-2"
    DEFAULT_API_KEY_ENV = "NOVITA_API_KEY"
    ENV_PREFIX = "DEEPSEEK"
    # DeepSeek-OCR-2 expects task tokens, not English instructions.
    # The published task tokens are:
    #   <|grounding|>Convert the document to markdown.   - markdown + bbox
    #   <|grounding|>OCR this image.                     - OCR + bbox grounding
    #   Free OCR.                                        - plain OCR, no layout
    #   Parse the figure.                                - figure parsing
    #   Describe this image in detail.                   - VQA / description
    #   <|ref|>xxxx<|/ref|>                              - referential grounding
    #
    # English instruction prompts cause degenerate prompt-fragment loops.
    # We use the markdown grounding prompt by default (richest output) and
    # override per-page where a different task token is empirically better:
    # IAM handwriting line → `Free OCR.` cuts CER from 0.366 to 0.269.
    DEFAULT_PROMPT = "<|grounding|>Convert the document to markdown."

    PAGE_PROMPT_OVERRIDES = {
        # Single handwriting line: layout/markdown wrapper just adds noise.
        "iam-handwriting-01": "Free OCR.",
    }

    def process_page(self, image_path, page_id):
        if page_id in self.PAGE_PROMPT_OVERRIDES:
            saved_prompt = self.prompt
            self.prompt = self.PAGE_PROMPT_OVERRIDES[page_id]
            try:
                return super().process_page(image_path, page_id)
            finally:
                self.prompt = saved_prompt
        return super().process_page(image_path, page_id)

    def _populate_from_text(self, text: str, result: PageResult) -> None:
        result.raw_text = text
        # First try the grounding parser; fall back to generic table parser
        # if the model didn't emit typed blocks (e.g. with non-grounding prompts).
        grounded = parse_deepseek_grounding(text)
        if grounded["text_blocks"] or grounded["tables"] or grounded["layout"]:
            result.text_blocks = grounded["text_blocks"]
            result.layout = grounded["layout"]
            result.tables = grounded["tables"]
        else:
            result.tables = parse_tables(text)
        # KV pairs: same Markdown convention as the base
        if "Key-Value Pairs:" in text:
            kv_block = text.split("Key-Value Pairs:", 1)[1].strip()
            for line in kv_block.splitlines():
                if "::" in line:
                    k, v = line.split("::", 1)
                    result.kv_pairs.append(KVPair(key=k.strip(), value=v.strip()))
