"""Shared base for VLM adapters that run locally via llama-cpp-python + GGUF.

llama.cpp's multimodal path uses a separate "mmproj" projector file plus the
main model weights. The HF GGUF mirrors typically ship both (e.g.
`*-Q4_K_M.gguf` + `mmproj-*.gguf`) — subclasses set `repo`, `model_file`, and
`mmproj_file` and the base handles download + load + inference.

Memory budget on a 4 GB 3050 (Windows compositor uses ~400 MB) means we set
`n_gpu_layers=-1` (all layers on GPU) for sub-2 GB models, and an explicit
layer split for larger ones. Subclasses can override `n_gpu_layers`.
"""
from __future__ import annotations

import base64
import os
import sys
import time
from io import BytesIO
from pathlib import Path

from .base import StackAdapter
from .schema import KVPair, PageResult


def _ensure_cuda_dlls_on_path() -> None:
    """llama-cpp-python's prebuilt CUDA wheel needs cudart64_12.dll +
    cublas64_12.dll on the DLL search path. PyTorch already ships these in
    torch/lib but Windows doesn't auto-include other site-packages. Add it
    explicitly before any llama_cpp import.
    """
    if sys.platform != "win32":
        return
    try:
        import torch  # noqa: F401  (ensures torch is importable)
        torch_lib = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
        if not torch_lib.exists():
            # fall back to inferring from sys.path
            import torch as _t
            torch_lib = Path(_t.__file__).parent / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))
    except Exception:
        # best-effort; if it fails llama_cpp import will surface a clear error
        pass


_ensure_cuda_dlls_on_path()

DEFAULT_PROMPT = (
    "Extract the full content of this document image as Markdown. Preserve "
    "tables (use Markdown table syntax), headers, lists, and reading order. "
    "If the document contains form fields or key-value pairs, list them at "
    "the end under a 'Key-Value Pairs:' section, one per line as 'KEY :: VALUE'."
)


def _ensure_pil_image(image_path: Path):
    """Load image (PNG/JPG) or rasterize first page of PDF."""
    suffix = image_path.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(image_path))
        try:
            page = pdf[0]
            return page.render(scale=200 / 72.0).to_pil()
        finally:
            pdf.close()
    from PIL import Image

    return Image.open(image_path).convert("RGB")


def _pil_to_data_uri(img) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    b = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b}"


class LlamaCppVLMAdapter(StackAdapter):
    """Subclasses set the four class attrs and (optionally) override prompt."""
    repo: str = ""             # HF repo, e.g. "PaddlePaddle/PaddleOCR-VL-1.5-GGUF"
    model_file: str = ""       # filename inside the repo, e.g. "PaddleOCR-VL-1.5-Q4_K_M.gguf"
    mmproj_file: str = ""      # filename for the multimodal projector
    n_gpu_layers: int = -1     # -1 = all on GPU; smaller = partial offload to CPU
    n_ctx: int = 8192
    prompt: str = DEFAULT_PROMPT

    def __init__(self) -> None:
        self._llm = None
        self._chat_handler = None
        # subclasses set stack_id and model_revision

    def _download_assets(self) -> tuple[Path, Path]:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=self.repo, filename=self.model_file)
        mmproj_path = hf_hub_download(repo_id=self.repo, filename=self.mmproj_file)
        return Path(model_path), Path(mmproj_path)

    chat_handler_name: str = "Llava15ChatHandler"  # subclasses pick the right one

    def _resolve_chat_handler_class(self):
        """Look up the chat-handler class by name in llama_cpp.llama_chat_format.

        Done at warmup time (not import time) so the DLL-search-path shim has
        already run before we touch llama_cpp internals.
        """
        from llama_cpp import llama_chat_format

        cls = getattr(llama_chat_format, self.chat_handler_name, None)
        if cls is None:
            raise RuntimeError(
                f"Chat handler '{self.chat_handler_name}' not found in this "
                f"llama-cpp-python version. Available: "
                f"{[n for n in dir(llama_chat_format) if 'Handler' in n]}"
            )
        return cls

    def warmup(self) -> None:
        from llama_cpp import Llama

        model_path, mmproj_path = self._download_assets()
        handler_cls = self._resolve_chat_handler_class()
        self._chat_handler = handler_cls(clip_model_path=str(mmproj_path))
        self._llm = Llama(
            model_path=str(model_path),
            chat_handler=self._chat_handler,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            logits_all=False,
            verbose=False,
        )

    def process_page(self, image_path: Path, page_id: str) -> PageResult:
        if self._llm is None:
            self.warmup()

        result = PageResult(
            page_id=page_id,
            stack_id=self.stack_id,
            model_revision=self.model_revision,
        )
        try:
            img = _ensure_pil_image(image_path)
            data_uri = _pil_to_data_uri(img)

            t0 = time.perf_counter()
            out = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            result.latency_ms = (time.perf_counter() - t0) * 1000.0
            text = out["choices"][0]["message"]["content"]
            result.raw_text = text
            if "Key-Value Pairs:" in text:
                kv_block = text.split("Key-Value Pairs:", 1)[1].strip()
                for line in kv_block.splitlines():
                    if "::" in line:
                        k, v = line.split("::", 1)
                        result.kv_pairs.append(KVPair(key=k.strip(), value=v.strip()))
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        return result
