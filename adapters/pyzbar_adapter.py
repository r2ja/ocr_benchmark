"""pyzbar specialist — barcode/QR decoder.

Not a VLM. A specialist `StackAdapter` that wraps the `pyzbar` library
(which itself wraps the open-source ZBar barcode reader). Included in the
bench-off as a demonstration of the recommended production architecture:
the OSS replica's orchestrator should bolt on a specialist barcode decoder
rather than rely on a VLM to do protocol decoding via pattern recognition.

Decodes QR codes, EAN-8/13, Code-128, ITF, UPC, and ~10 other 1D/2D formats.
Output format mirrors the `LABEL :: VALUE` convention used by the codes
axis prompt so the existing scorer accepts it without modification.
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import StackAdapter
from .schema import PageResult


class PyzbarSpecialistAdapter(StackAdapter):
    stack_id = "pyzbar"
    model_revision = "pyzbar (ZBar 0.10)"

    def warmup(self) -> None:
        # imports at warmup so a missing pyzbar surfaces a clear error
        from pyzbar.pyzbar import decode  # noqa: F401

    def process_page(self, image_path: Path, page_id: str) -> PageResult:
        from pyzbar.pyzbar import decode
        from PIL import Image

        result = PageResult(
            page_id=page_id,
            stack_id=self.stack_id,
            model_revision=self.model_revision,
        )
        try:
            img = Image.open(image_path)
            t0 = time.perf_counter()
            detections = decode(img)
            result.latency_ms = (time.perf_counter() - t0) * 1000.0

            if not detections:
                result.raw_text = "NO_CODES_FOUND"
                return result

            lines: list[str] = []
            for d in detections:
                code_type = d.type.lower()
                # Normalize a few names to match the codes-axis prompt vocabulary
                if code_type == "qrcode":
                    code_type = "qr"
                value = d.data.decode("utf-8", errors="replace")
                lines.append(f"{code_type} :: {value}")
            result.raw_text = "\n".join(lines)
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        return result
