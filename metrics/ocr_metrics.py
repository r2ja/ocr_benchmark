"""CER / WER for raw OCR axis. Wraps `jiwer` so the rest of the harness only
imports from here.
"""
from __future__ import annotations

import jiwer


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate. Lower is better. Returns 1.0 on empty hypothesis."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.cer(reference, hypothesis)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate. Lower is better."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.wer(reference, hypothesis)
