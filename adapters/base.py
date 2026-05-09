"""Abstract base class every stack adapter implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .schema import PageResult


class StackAdapter(ABC):
    stack_id: str = ""
    model_revision: str = ""

    @abstractmethod
    def process_page(self, image_path: Path, page_id: str) -> PageResult:
        """Run the stack on a single rendered page image, return a PageResult.

        Adapters must:
          - set page_result.stack_id and model_revision
          - measure latency_ms around the model call only (not file I/O)
          - persist the raw stack-specific output and set raw_response_path
          - on failure, return a PageResult with error populated rather than raising
        """
        ...

    def warmup(self) -> None:
        """Optional: load weights, hit a dummy page to amortize cold-start."""
        return None
