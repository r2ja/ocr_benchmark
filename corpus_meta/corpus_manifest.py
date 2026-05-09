"""The mini-corpus manifest — 8 pages, hand-picked, each tagged with the axes
it tests.

The actual files live in `corpus/` once `scripts/build_corpus.py` has run.
This module is the source of truth for which page tests what.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CorpusPage:
    page_id: str
    source: str        # human-readable origin
    local_path: str    # relative to repo root
    axes: list[str]    # which scoring axes apply
    has_kv_gold: bool
    has_table_gold: bool
    has_text_gold: bool
    notes: str = ""


MINI_CORPUS: list[CorpusPage] = [
    CorpusPage(
        page_id="sec-10k-tech-01",
        source="SEC EDGAR — large-cap tech 10-K, page with consolidated income statement",
        local_path="corpus/sec_10k/tech_income_statement.pdf",
        axes=["layout", "tables", "text"],
        has_kv_gold=False,
        has_table_gold=True,
        has_text_gold=True,
        notes="Dense table + footnotes + multi-column header. Manual rubric for layout.",
    ),
    CorpusPage(
        page_id="sec-10k-bank-01",
        source="SEC EDGAR — JPM 10-K (FY2025), Consolidated Statements of Income",
        local_path="corpus/sec_10k/bank_income_statement.pdf",
        axes=["layout", "tables", "text"],
        has_kv_gold=False,
        has_table_gold=True,
        has_text_gold=True,
        notes="Multi-page consolidated income + comprehensive income tables.",
    ),
    CorpusPage(
        page_id="docile-invoice-01",
        source="katanaml-org/invoices-donut-data-v1 (HF, replaced DocILE)",
        local_path="corpus/docile/invoice_01.png",
        axes=["kv", "layout", "tables"],
        has_kv_gold=True,
        has_table_gold=True,
        has_text_gold=False,
        notes="Standard invoice with line items.",
    ),
    CorpusPage(
        page_id="funsd-form-01",
        source="FUNSD test form #82200067_0069",
        local_path="corpus/funsd/form_01.png",
        axes=["kv", "layout"],
        has_kv_gold=True,
        has_table_gold=False,
        has_text_gold=True,
        notes="Form-understanding KV with relations.",
    ),
    CorpusPage(
        page_id="cord-receipt-01",
        source="CORD test receipt sample",
        local_path="corpus/cord/receipt_01.png",
        axes=["kv", "text"],
        has_kv_gold=True,
        has_table_gold=False,
        has_text_gold=True,
        notes="Receipt KV — Azure prebuilt-receipt parity test.",
    ),
    CorpusPage(
        page_id="fintabnet-table-01",
        source="FinTabNet sample, complex multi-span header",
        local_path="corpus/fintabnet/table_01.png",
        axes=["tables"],
        has_kv_gold=False,
        has_table_gold=True,
        has_text_gold=False,
        notes="Table-only sanity check.",
    ),
    CorpusPage(
        page_id="iam-handwriting-01",
        source="IAM Handwriting Database line image",
        local_path="corpus/iam/line_01.png",
        axes=["text"],
        has_kv_gold=False,
        has_table_gold=False,
        has_text_gold=True,
        notes="Handwriting CER. Skip for stacks that don't claim handwriting support.",
    ),
    CorpusPage(
        page_id="omnidocbench-multilingual-01",
        source="OmniDocBench v1.6 sample, multilingual + complex layout",
        local_path="corpus/omnidocbench/page_01.png",
        axes=["layout", "text"],
        has_kv_gold=False,
        has_table_gold=False,
        has_text_gold=True,
        notes="Multilingual layout test.",
    ),
]
