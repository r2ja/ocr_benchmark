"""Build the local mini-corpus by fetching one page each from public sources.

This script makes ONLY HTTP calls — no model loading, no GPU. Safe to run
while the user's PC is loaded with other work.

Sources:
  SEC 10-K        : EDGAR full-text-search API + direct PDF/HTML fetch
  DocILE          : HuggingFace datasets (downloads small sample)
  FUNSD           : direct GitHub raw download (single image + JSON)
  CORD            : HuggingFace datasets (datasets-server slice)
  FinTabNet       : HuggingFace datasets (sample one row)
  IAM             : skipped — IAM requires registration; we'll substitute a
                    public-domain handwriting line image
  OmniDocBench    : HuggingFace datasets

Usage:
    python -m scripts.build_corpus            # builds everything
    python -m scripts.build_corpus --only sec # one source at a time

STATUS: skeleton. Each handler is a TODO that we wire up before the corpus
build pass. Splitting it this way means each fetch is independently testable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_ROOT = REPO_ROOT / "corpus"


def fetch_sec_10k() -> None:
    """Pick 2 pages from real 10-K filings. EDGAR is HTTP-only.

    TODO:
      - hit https://efts.sec.gov/LATEST/search-index?q=%22consolidated+statements+of+operations%22&forms=10-K
      - pick one tech (e.g. NVDA) and one bank (e.g. JPM) most recent 10-K
      - download the PDF or HTML, isolate one page each, save as corpus/sec_10k/*.pdf
      - hand-label table ground truth (cells with row/col/text) into a sibling .json
    """
    out = CORPUS_ROOT / "sec_10k"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[sec_10k] target dir: {out} — TODO")


def fetch_docile() -> None:
    """One invoice page from DocILE.

    Note: DocILE main split is non-commercial-research-only — we use it for our
    OWN evaluation (allowed) but do NOT redistribute the page in any deliverable.
    The workshop deck shows the SEC pages as the visual demo, not DocILE.

    TODO: load via `datasets.load_dataset("ctu-aic/docile", split="test")`,
    sample 1 invoice.
    """
    out = CORPUS_ROOT / "docile"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[docile] target dir: {out} — TODO")


def fetch_funsd() -> None:
    """One form image + KV ground truth from FUNSD.

    TODO: GitHub raw download from https://guillaumejaume.github.io/FUNSD/
    """
    out = CORPUS_ROOT / "funsd"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[funsd] target dir: {out} — TODO")


def fetch_cord() -> None:
    """One receipt + KV ground truth from CORD.

    TODO: `datasets.load_dataset("naver-clova-ix/cord-v2", split="test")`,
    sample 1 receipt.
    """
    out = CORPUS_ROOT / "cord"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cord] target dir: {out} — TODO")


def fetch_fintabnet() -> None:
    """One table image from FinTabNet.

    TODO: HF dataset, sample 1 table with complex header.
    """
    out = CORPUS_ROOT / "fintabnet"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[fintabnet] target dir: {out} — TODO")


def fetch_handwriting() -> None:
    """One handwriting line image. IAM requires registration; we substitute a
    public-domain alternative (e.g. a line from the open Bentham dataset, or a
    Wikimedia-Commons handwriting page) so the workshop is fully redistributable.
    """
    out = CORPUS_ROOT / "iam"  # keep folder name for the manifest
    out.mkdir(parents=True, exist_ok=True)
    print(f"[handwriting] target dir: {out} — TODO")


def fetch_omnidocbench() -> None:
    """One page from OmniDocBench v1.6 with mixed multilingual layout.

    TODO: HF dataset opendatalab/OmniDocBench-v1_6, sample 1 page.
    """
    out = CORPUS_ROOT / "omnidocbench"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[omnidocbench] target dir: {out} — TODO")


HANDLERS = {
    "sec": fetch_sec_10k,
    "docile": fetch_docile,
    "funsd": fetch_funsd,
    "cord": fetch_cord,
    "fintabnet": fetch_fintabnet,
    "handwriting": fetch_handwriting,
    "omnidocbench": fetch_omnidocbench,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(HANDLERS), help="Run only one source")
    args = parser.parse_args()

    sources = [args.only] if args.only else list(HANDLERS)
    for src in sources:
        HANDLERS[src]()


if __name__ == "__main__":
    main()
