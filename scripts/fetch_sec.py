"""Fetch SEC EDGAR 10-K filings and render the page that contains a target
table (e.g. consolidated income statement, RWA schedule) to PDF.

Pipeline:
  1. EDGAR submissions API -> most recent 10-K accession + primary document
  2. Download the primary .htm file (the iXBRL filing)
  3. Use BeautifulSoup to find the section heading we want
  4. Snip a self-contained HTML window around that section
  5. Render the snippet to PDF via Playwright (Chromium headless)

The output PDF is a single page that mirrors what an analyst would see when
printing that section — a real, redistributable, dense financial document.

We target two filings to populate `corpus/sec_10k/`:
  tech : NVDA 10-K    -> Consolidated Statements of Income
  bank : JPM 10-K     -> Risk-weighted assets (or Income Statement if RWA hard
                         to isolate)
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEC_DIR = REPO_ROOT / "corpus" / "sec_10k"
SEC_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "DocIntel Benchmark Research contact@example.com"}


@dataclass
class FilingTarget:
    label: str
    cik: str               # zero-padded to 10
    section_keywords: list[str]  # case-insensitive substrings to match an h-tag near the target table
    out_pdf: str           # filename inside SEC_DIR


# CIKs: NVDA 0001045810, JPM 0000019617
TARGETS = [
    FilingTarget(
        label="tech_income_statement",
        cik="0001045810",
        section_keywords=["consolidated statements of income"],
        out_pdf="tech_income_statement.pdf",
    ),
    FilingTarget(
        label="bank_rwa_table",
        cik="0000019617",
        section_keywords=[
            "consolidated statements of income",  # fallback; RWA is harder to isolate consistently
        ],
        out_pdf="bank_income_statement.pdf",
    ),
]


def _http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _latest_10k(cik: str) -> tuple[str, str]:
    """Return (accession_no_with_hyphens, primary_document) for the most
    recent 10-K filing for the given CIK."""
    data = json.loads(_http_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return recent["accessionNumber"][i], recent["primaryDocument"][i]
    raise RuntimeError(f"no 10-K found for CIK {cik}")


def _filing_url(cik: str, accession: str, primary_doc: str) -> str:
    cik_no_zeros = str(int(cik))
    accession_no_hyphens = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_zeros}/{accession_no_hyphens}/{primary_doc}"
    )


def _build_snippet(html: str, keywords: list[str]) -> str:
    """Extract a self-contained HTML window around the first heading matching a
    keyword. We grab the heading and the next ~1 sibling block (typically the
    table) so the rendered PDF is a focused 1-page view, not the whole 200-page
    filing.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    heading = None
    for tag in soup.find_all(re.compile(r"^h[1-6]$|^p$|^div$|^span$")):
        text = (tag.get_text(" ", strip=True) or "").lower()
        if any(k in text for k in keywords) and len(text) < 200:
            heading = tag
            break

    if heading is None:
        # fallback: find first table that contains a row mentioning "Net revenue" or "Total revenue"
        for tbl in soup.find_all("table"):
            t = tbl.get_text(" ", strip=True).lower()
            if any(k in t for k in ["net revenue", "total revenues", "interest income"]):
                heading = tbl
                break

    if heading is None:
        raise RuntimeError(f"no section/table matched keywords {keywords}")

    # collect the heading + the next table sibling (or the heading itself if it IS a table)
    fragments: list[str] = [str(heading)]
    nxt = heading
    for _ in range(8):
        nxt = nxt.find_next_sibling()
        if nxt is None:
            break
        fragments.append(str(nxt))
        if nxt.name == "table":
            break

    inline_styles = """
      <style>
        @page { size: Letter; margin: 0.5in; }
        body { font-family: Arial, sans-serif; font-size: 10pt; }
        table { border-collapse: collapse; width: 100%; }
        td, th { border: 1px solid #ccc; padding: 4px 6px; vertical-align: top; }
        h1, h2, h3, h4 { margin: 0.4em 0; }
      </style>
    """
    return f"<html><head><meta charset='utf-8'>{inline_styles}</head><body>{''.join(fragments)}</body></html>"


def _render_html_to_pdf(html: str, out_pdf: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="load")
            page.pdf(
                path=str(out_pdf),
                format="Letter",
                margin={"top": "0.5in", "bottom": "0.5in", "left": "0.5in", "right": "0.5in"},
                print_background=True,
            )
        finally:
            browser.close()


def fetch_one(t: FilingTarget) -> Path:
    print(f"[{t.label}] looking up most recent 10-K for CIK {t.cik}")
    accession, primary = _latest_10k(t.cik)
    print(f"[{t.label}]   accession={accession}  primary={primary}")

    url = _filing_url(t.cik, accession, primary)
    print(f"[{t.label}]   downloading {url}")
    html = _http_get(url, timeout=120).decode("utf-8", errors="replace")

    snippet = _build_snippet(html, t.section_keywords)
    snippet_path = SEC_DIR / f"{t.label}.snippet.html"
    snippet_path.write_text(snippet, encoding="utf-8")
    print(f"[{t.label}]   snippet saved -> {snippet_path}")

    out_pdf = SEC_DIR / t.out_pdf
    print(f"[{t.label}]   rendering -> {out_pdf}")
    _render_html_to_pdf(snippet, out_pdf)
    print(f"[{t.label}]   done ({out_pdf.stat().st_size} bytes)")

    # write a sibling metadata json
    meta = {
        "cik": t.cik,
        "accession": accession,
        "primary_document": primary,
        "source_url": url,
        "section_keywords": t.section_keywords,
        "snippet_html_path": str(snippet_path.relative_to(REPO_ROOT)),
    }
    (SEC_DIR / f"{t.label}.meta.json").write_text(json.dumps(meta, indent=2))
    return out_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[t.label for t in TARGETS], help="Run one target")
    args = parser.parse_args()

    targets = [t for t in TARGETS if not args.only or t.label == args.only]
    for t in targets:
        try:
            fetch_one(t)
        except Exception as e:
            print(f"[{t.label}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
