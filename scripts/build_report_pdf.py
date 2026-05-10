"""Render docs/workshop_report.md to a paper-style PDF via Playwright/Chromium.

Pipeline:
  Markdown → HTML (python-markdown with tables + fenced-code extensions)
            → wrapped in a CSS-styled HTML document
            → rendered to PDF via headless Chromium (Playwright, already
              available in the venv from scripts/fetch_sec.py)

Output: docs/workshop_report.pdf
"""
from __future__ import annotations

from pathlib import Path

import markdown

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_MD = REPO_ROOT / "docs" / "workshop_report.md"
OUT_PDF = REPO_ROOT / "docs" / "workshop_report.pdf"

CSS = """
@page {
  size: Letter;
  margin: 0.85in 0.75in 0.85in 0.75in;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 9pt;
    color: #888;
  }
}

* { box-sizing: border-box; }

body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #111;
  max-width: 100%;
}

h1 {
  font-size: 22pt;
  margin-top: 0;
  margin-bottom: 0.3em;
  border-bottom: 1.5px solid #222;
  padding-bottom: 0.2em;
  page-break-after: avoid;
}

h2 {
  font-size: 14pt;
  margin-top: 1.4em;
  margin-bottom: 0.4em;
  page-break-after: avoid;
}

h3 {
  font-size: 11.5pt;
  margin-top: 1.2em;
  margin-bottom: 0.3em;
  page-break-after: avoid;
}

p { margin: 0.5em 0; orphans: 3; widows: 3; }

ul, ol { margin: 0.4em 0 0.6em 1.2em; }
li { margin: 0.15em 0; }

strong { color: #000; }
em { color: #333; }

code {
  font-family: "SF Mono", "Consolas", "Menlo", monospace;
  font-size: 9.5pt;
  background: #f4f4f5;
  padding: 1px 4px;
  border-radius: 2px;
}

pre {
  font-family: "SF Mono", "Consolas", "Menlo", monospace;
  font-size: 9pt;
  background: #f7f7f8;
  border: 1px solid #e5e5e8;
  border-radius: 3px;
  padding: 8px 10px;
  overflow-x: auto;
  page-break-inside: avoid;
}

pre code {
  background: transparent;
  padding: 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.7em 0;
  font-size: 9.5pt;
  page-break-inside: avoid;
}

th, td {
  border: 1px solid #ccd;
  padding: 4px 7px;
  text-align: left;
  vertical-align: top;
}

th {
  background: #f1f3f6;
  font-weight: 600;
}

tr:nth-child(even) td { background: #fafbfc; }

hr {
  border: 0;
  border-top: 1px solid #ccd;
  margin: 1.6em 0;
}

blockquote {
  margin: 0.6em 0;
  padding: 0.4em 0.8em;
  border-left: 3px solid #999;
  color: #444;
  background: #fafafa;
}

/* Force page breaks before each top-level section */
h2 { page-break-before: auto; }
h2#appendix-a-failed-integration-attempts-roadblocks,
h2#appendix-b-per-page-per-stack-detail,
h2#appendix-c-repository-structure { page-break-before: always; }
"""


def main() -> None:
    md_text = SRC_MD.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )

    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Open-Source Replacements for Azure Document Intelligence</title>
<style>{CSS}</style>
</head><body>{html_body}</body></html>
"""

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="load")
            page.pdf(
                path=str(OUT_PDF),
                format="Letter",
                print_background=True,
                margin={"top": "0.85in", "bottom": "0.85in", "left": "0.75in", "right": "0.75in"},
                display_header_footer=False,
            )
        finally:
            browser.close()

    print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
