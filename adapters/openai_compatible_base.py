"""Shared base for VLM adapters that speak OpenAI-compatible chat completions.

This is the wire format for OpenRouter, vLLM ServingRuntime (RHOAI / RunPod),
llama-server, Together, and most other inference hosts. By keeping every VLM
adapter on this base we get one path to test/debug, and the same code that
runs against OpenRouter today runs against an H200 vLLM pod tomorrow with
only `base_url` and `model_slug` changing.

Subclasses set defaults; environment variables override at runtime so the
H200 runbook can point all adapters at `localhost:8000` without code changes:

    <STACK>_BASE_URL    e.g. DOTS_BASE_URL=http://localhost:8000/v1
    <STACK>_MODEL_SLUG  e.g. DOTS_MODEL_SLUG=rednote-hilab/dots.ocr
    <STACK>_API_KEY     e.g. DOTS_API_KEY=EMPTY  (vLLM ignores it)
"""
from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from pathlib import Path

from .base import StackAdapter
from .schema import BBox, KVPair, LayoutBlock, PageResult, Table, TableCell, TextBlock


def parse_markdown_tables(text: str) -> list[Table]:
    """Walk Markdown text and extract every table as a Table object.

    A Markdown table is a contiguous run of lines that:
      - Each starts and ends with `|`
      - The second line is a separator like `|----|----|` (only `|`, `-`, `:`, spaces)

    Multiple tables in one document are returned in document order. Cell text
    is unescaped for `\\|` and stripped. Header row sets is_header=True.
    """
    tables: list[Table] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        is_pipe_row = line.startswith("|") and line.endswith("|") and line.count("|") >= 2
        is_separator = (
            next_line.startswith("|")
            and next_line.endswith("|")
            and all(c in "|-: " for c in next_line)
            and "-" in next_line
        )
        if is_pipe_row and is_separator:
            header_cells = _split_md_row(line)
            data_rows: list[list[str]] = []
            j = i + 2
            while j < len(lines):
                row = lines[j].strip()
                if not (row.startswith("|") and row.endswith("|") and row.count("|") >= 2):
                    break
                data_rows.append(_split_md_row(row))
                j += 1

            n_cols = max([len(header_cells)] + [len(r) for r in data_rows]) if data_rows else len(header_cells)
            cells: list[TableCell] = []
            for col_idx, h in enumerate(header_cells[:n_cols]):
                cells.append(TableCell(row=0, col=col_idx, text=h, is_header=True))
            for row_idx, row in enumerate(data_rows, start=1):
                for col_idx, c in enumerate(row[:n_cols]):
                    cells.append(TableCell(row=row_idx, col=col_idx, text=c, is_header=False))

            tables.append(
                Table(
                    cells=cells,
                    bbox=None,
                    html=None,
                    n_rows=1 + len(data_rows),
                    n_cols=n_cols,
                )
            )
            i = j
            continue
        i += 1
    return tables


def _split_md_row(row: str) -> list[str]:
    """Split a Markdown table row on `|`, drop the leading/trailing empties,
    unescape `\\|`, and strip each cell."""
    inner = row.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    # unescape \| to a placeholder, split, restore
    cells = inner.replace("\\|", "\x00").split("|")
    return [c.strip().replace("\x00", "|") for c in cells]


def parse_html_tables(text: str) -> list[Table]:
    """Extract every `<table>...</table>` block as a Table object.

    Some OCR VLMs (notably Baidu Qianfan-OCR-Fast) emit HTML tables instead
    of Markdown. This regex-based parser covers the common shape
    (no nested tables, simple `<tr><td>`/`<th>` rows) without pulling in
    a full HTML parser dependency.

    Falls back to a no-`<tr>` mode (a single synthetic row of all `<td>`s)
    so DeepSeek-OCR-2's row-less HTML still produces a Table object.
    """
    import re

    out: list[Table] = []
    blocks = re.findall(r"<table[^>]*>(.*?)</table>", text, re.DOTALL | re.IGNORECASE)
    for block in blocks:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", block, re.DOTALL | re.IGNORECASE)
        if rows:
            cells: list[TableCell] = []
            n_cols = 0
            for row_idx, row_html in enumerate(rows):
                row_cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.DOTALL | re.IGNORECASE)
                for col_idx, raw_cell in enumerate(row_cells):
                    clean = re.sub(r"<[^>]+>", "", raw_cell)
                    clean = re.sub(r"\s+", " ", clean).strip()
                    cells.append(
                        TableCell(row=row_idx, col=col_idx, text=clean, is_header=(row_idx == 0))
                    )
                n_cols = max(n_cols, len(row_cells))
            if cells:
                out.append(Table(cells=cells, bbox=None, html=None, n_rows=len(rows), n_cols=n_cols))
            continue

        # No <tr>: treat content as a single synthetic row, with each <td>/<th> a cell.
        tds = re.findall(r"<t[hd][^>]*>(.*?)(?=<t[hd]|$)", block, re.DOTALL | re.IGNORECASE)
        if tds:
            cells = []
            for col_idx, raw in enumerate(tds):
                clean = re.sub(r"<[^>]+>", "", raw)
                clean = re.sub(r"\s+", " ", clean).strip()
                cells.append(TableCell(row=0, col=col_idx, text=clean, is_header=False))
            out.append(Table(cells=cells, bbox=None, html=None, n_rows=1, n_cols=len(cells)))
        else:
            clean = re.sub(r"<[^>]+>", "", block)
            clean = re.sub(r"\s+", " ", clean).strip()
            if clean:
                out.append(
                    Table(
                        cells=[TableCell(row=0, col=0, text=clean[:500])],
                        bbox=None, html=None, n_rows=1, n_cols=1,
                    )
                )
    return out


# DeepSeek-OCR-2 emits typed blocks with bbox grounding:
#   <type>[[x1,y1,x2,y2]]\n<content>\n\n
# Types observed: text, sub_title, title, table, image, list. Coords are
# 0-1000 normalized (not pixel space). Multi-line content can have multi-bbox
# `[[x1,y1,x2,y2], [x3,y3,x4,y4]]` lists; we take the first 4 numbers.
import re as _re

_DS_BLOCK = _re.compile(
    r"^(?P<type>text|sub_title|title|table|image|list)\s*\[\[(?P<bbox>[\d,\s]+)\]\]\s*\n(?P<content>.*?)$",
    _re.DOTALL | _re.MULTILINE,
)


def parse_deepseek_grounding(text: str) -> dict:
    """Walk DeepSeek-OCR-2's typed-block grounding format.

    Returns a dict with text_blocks / layout / tables ready to merge into
    a `PageResult`. Empty dict if the input doesn't look like DeepSeek
    grounding (no typed-bbox blocks found).
    """
    out: dict[str, list] = {"text_blocks": [], "layout": [], "tables": []}
    chunks = _re.split(r"\n\n+", text)
    found_any = False
    for chunk in chunks:
        m = _re.match(
            r"^(text|sub_title|title|table|image|list)\s*\[\[([\d,\s]+)\]\]\s*\n(.*)$",
            chunk.strip(),
            _re.DOTALL,
        )
        if not m:
            continue
        found_any = True
        block_type, bbox_raw, content = m.group(1), m.group(2), m.group(3).strip()
        nums = [int(n) for n in _re.findall(r"\d+", bbox_raw)]
        bbox = BBox(x0=nums[0], y0=nums[1], x1=nums[2], y1=nums[3]) if len(nums) >= 4 else None

        if block_type in ("text", "list"):
            out["text_blocks"].append(TextBlock(text=content, bbox=bbox))
            if bbox is not None:
                out["layout"].append(LayoutBlock(label=block_type, bbox=bbox))
        elif block_type in ("title", "sub_title"):
            out["text_blocks"].append(TextBlock(text=content, bbox=bbox))
            if bbox is not None:
                out["layout"].append(LayoutBlock(label=block_type, bbox=bbox))
        elif block_type == "image":
            if bbox is not None:
                out["layout"].append(LayoutBlock(label="figure", bbox=bbox))
        elif block_type == "table":
            tables = parse_html_tables(content)
            if not tables:
                tables = parse_markdown_tables(content)
            for t in tables:
                if bbox is not None:
                    t.bbox = bbox
            out["tables"].extend(tables)
            if bbox is not None:
                out["layout"].append(LayoutBlock(label="table", bbox=bbox))

    return out if found_any else {"text_blocks": [], "layout": [], "tables": []}


def parse_tables(text: str) -> list[Table]:
    """Try both Markdown and HTML table syntaxes — different VLMs prefer different output formats."""
    return parse_markdown_tables(text) + parse_html_tables(text)


GENERIC_PROMPT = (
    "Extract the full content of this document image as Markdown. Preserve "
    "tables (use Markdown table syntax), headers, lists, and reading order. "
    "If the document contains form fields or key-value pairs, list them at "
    "the end under a 'Key-Value Pairs:' section, one per line as 'KEY :: VALUE'."
)


def _image_data_uri(image_path: Path) -> str:
    """Return a data:image/png;base64,... URI for the page.

    Rasterizes the first page of a PDF if needed.
    """
    suffix = image_path.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(image_path))
        try:
            page = pdf[0]
            pil = page.render(scale=200 / 72.0).to_pil()
            buf = BytesIO()
            pil.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        finally:
            pdf.close()
    if suffix == ".png":
        return f"data:image/png;base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
    return f"data:image/jpeg;base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


class OpenAICompatibleVLMAdapter(StackAdapter):
    """Subclasses set the four DEFAULT_* class attrs."""

    DEFAULT_BASE_URL: str = ""
    DEFAULT_MODEL_SLUG: str = ""
    DEFAULT_API_KEY_ENV: str = ""
    DEFAULT_PROMPT: str = GENERIC_PROMPT
    ENV_PREFIX: str = ""  # e.g. "DOTS" → reads DOTS_BASE_URL, DOTS_MODEL_SLUG

    HTTP_REFERER: str = "https://logarithmtech.example"
    APP_TITLE: str = "Logarithm DocIntel Benchmark"
    MAX_TOKENS: int = 4096
    REQUEST_TIMEOUT: int = 240

    def __init__(
        self,
        model_slug: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.model_slug = (
            model_slug
            or os.environ.get(f"{self.ENV_PREFIX}_MODEL_SLUG")
            or self.DEFAULT_MODEL_SLUG
        )
        self.base_url = (
            base_url
            or os.environ.get(f"{self.ENV_PREFIX}_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        self.api_key = (
            api_key
            or os.environ.get(f"{self.ENV_PREFIX}_API_KEY")
            or os.environ.get(self.DEFAULT_API_KEY_ENV, "")
        )
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.stack_id = self._derive_stack_id(self.model_slug)
        self.model_revision = self.model_slug

    @staticmethod
    def _derive_stack_id(slug: str) -> str:
        last = slug.split("/")[-1]
        if last.endswith("-instruct"):
            last = last[: -len("-instruct")]
        return last

    def warmup(self) -> None:
        if not self.base_url:
            raise RuntimeError(
                f"{self.__class__.__name__}: base_url not set. "
                f"Set {self.ENV_PREFIX}_BASE_URL or pass base_url=..."
            )

    def _build_messages(self, data_uri: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

    def _build_headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        # OpenRouter-only attribution headers; harmless on vLLM/llama-server.
        h["HTTP-Referer"] = self.HTTP_REFERER
        h["X-Title"] = self.APP_TITLE
        return h

    def process_page(self, image_path: Path, page_id: str) -> PageResult:
        import requests

        result = PageResult(
            page_id=page_id,
            stack_id=self.stack_id,
            model_revision=self.model_revision,
        )
        try:
            data_uri = _image_data_uri(image_path)
            payload = {
                "model": self.model_slug,
                "messages": self._build_messages(data_uri),
                "max_tokens": self.MAX_TOKENS,
                "temperature": 0.0,
            }
            url = self.base_url.rstrip("/") + "/chat/completions"
            t0 = time.perf_counter()
            r = requests.post(
                url,
                json=payload,
                headers=self._build_headers(),
                timeout=self.REQUEST_TIMEOUT,
            )
            result.latency_ms = (time.perf_counter() - t0) * 1000.0
            if r.status_code != 200:
                result.error = f"HTTP {r.status_code}: {r.text[:300]}"
                return result
            body = r.json()
            text = body["choices"][0]["message"]["content"]
            self._populate_from_text(text, result)
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        return result

    def _populate_from_text(self, text: str, result: PageResult) -> None:
        result.raw_text = text
        if "Key-Value Pairs:" in text:
            kv_block = text.split("Key-Value Pairs:", 1)[1].strip()
            for line in kv_block.splitlines():
                if "::" in line:
                    k, v = line.split("::", 1)
                    result.kv_pairs.append(KVPair(key=k.strip(), value=v.strip()))
        result.tables = parse_tables(text)
