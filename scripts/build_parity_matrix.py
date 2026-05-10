"""Generate the Azure DI feature-parity matrix as HTML.

Rows = Azure Document Intelligence's full feature surface.
Columns = each stack (Docling + 4 Qwen sizes + Qianfan + DeepSeek + pyzbar).
Cells = quantitative score where we measured directly, qualitative ✓/✗/— otherwise.
Best per row highlighted with a light-green background.

The output is written to docs/parity_matrix.html and ALSO injected into
docs/workshop_report.md between the markers
    <!--PARITY_MATRIX_BEGIN-->
    <!--PARITY_MATRIX_END-->
so the next run of build_report_pdf.py picks it up.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
DOCS = REPO_ROOT / "docs"

STACKS = [
    ("docling",        "Docling"),
    ("qwen-8b",        "Qwen-8B"),
    ("qwen-30b-a3b",   "Qwen-30B-A3B"),
    ("qwen-32b",       "Qwen-32B"),
    ("qwen-235b-a22b", "Qwen-235B"),
    ("qianfan-ocr",    "Qianfan"),
    ("deepseek-ocr",   "DeepSeek"),
    ("pyzbar",         "pyzbar"),
]


# ---------------------------------------------------------------------------
# Pull the score data
# ---------------------------------------------------------------------------
def _load_scores() -> dict:
    general = pd.read_parquet(RESULTS / "scores_local.parquet")
    feature = pd.read_parquet(RESULTS / "feature_scores.parquet")

    out: dict[str, dict] = {s: {} for s, _ in STACKS}

    # General-corpus aggregates
    g_grouped = general.groupby("stack").agg(
        cer_iam=("cer", "mean"),
        kv_f1=("kv_f1", "mean"),
        n_tables=("n_tables", "sum"),
        table_content=("table_content_vs_docling", "mean"),
        latency_med=("latency_ms", "median"),
    )
    for stack in g_grouped.index:
        if stack in out:
            out[stack].update({
                "cer_iam": g_grouped.loc[stack, "cer_iam"],
                "kv_f1": g_grouped.loc[stack, "kv_f1"],
                "n_tables": g_grouped.loc[stack, "n_tables"],
                "table_content": g_grouped.loc[stack, "table_content"],
                "latency_med": g_grouped.loc[stack, "latency_med"],
            })

    # Feature-axis headlines
    headline = {
        "checkboxes":      "detection_f1",
        "signatures":      "detection_f1",
        "formulas":        "mean_similarity",
        "codes":           "exact_match_f1",
        "receipt_schema":  "schema_score",
        "invoice_schema":  "schema_score",
    }
    for axis, col in headline.items():
        sub = feature[feature["axis"] == axis][["stack", col]]
        for _, row in sub.iterrows():
            stack = row["stack"]
            if stack in out:
                out[stack][axis] = row[col]
    return out


# ---------------------------------------------------------------------------
# Define the Azure DI feature rows
# ---------------------------------------------------------------------------
# `kind` field: "score" (numeric, higher better; honors invert flag),
#               "qual"  (qualitative ✓/✗/— from a hand-set dict),
# `metric_label`: short name shown in the cell header
# `score_key`:    key into the scores dict
# `invert`:       for CER (lower is better), invert via 1 - cer (clamped to 0)
ROWS = [
    # ----- READ API -----
    {"section": "Read API", "feature": "Raw OCR text accuracy",
     "kind": "qual",
     "values": {"docling": "✓", "qwen-8b": "✓", "qwen-30b-a3b": "✓", "qwen-32b": "✓",
                "qwen-235b-a22b": "✓", "qianfan-ocr": "✓", "deepseek-ocr": "✓",
                "pyzbar": "—"},
     "note": "All VLM stacks transcribe text. pyzbar is barcode-only."},
    {"section": "Read API", "feature": "Handwriting recognition (1−CER on IAM)",
     "kind": "score", "score_key": "cer_iam", "invert": True},
    {"section": "Read API", "feature": "Multilingual coverage",
     "kind": "qual",
     "values": {"docling": "80+", "qwen-8b": "32", "qwen-30b-a3b": "32",
                "qwen-32b": "32", "qwen-235b-a22b": "32",
                "qianfan-ocr": "192", "deepseek-ocr": "EN+ZH",
                "pyzbar": "—"},
     "note": "Per vendor docs. Qianfan-OCR is the multilingual leader (192 langs)."},

    # ----- LAYOUT API -----
    {"section": "Layout API", "feature": "Tables — structural extraction (count)",
     "kind": "score", "score_key": "n_tables_norm"},
    {"section": "Layout API", "feature": "Tables — content fidelity (vs Docling)",
     "kind": "score", "score_key": "table_content"},
    {"section": "Layout API", "feature": "Selection marks / checkboxes (F1)",
     "kind": "score", "score_key": "checkboxes"},
    {"section": "Layout API", "feature": "Reading order",
     "kind": "qual",
     "values": {"docling": "✓", "qwen-8b": "✓", "qwen-30b-a3b": "✓", "qwen-32b": "✓",
                "qwen-235b-a22b": "✓", "qianfan-ocr": "✓", "deepseek-ocr": "✓ (bbox)",
                "pyzbar": "—"},
     "note": "Implicit in Markdown sequencing or bbox grounding. Not scored against gold."},
    {"section": "Layout API", "feature": "Figures / pictures detection",
     "kind": "qual",
     "values": {"docling": "✓ (DocLayNet)", "qwen-8b": "~", "qwen-30b-a3b": "~",
                "qwen-32b": "~", "qwen-235b-a22b": "~",
                "qianfan-ocr": "~", "deepseek-ocr": "✓ (image bbox)",
                "pyzbar": "—"},
     "note": "Docling has native Picture class. DeepSeek emits image[[bbox]]. Qwen-family detect via prompt."},
    {"section": "Layout API", "feature": "Formulas / equations (mean LaTeX similarity)",
     "kind": "score", "score_key": "formulas"},

    # ----- GENERAL DOCUMENT API -----
    {"section": "General Document", "feature": "KV extraction (F1, FUNSD+CORD)",
     "kind": "score", "score_key": "kv_f1"},
    {"section": "General Document", "feature": "Per-field confidence (calibrated)",
     "kind": "qual",
     "values": {"docling": "element-level", "qwen-8b": "✗", "qwen-30b-a3b": "✗",
                "qwen-32b": "✗", "qwen-235b-a22b": "✗",
                "qianfan-ocr": "✗", "deepseek-ocr": "✗",
                "pyzbar": "✗"},
     "note": "Generative VLMs only emit token-likelihoods. Real gap requiring in-house calibration classifier."},

    # ----- PREBUILT VERTICALS -----
    {"section": "Prebuilt Verticals", "feature": "Receipt schema extraction",
     "kind": "score", "score_key": "receipt_schema"},
    {"section": "Prebuilt Verticals", "feature": "Invoice schema extraction",
     "kind": "score", "score_key": "invoice_schema"},
    {"section": "Prebuilt Verticals", "feature": "Other verticals (W-2, 1099, ID, etc.)",
     "kind": "qual",
     "values": {"docling": "✗", "qwen-8b": "via prompt", "qwen-30b-a3b": "via prompt",
                "qwen-32b": "via prompt", "qwen-235b-a22b": "via prompt",
                "qianfan-ocr": "via prompt", "deepseek-ocr": "✗",
                "pyzbar": "—"},
     "note": "Inferred from receipt/invoice results: VLMs handle schema-locked extraction generically."},

    # ----- ADD-ONS -----
    {"section": "Add-ons", "feature": "Signature / seal detection (F1)",
     "kind": "score", "score_key": "signatures"},
    {"section": "Add-ons", "feature": "Barcodes / QR (exact-match F1)",
     "kind": "score", "score_key": "codes"},
    {"section": "Add-ons", "feature": "Office formats (DOCX/XLSX/PPTX)",
     "kind": "qual",
     "values": {"docling": "✓ native", "qwen-8b": "✗", "qwen-30b-a3b": "✗",
                "qwen-32b": "✗", "qwen-235b-a22b": "✗",
                "qianfan-ocr": "✗", "deepseek-ocr": "✗",
                "pyzbar": "—"},
     "note": "Only Docling supports natively."},
    {"section": "Add-ons", "feature": "Searchable PDF output",
     "kind": "qual",
     "values": {"docling": "✓", "qwen-8b": "✗", "qwen-30b-a3b": "✗",
                "qwen-32b": "✗", "qwen-235b-a22b": "✗",
                "qianfan-ocr": "✗", "deepseek-ocr": "✗",
                "pyzbar": "—"},
     "note": "Only Docling."},
    {"section": "Add-ons", "feature": "Multi-page document handling",
     "kind": "qual",
     "values": {"docling": "✓", "qwen-8b": "✓ (per-page)", "qwen-30b-a3b": "✓",
                "qwen-32b": "✓", "qwen-235b-a22b": "✓",
                "qianfan-ocr": "✓", "deepseek-ocr": "✓",
                "pyzbar": "✓"},
     "note": "All can process pages serially; cross-page coherence not benched."},
    {"section": "Add-ons", "feature": "Median latency (per page, lower better)",
     "kind": "score", "score_key": "latency_inv"},

    # ----- CUSTOM TRAINING -----
    {"section": "Custom Training", "feature": "Custom model trainer UI (Studio equivalent)",
     "kind": "qual",
     "values": {"docling": "✗", "qwen-8b": "✗", "qwen-30b-a3b": "✗",
                "qwen-32b": "✗", "qwen-235b-a22b": "✗",
                "qianfan-ocr": "✗", "deepseek-ocr": "✗",
                "pyzbar": "—"},
     "note": "No OSS equivalent of Azure DI Studio. Replace with Label Studio + Kubeflow Pipelines."},
    {"section": "Custom Training", "feature": "Fine-tuning available",
     "kind": "qual",
     "values": {"docling": "✓ (per component)", "qwen-8b": "✓ (LoRA)",
                "qwen-30b-a3b": "✓ (LoRA)", "qwen-32b": "✓ (LoRA, multi-GPU)",
                "qwen-235b-a22b": "✓ (multi-GPU)", "qianfan-ocr": "✓ (ERNIEKit)",
                "deepseek-ocr": "✓ (Unsloth)", "pyzbar": "—"},
     "note": "All have published fine-tune recipes; Qwen-8B is the cheapest path."},
]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def _format_score(stack: str, scores: dict, key: str, invert: bool = False) -> tuple[float | None, str]:
    """Return (numeric_value, display_string). Numeric is for ranking,
    display is what shows in the cell."""
    if key == "n_tables_norm":
        # normalize n_tables/(max=11 from Docling) to 0-1
        v = scores.get(stack, {}).get("n_tables")
        if v is None or pd.isna(v):
            return None, "—"
        return float(v) / 11.0, f"{int(v)}"
    if key == "latency_inv":
        v = scores.get(stack, {}).get("latency_med")
        if v is None or pd.isna(v):
            return None, "—"
        # show ms, but rank by inverse
        return -float(v), f"{int(v)} ms"

    v = scores.get(stack, {}).get(key)
    if v is None or pd.isna(v):
        return None, "—"

    if invert:
        # 1 - CER, clamped 0..1
        rank_val = max(0.0, min(1.0, 1.0 - float(v)))
        return rank_val, f"{rank_val:.3f}"
    return float(v), f"{float(v):.3f}"


def render_html() -> str:
    scores = _load_scores()

    rows_html: list[str] = []
    current_section = None
    section_order: list[str] = []

    for row in ROWS:
        section = row["section"]
        if section != current_section:
            section_order.append(section)
            rows_html.append(
                f'<tr class="section-row"><td colspan="{len(STACKS) + 2}">'
                f'<strong>{section}</strong></td></tr>'
            )
            current_section = section

        feature = row["feature"]
        kind = row["kind"]

        cells: list[tuple[str, float | None, str]] = []  # (stack_id, rank_value, display)
        for stack_id, _ in STACKS:
            if kind == "qual":
                disp = (row["values"] or {}).get(stack_id, "—")
                cells.append((stack_id, None, disp))
            else:
                rank_val, disp = _format_score(stack_id, scores, row["score_key"], invert=row.get("invert", False))
                cells.append((stack_id, rank_val, disp))

        # Identify winner(s) for numeric rows
        winners: set[str] = set()
        if kind == "score":
            ranked = [(stack_id, v) for stack_id, v, _ in cells if v is not None]
            if ranked:
                top = max(ranked, key=lambda t: t[1])[1]
                winners = {sid for sid, v in ranked if abs(v - top) < 1e-6}

        # Build the row
        cell_html: list[str] = [f'<td class="feature">{feature}</td>']
        for stack_id, _, disp in cells:
            cls = "win" if stack_id in winners else "norm"
            cell_html.append(f'<td class="{cls}">{disp}</td>')
        note = row.get("note", "")
        cell_html.append(f'<td class="note">{note}</td>')
        rows_html.append("<tr>" + "".join(cell_html) + "</tr>")

    header = (
        '<tr><th class="feature">Azure DI feature</th>'
        + "".join(f'<th>{label}</th>' for _, label in STACKS)
        + '<th class="note">Notes</th></tr>'
    )

    style = """
<style>
table.parity {
  width: 100%;
  border-collapse: collapse;
  font-size: 7.5pt;
  table-layout: fixed;
  page-break-inside: auto;
}
table.parity th, table.parity td {
  border: 1px solid #c8c8d0;
  padding: 3px 4px;
  vertical-align: top;
}
table.parity th { background:#eef0f4; font-weight:600; text-align:center; }
table.parity td.feature { text-align:left; font-weight:500; }
table.parity td.note { font-size:7pt; color:#555; }
table.parity td.norm { text-align:center; }
table.parity td.win  { text-align:center; background:#c8eccc; font-weight:600; color:#114011; }
table.parity tr.section-row td {
  background:#eaeaf0;
  font-size:8pt;
  letter-spacing:0.4px;
  text-transform:uppercase;
}
table.parity colgroup col.feature { width: 22%; }
table.parity colgroup col.stack   { width: 7%; }
table.parity colgroup col.note    { width: 22%; }
</style>
"""
    colgroup = (
        '<colgroup>'
        '<col class="feature">'
        + ''.join('<col class="stack">' for _ in STACKS)
        + '<col class="note">'
        '</colgroup>'
    )
    table = f'<table class="parity">{colgroup}<thead>{header}</thead><tbody>' + "".join(rows_html) + "</tbody></table>"
    return style + table


def main() -> None:
    html = render_html()
    out_html = DOCS / "parity_matrix.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"wrote {out_html}")

    # Inject into workshop_report.md between markers
    md_path = DOCS / "workshop_report.md"
    md = md_path.read_text(encoding="utf-8")
    BEGIN = "<!--PARITY_MATRIX_BEGIN-->"
    END = "<!--PARITY_MATRIX_END-->"
    if BEGIN in md and END in md:
        before, _, rest = md.partition(BEGIN)
        _, _, after = rest.partition(END)
        new = before + BEGIN + "\n\n" + html + "\n\n" + END + after
        md_path.write_text(new, encoding="utf-8")
        print(f"injected matrix into {md_path}")
    else:
        print("WARN: markers not found in workshop_report.md — not injecting.")


if __name__ == "__main__":
    main()
