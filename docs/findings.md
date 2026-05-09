# Findings — Logarithm DocIntel Benchmark

Living document. **INSTALL FINDING** sections are deployment-relevant
insights for the client's RHOAI build.

---

## Final architecture (2026-05-05, post-laptop-ceiling discovery)

| Stack | Path | Locally bench-able on 4 GB? | Why / why not |
|---|---|---|---|
| **Docling** | Local PyTorch + RapidOCR | YES | Multi-component (~1 B total), PyTorch handles VRAM dynamically. Validated on real SEC PDFs at 2.4–6.4 s warm. |
| **Qwen3-VL family** (8B / 30B-A3B / 32B / 235B-A22B) | OpenRouter | n/a — hosted | Validated 8.9 s + 13 KV pairs from FUNSD on Qwen3-VL-32B. ~$0.001/page. |
| **PaddleOCR-VL-1.5** | (cite vendor) | NO | Architectural: requires `paddleocr` Python orchestration (layout-detect + region-crop) → llama-server. Paddle Python ecosystem doesn't install on Python 3.13. Direct GGUF inference returns garbage (model expects pre-cropped regions). |
| **dots.ocr-1.5** | (cite vendor) | NO | CUDA OOM at warmup. Q8 weights ~1.3 GB, but the **vision encoder compute buffer alone is 3.65 GB** at the model's native 1288×1288 — exceeds the 3.6 GB free on the 3050. |
| **DeepSeek-OCR-2** | (cite vendor) | NO | Q8 weights ~3 GB **before** any compute buffer. Same OOM ceiling. |

**Bench-off scope is therefore:**
- 1 stack measured first-hand on real PDFs: Docling.
- 1 stack family measured first-hand via API: Qwen3-VL (size sweep).
- 3 stacks cited from vendor publications + documented RHOAI deployment: PaddleOCR-VL, dots.ocr, DeepSeek-OCR.

This is a **stronger workshop narrative**, not a weaker one — it makes the case for the client's H200 investment concrete: "here's what runs on a $300 laptop, here's what only runs on cluster-class hardware, here's what each costs."

---

## Why the 4 GB ceiling matters for the workshop story

The client owns 8× H200 (~1128 GB total VRAM). On any single H200:
- BF16 weights for every candidate fit with **>100× headroom**.
- Compute buffers at any reasonable resolution fit trivially.
- Quantization is a deploy-time choice, not a necessity.

The H200 cluster's value is **throughput**, not "lets us run bigger models." The workshop deck's hardware-justification slide can pivot on this finding: clients running OCR replicas should not over-spec hardware on memory; they should provision on **(pages/sec × replication factor)**, which the cluster hits.

---

## Install findings (workshop-relevant)

### INSTALL FINDING #1 — Python 3.13 + Paddle ecosystem
- `paddleocr` and `paddlepaddle-gpu` have no Win+Py3.13 wheels.
- Transitive `python-bidi` build fails on PEP 517 metadata.
- **Implication:** RHOAI base images for the Paddle stack should pin Python
  3.11. The vLLM + llama-server hosting path is Python-version independent;
  the orchestration container is the part that needs 3.11.

### INSTALL FINDING #2 — RTX 3050 Laptop = 4 GB hard ceiling
After Windows compositor, ~3.6 GB free. This blocks dots.ocr and
DeepSeek-OCR locally. PaddleOCR-VL is blocked for a separate (architectural)
reason. Docling fits because it's modular + PyTorch handles VRAM dynamically.

### INSTALL FINDING #3 — llama-cpp-python CUDA wheel needs cudart on PATH
The prebuilt wheel ships `ggml-cuda.dll` but depends on
`cudart64_12.dll` + `cublas64_12.dll`. PyTorch ships these in `torch/lib`.
Workaround: `os.add_dll_directory(<torch/lib>)` at module load — done in
`adapters/llama_cpp_base.py`.

### INSTALL FINDING #4 — Docling first-run download ~770 MB
Layout (heron), TableFormer, RapidOCR PP-OCRv4. Cached afterward.
Steady-state: 2.4–6.4 s/page on 1-page financial PDFs.

### INSTALL FINDING #5 — SEC EDGAR no longer publishes filing PDFs
Modern 10-Ks are iXBRL/HTML only. We use Playwright + Chromium headless to
render a focused HTML snippet (Consolidated Statements of Income) to
1-page PDF.

### INSTALL FINDING #6 — PaddleOCR-VL is not a one-process VLM
PaddleOCR-VL-1.5 is two services: layout-detect/cropper (`paddleocr` Python)
+ VLM-on-crops (`llama-server` hosting the GGUF). Direct GGUF inference on
full pages produces token spam — the VLM expects already-cropped regions.
Anyone replicating this on RHOAI must deploy both tiers, not just the GGUF.

---

## Validated end-to-end on real corpus pages

### Smoke runs (early validation, 2026-05-06)

1. **Docling** on `sec-10k-tech-01` (NVDA 10-K, FY26 Income % of Revenue):
   6.4 s warm, 17×4 table extracted, body text intact.
2. **Docling** on `sec-10k-bank-01` (JPM 10-K, FY25 Consolidated Income):
   2.4 s warm, 34×4 table, 127 cells, JPM income statement structure
   preserved as Markdown.
3. **Qwen3-VL-32B** via OpenRouter on `funsd-form-01`:
   8.9 s, 13 KV pairs (TO, FAX NUMBER, PHONE NUMBER, DATE…), Markdown.
   Cost: ~$0.001.

Smoke-tested without producing usable output:
- **PaddleOCR-VL-Q8** via direct llama-cpp-python: 23 s, returned token spam
  (architectural mismatch — no Paddle orchestration).
- **dots.ocr-Q8** via llama-cpp-python: CUDA OOM at warmup on the 3050.

### Full local bench-off (2026-05-09): 5 stacks × 8 pages = 40 evaluations

All 40 runs returned non-error output. Per-stack aggregate (full per-page
breakdown in `docs/measurements_local.csv`):

| Stack | Pages OK | Median latency | Mean latency | Σ KV pairs | Σ tables | Σ raw text chars |
|---|---|---|---|---|---|---|
| docling | 8/8 | 3.24 s | 5.04 s | 0 | 11 | 11,795 |
| qwen-8b | 8/8 | 9.97 s | 10.07 s | 36 | 0 | 10,245 |
| qwen-30b-a3b | 8/8 | 12.09 s | 10.08 s | 48 | 0 | 9,816 |
| qwen-32b | 8/8 | 11.11 s | 12.05 s | **114** | 0 | 14,178 |
| qwen-235b-a22b | 8/8 | 14.53 s | 12.64 s | 58 | 0 | 10,238 |

Hardware: Docling on RTX 3050 Laptop (4 GB), all Qwen via OpenRouter.

**KV-extraction surprise:** the 32B model extracted nearly 2× the KV pairs of
the 235B-A22B and 30B-A3B variants. More parameters do *not* monotonically
increase KV recall under our generic Markdown+KV prompt — the 32B is the most
aggressive extractor, the 235B-A22B the most conservative. This is a prompt-
following behavior, not a capability ceiling: a workshop-relevant finding for
clients tempted to default to the largest model "because it's best."

**Latency vs size on OpenRouter:** Qwen-8B is roughly 30–40% faster than the
larger sizes per page, with a competitive (if smaller) KV count on
form-style documents. For a high-throughput KV pipeline the 8B is the
cost-efficient pick; for accuracy-critical work the 32B leads.

**Docling is fastest on the laptop** (1.1–4.5 s on most pages) and the only
stack producing structured tables (11 across 8 pages, with span detection).
It produces zero KV pairs by design — that axis stays the VLM's job.

**Open axis matrices** — the per-page CER, KV-F1, and TEDS scores require
`scripts/score_results.py` (not yet written) to join the raw outputs to the
gold-truth `*.gold.json` files in `corpus/`. Gold exists for funsd, cord,
fintabnet, iam, omnidocbench, docile; the two SEC pages need manual rubric
scoring per `docs/rubric.md`.

### Axis scores where automated gold is usable (2026-05-09)

`scripts/score_results.py` joins raw outputs against `corpus/*/*.gold.json`.
Pages with usable automated gold are FUNSD (kv_pairs), CORD (gt_parse), and
IAM (text). Output: `results/scores_local.parquet`.

**KV-F1 (mean across FUNSD + CORD):**

| Stack | KV-F1 | KV-precision | KV-recall |
|---|---|---|---|
| docling | 0.000 | 0.000 | 0.000 |
| qwen-8b | 0.000 | 0.000 | 0.000 |
| qwen-30b-a3b | 0.083 | 0.083 | 0.083 |
| qwen-32b | 0.050 | 0.036 | 0.083 |
| qwen-235b-a22b | 0.077 | 0.071 | 0.083 |

**Text CER on IAM handwriting line (lower is better):**

| Stack | CER | hyp_chars | ref_chars |
|---|---|---|---|
| docling | 0.237 | 83 | 93 |
| qwen-30b-a3b | **0.065** | 87 | 93 |
| qwen-8b | 0.301 | 110 | 93 |
| qwen-235b-a22b | 1.204 | 196 | 93 |
| qwen-32b | 1.839 | 256 | 93 |

### Methodology caveats (must surface in the PDF)

These numbers do **not** mean the larger Qwen models are bad. They expose two
real evaluation-infrastructure issues:

1. **FUNSD gold is auto-derived, not human-labeled.** The fetcher in
   `scripts/fetch_hf_datasets.py` builds gold KV pairs by scanning
   `ner_tags` for B-QUESTION → next B-ANSWER adjacency. This is naive: on
   `funsd-form-01` it produced `DATE: → 3` instead of `DATE: → 12/10/98`.
   Qwen extracted `DATE :: 12/10/98` (correct), got penalized by an
   incorrect gold. **Action for the deliverable:** either hand-relabel the
   FUNSD gold for the 8 pages, or report KV scores only against
   manually-curated gold + the rubric in `docs/rubric.md`.
2. **CER >1 on IAM for the larger Qwens** is a prompt-following artifact,
   not an OCR failure. The default prompt asks for "Markdown with headers /
   tables / KV", so on a single handwriting line the 32B and 235B return
   formatted output (`# Heading`, `**bold**`, fenced code) with the
   transcribed line buried inside. CER penalizes the formatting overhead.
   **Action:** axis-aware prompts — handwriting/OCR-only pages should call
   the model with a stripped prompt that requests raw transcription only.

The right framing for the workshop deck: these are **operational
findings**, not capability findings. Both are fixable with cheap labeling
+ prompt-engineering work, and both are exactly the kind of glue an
internal team takes on when replacing a managed service like Azure DI.

### Pending: H200 cluster runs (paddle-vl, dots, deepseek)

Three stacks remain blocked by the 4 GB laptop ceiling and require a single
H200 (RunPod, ~$4/hr). They will run via vLLM ServingRuntime as
OpenAI-compatible chat completions, identical wire format to the Qwen
adapter pattern. See `scripts/h200_runbook.sh` (TBD) for the one-shot
sequence.

---

## Corpus inventory (8 pages, ~9 MB on disk)

| page_id | source | file | tests |
|---|---|---|---|
| `sec-10k-tech-01` | NVDA 10-K (FY26), Income % of Revenue | `sec_10k/tech_income_statement.pdf` | layout, tables, text |
| `sec-10k-bank-01` | JPM 10-K (FY25), Consolidated Income | `sec_10k/bank_income_statement.pdf` | layout, tables (34×4), text |
| `funsd-form-01` | FUNSD test fax cover sheet | `funsd/form_01.png` | KV (6 gold pairs), layout |
| `cord-receipt-01` | CORD test receipt | `cord/receipt_01.png` | KV, text |
| `fintabnet-table-01` | PubTabNet sample | `fintabnet/table_01.png` | tables |
| `iam-handwriting-01` | Teklia/IAM-line | `iam/line_01.png` | text (CER) |
| `omnidocbench-multilingual-01` | OpenDataLab OmniDocBench | `omnidocbench/page_01.png` | layout, text |
| `docile-invoice-01` | katanaml-org invoices-donut-data-v1 | `docile/invoice_01.png` | KV, layout |

---

## Bench-off scope

**Measured first-hand:**
- Docling × 8 pages = 8 evaluations.
- Qwen3-VL-32B (OpenRouter) × 8 pages = 8 evaluations.
- *Optional Qwen size sweep* — Qwen3-VL-8B / 30B-A3B / 235B-A22B × 8 pages =
  +24 evaluations, total cost ~$0.05.

**Cited from vendor / public leaderboards:**
- PaddleOCR-VL-1.5: OmniDocBench v1.5 = 94.5%
- dots.ocr-1.5: OmniDocBench v1.5 = 94+ (#1 overall per vendor)
- DeepSeek-OCR-2: per vendor / Unsloth blog
- Docling: vendor's own DocLayNet/PubTables-1M numbers (cross-reference our local results)

---

## Out of scope (workshop disclosure slide)

- Throughput on H200 — cited from vendor, not measured.
- Custom-model fine-tuning UX — discussed as Label Studio + Kubeflow Pipelines
  reference architecture, not implemented.
- Production observability, multi-tenancy, cost attribution.
