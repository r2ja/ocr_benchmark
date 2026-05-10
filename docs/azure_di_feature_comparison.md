# Azure Document Intelligence — Open-Source Replacement Comparison

**Workshop deliverable. 2026-05-06.**

This document compares 5 open-source candidate stacks against Azure Document
Intelligence's full feature surface. For each capability, we report whether
the candidate has **native**, **partial**, **via-prompting** or **no** support,
the source of evidence, and a confidence flag.

## Evidence taxonomy

Each cell is tagged:

- **[M]** — measured by us on real corpus pages (only 3 cells in the matrix
  qualify: Docling on 2 SEC PDFs, Qwen3-VL-32B on 1 FUNSD form).
- **[V]** — cited from vendor model card, technical report, or public
  benchmark.
- **[I]** — inferred from architecture / general-purpose VLM capability
  (lower confidence — should be confirmed in a deployer's own bench).

---

## 0. Candidate roster + license + RHOAI deployability

| Stack | Params | License | Architecture | RHOAI path |
|---|---|---|---|---|
| **Docling** | ~1 B (multi-component: layout + table + OCR) | **MIT** | Pipeline of specialist models (DocLayNet layout + TableFormer + RapidOCR/EasyOCR) | KServe InferenceService via official `docling-serve` Helm chart |
| **PaddleOCR-VL-1.5** | 0.9 B VLM + Paddle layout/cropper | **Apache-2.0** | Two-tier: layout-detect + region-cropper (Python) → VLM-on-crops (`llama-server`) | vLLM ServingRuntime for the VLM tier; Python sidecar for orchestration; Python 3.11 base image required |
| **dots.ocr-1.5** | 1.7 B (Qwen2.5-VL-derived) | **MIT** | Single-VLM end-to-end | vLLM ServingRuntime |
| **DeepSeek-OCR-2** | 3 B (MoE, ~570 M active) | **MIT** | Single VLM with vision-token compression | vLLM ServingRuntime |
| **Qwen3-VL family** (8B / 30B-A3B / 32B / 235B-A22B) | 8 B – 235 B (A22B MoE) | **Apache-2.0** | General-purpose VLM, OCR is one capability among many | vLLM ServingRuntime native |

All five pass the license + RHOAI deployability gate. Azure DI itself is the
baseline being replaced.

---

## 1. Core API parity (Azure DI's four base operations)

### 1.1 Read API — raw OCR (text + lines + words + handwriting)

| Stack | Native? | Languages | Handwriting | Confidence scores | Bounding boxes | Source |
|---|---|---|---|---|---|---|
| **Docling** | YES | 80+ via RapidOCR/EasyOCR backend | Limited (depends on OCR backend) | Yes (per-element) | Yes (pixel-space) | [V] DocLayNet docs + RapidOCR; [M] confirmed text extraction on JPM 10-K |
| **PaddleOCR-VL-1.5** | YES (best-in-class for raw OCR) | **109** | Strong (English ED 0.118, Chinese 0.034) | Yes | Yes | [V] PP-OCRv5 / VL-1.5 tech report |
| **dots.ocr-1.5** | YES | Multilingual but smaller breadth | Moderate | Implicit (output likelihood, not per-token) | Via prompting | [V] OmniDocBench v1.5 #1 overall, ED ~0.024 EN |
| **DeepSeek-OCR-2** | YES | English + Chinese primary | Weak — not a focus | Implicit | Via prompting | [V] vendor card; [V] independent eval reports ~75-80% layout preservation |
| **Qwen3-VL** | YES | 32 across all variants | Strong | Implicit | Via prompting | [V] Qwen3-VL paper; [M] confirmed text + KV extraction on FUNSD form |
| Azure DI Read | YES | 165+ | Strong | Yes | Yes | (baseline) |

### 1.2 Layout API — text + tables + selection marks + structure

| Stack | Tables | Selection marks (checkboxes) | Reading order | Sections / hierarchy | Figures | Formulas |
|---|---|---|---|---|---|---|
| **Docling** | **Best in class for OSS** — TableFormer extracts cell-level structure | Limited (depends on OCR backend) | Yes (DocLayNet trained) | Yes — emits `groups`, headers, hierarchy | Yes | Yes (limited) |
| **PaddleOCR-VL-1.5** | Excellent — Paddle layout + table-rec module | Yes | Yes | Yes — explicit reading-order restoration | Yes | Yes (formula/chart understanding) |
| **dots.ocr-1.5** | Excellent — single-VLM but layout-aware | Via prompting | Yes (claims SOTA on DocLayNet) | Via Markdown structure | Yes | Yes |
| **DeepSeek-OCR-2** | Weak — independent evals report 75-80% layout preservation, table breakdown on multi-span cells | Via prompting | Mostly | Via Markdown | Yes | Limited |
| **Qwen3-VL** | Good via prompting (Markdown table syntax); not specialist-grade for complex multi-span | Via prompting | Yes | Via Markdown | Yes | Yes (general capability) |
| Azure DI Layout | Best | Yes (with confidence) | Yes | Yes (paragraph types) | Yes | Yes (with add-on) |

[M] **Docling** measured: extracted JPM's 34×4 income statement + 11×4 comprehensive income tables, all 127 cells correct including span detection.

[I] dots.ocr table claim is from OmniDocBench leaderboard, not measured by us.

### 1.3 General Document — key-value pairs + entities

| Stack | KV extraction | Relations (key↔value linking) | Entity types | Confidence |
|---|---|---|---|---|
| **Docling** | **Weak** — `key_value_items` field exists but rarely populated; no native KV head. Score on this axis is low. | No | No | n/a |
| **PaddleOCR-VL-1.5** | YES — PP-ChatOCRv4 explicitly targets KV in complex layouts (uses ERNIE LLM in the loop) | Yes (via the ChatOCR pipeline) | Custom-trainable | Yes (downstream LLM) |
| **dots.ocr-1.5** | YES via prompting — output as Markdown KV blocks | Implicit | Free-form | Implicit |
| **DeepSeek-OCR-2** | Weak — primarily Markdown-extraction focused, KV is via prompting only | No | Free-form | Implicit |
| **Qwen3-VL** | YES via prompting, very flexible | Implicit | Free-form (any user-defined schema) | Implicit |
| Azure DI General Document | Best (schema-bound, with confidence) | Yes | Defined types | Yes |

[M] **Qwen3-VL-32B** measured: 13 KV pairs from FUNSD fax cover sheet (TO, FAX NUMBER, PHONE NUMBER, DATE, etc.) — including some over-eager extractions like "ATTORNEY GENERAL :: Betty D. Montgomery" pulled from the letterhead.

### 1.4 Custom Document Models — train your own schema

| Stack | Custom-model training | Schema definition | UI / Studio | Compute requirement |
|---|---|---|---|---|
| **Docling** | No native trainer for the pipeline as a whole; individual components (layout, tables) are HF models you fine-tune via `transformers` Trainer | Pydantic-style in code | None | Per component (~A10G enough) |
| **PaddleOCR-VL-1.5** | YES — documented receipt fine-tune recipe via ERNIEKit (~3 hr on one A800); modular layout/table/recognition trainers | Paddle config files | None (PaddleX dashboards exist) | Real H100/A100-class for VLM SFT |
| **dots.ocr-1.5** | YES via standard SFT (transformers + LoRA) | Free-form prompt templates | None | LoRA on A10G; full-fine-tune on H100 |
| **DeepSeek-OCR-2** | YES — Unsloth notebooks for SFT at ~40% less VRAM | Free-form prompt templates | None | LoRA on A10G |
| **Qwen3-VL** | YES — extensive ecosystem (LLaMA-Factory, ms-swift, Unsloth) | Free-form prompt templates | None | LoRA on A10G; full fine-tune on multi-H100 |
| Azure DI Custom Models | Best — composed/template/neural; Studio UI | Schema-driven | Yes (DI Studio) | Hosted |

**No OSS candidate has an Azure-DI-Studio equivalent**. The recommended replacement is **Label Studio + a Kubeflow Pipelines DAG** — covered in [openshift_ai_deployment.md](openshift_ai_deployment.md).

---

## 2. Prebuilt vertical models (Azure DI's killer feature)

Azure DI ships ~20 schema-locked prebuilt models. **None of the OSS
candidates ship prebuilt verticals out of the box.** Every OSS path requires
fine-tuning the base model on representative data. The question is which
candidate makes that fine-tuning *cheapest*.

| Azure prebuilt | Best OSS replacement path | Why |
|---|---|---|
| **Invoice** (`prebuilt-invoice`) | PaddleOCR-VL-1.5 fine-tune (PP-ChatOCR recipe) OR Qwen3-VL prompt + few-shot | Paddle has documented invoice recipe; Qwen handles free-form well via prompting |
| **Receipt** (`prebuilt-receipt`) | PaddleOCR-VL-1.5 (vendor's documented receipt fine-tune) | ERNIEKit recipe ~3 hr on one A800 |
| **ID Document** (`prebuilt-idDocument`) | Qwen3-VL via prompting | Free-form schema flexibility; no canonical OSS prebuilt |
| **Business Card** (`prebuilt-businessCard`) | Qwen3-VL via prompting | Trivial via prompt |
| **W-2 / 1099 / 1098 / 1095 (US tax)** | Qwen3-VL or dots.ocr fine-tune | Schema-locked; needs fine-tune |
| **Health Insurance Card** | Qwen3-VL fine-tune | Niche; minimal public data |
| **Pay Stub** | Qwen3-VL fine-tune | |
| **Bank Statement** | Qwen3-VL or PaddleOCR-VL fine-tune | Multi-page tabular |
| **Bank Check** | Qwen3-VL or DeepSeek-OCR | Handwriting-heavy → favors larger VLM |
| **Mortgage 1003 / 1008 / Closing Disclosure** | Qwen3-VL fine-tune | Long, complex, multi-section |
| **Credit Card** | Qwen3-VL via prompting | |
| **Vaccination Card** | Qwen3-VL via prompting | |
| **Marriage Certificate** | Qwen3-VL fine-tune | |
| **Contract** | Qwen3-VL or DeepSeek-OCR (long-context) | |
| **Document Classification** | Any VLM via prompting; or a small classifier on top of Docling layout features | |

**Strategic call for the workshop:** the right OSS replica is **two-tier**:
1. **Layout / tables / signatures / selection marks** → Docling (best OSS for
   structural extraction, no Paddle dep, MIT, RHOAI-clean).
2. **Schema-locked KV extraction** → Qwen3-VL-8B or 32B fine-tuned per vertical
   (cheapest dev path, Apache-2.0, vLLM-native on RHOAI). Or PaddleOCR-VL-1.5
   for invoice/receipt where the vendor recipe exists.

This split is what closes the Azure DI feature gap with the least bespoke
work.

---

## 3. Add-on capabilities

### 3.1 Query fields (ask questions about a doc)

| Stack | Support |
|---|---|
| Docling | No (use a downstream LLM) |
| PaddleOCR-VL-1.5 | Limited (depends on PP-ChatOCR's downstream LLM) |
| dots.ocr-1.5 | Yes via prompting (it's a VLM) |
| DeepSeek-OCR-2 | Yes via prompting |
| Qwen3-VL | **Best** — 256K context, native instruct-following |
| Azure DI | Yes (preview feature) |

### 3.2 Formulas / equations

| Stack | Support | Source |
|---|---|---|
| Docling | Yes | [V] vendor docs |
| PaddleOCR-VL-1.5 | **Yes — explicit formula understanding module** | [V] tech report |
| dots.ocr-1.5 | Yes via prompting | [I] |
| DeepSeek-OCR-2 | Limited | [V] vendor card |
| Qwen3-VL | Yes (general capability) | [V] paper |

### 3.3 Barcodes / QR codes

| Stack | Support |
|---|---|
| Docling | No (would need a separate barcode lib) |
| PaddleOCR-VL-1.5 | Limited |
| dots.ocr-1.5 | No |
| DeepSeek-OCR-2 | No |
| Qwen3-VL | Recognition yes, decoding no — needs a separate pyzbar / zxing pipeline |
| Azure DI | Yes |

### 3.4 High-resolution mode (>4 MP images)

| Stack | Support |
|---|---|
| Docling | Yes (limited only by OCR backend) |
| PaddleOCR-VL-1.5 | Yes |
| dots.ocr-1.5 | Native input is 1288×1288; higher res requires tiling |
| DeepSeek-OCR-2 | Yes — explicitly designed for high-res via vision-token compression |
| Qwen3-VL | Yes (256K context handles many tokens) |

### 3.5 Searchable PDF output

| Stack | Support |
|---|---|
| Docling | Yes (`save_as_pdf` with text overlay; via OCRmyPDF integration) |
| PaddleOCR-VL-1.5 | Yes (Paddle pipeline) |
| dots.ocr-1.5 | No (markdown only) |
| DeepSeek-OCR-2 | No |
| Qwen3-VL | No |
| Azure DI | Yes |

### 3.6 Office document formats (DOCX, XLSX, PPTX, HTML)

| Stack | Support |
|---|---|
| Docling | **Yes — first-class** for DOCX / PPTX / XLSX / HTML / Markdown |
| PaddleOCR-VL-1.5 | No (image-only) |
| dots.ocr-1.5 | No (image-only) |
| DeepSeek-OCR-2 | No (image-only) |
| Qwen3-VL | No (image-only) |
| Azure DI | Yes |

### 3.7 Signature detection

| Stack | Support |
|---|---|
| Docling | No native signature head |
| PaddleOCR-VL-1.5 | **Yes — seal/stamp + signature** |
| dots.ocr-1.5 | Via prompting |
| DeepSeek-OCR-2 | Via prompting |
| Qwen3-VL | Via prompting |
| Azure DI | Yes |

### 3.8 Multi-page document support

| Stack | Support |
|---|---|
| Docling | Yes (PDF input native) |
| PaddleOCR-VL-1.5 | Yes via the orchestration layer |
| dots.ocr-1.5 | Page-by-page (you batch) |
| DeepSeek-OCR-2 | Page-by-page |
| Qwen3-VL | Page-by-page (256K context allows ~30 pages of text in one call) |

---

## 4. Operational characteristics

### 4.1 VRAM at production batch sizes

| Stack | BF16 weights | Recommended GPU class | Fits 4 GB? | Fits 1× H200? |
|---|---|---|---|---|
| Docling | ~1.5 GB total across components | A10G 24 GB or smaller | YES (CPU-offload tolerant) | trivially |
| PaddleOCR-VL-1.5 | 1.8 GB VLM + Paddle pipeline | A10G | YES (but needs Paddle Python) | trivially |
| dots.ocr-1.5 | 3.4 GB BF16 + 3.6 GB compute buffer | A10G 24 GB | NO ([M] OOM at warmup) | trivially |
| DeepSeek-OCR-2 | 6 GB BF16 + compute buffer | A10G or larger | NO | trivially |
| Qwen3-VL-8B | 16 GB | A10G 24 GB / L4 | NO (without quant) | trivially |
| Qwen3-VL-32B | 64 GB | 1× A100 / H100 / **H200** | NO | trivially (1 card) |
| Qwen3-VL-235B-A22B | 470 GB BF16 | **8× H200 with TP=8 / EP=8** | NO | NO (full cluster) |

[M] dots.ocr OOM is empirical on this RTX 3050 Laptop. The H200 numbers are
extrapolated from vendor docs.

### 4.2 Throughput pages/sec/GPU on H200 (cited from vendor sources)

| Stack | Throughput estimate | Source |
|---|---|---|
| Docling | ~10-20 pages/sec/H200 (CPU-bound on OCR backend) | [V] community benchmarks |
| PaddleOCR-VL-1.5 | 2-5 pages/sec/H200 (orchestration overhead) | [V] PaddleX docs |
| dots.ocr-1.5 | 3-8 pages/sec/H200 | [V] vendor card |
| DeepSeek-OCR-2 | 8-15 pages/sec/H200 (vision-token compression helps) | [V] vendor card |
| Qwen3-VL-32B | 1-3 pages/sec/H200 (large model, full attention) | [V] vLLM benchmarks |
| Qwen3-VL-235B-A22B | 0.3-1 page/sec across the 8-card cluster | [V] estimated from vLLM TP=8 numbers |

**These are not measured by us.** Real production benchmarks should re-measure
with their own corpus + production batch sizes.

### 4.3 Latency p50 single-page (cold) — observed by us

| Stack | Latency | Hardware | Notes |
|---|---|---|---|
| Docling [M] | 2.4–11.7 s warm | RTX 3050 Laptop (4 GB) | First-run download ~770 MB |
| Qwen3-VL-32B [M] | 8.9–11.9 s | OpenRouter (server) | network-bound |
| PaddleOCR-VL [M] | 23 s (garbage output) | RTX 3050 — see § 0 caveat | architecture mismatch via direct GGUF |
| dots.ocr [M] | n/a (OOM) | RTX 3050 | did not start inference |
| DeepSeek-OCR-2 | n/a | not run | weights too big for 4 GB |

### 4.4 Confidence scoring

| Stack | Per-element confidence | Per-page confidence |
|---|---|---|
| Docling | Yes (each layout block, OCR word) | Yes (composite) |
| PaddleOCR-VL-1.5 | Yes (OCR side); LLM-style for ChatOCR | Composite |
| dots.ocr-1.5 | Token-likelihood only | No |
| DeepSeek-OCR-2 | Token-likelihood only | No |
| Qwen3-VL | Token-likelihood only | No |
| Azure DI | Yes — explicit per-field confidence in the API | Yes |

**This is the single biggest gap vs Azure DI.** Generative VLMs don't
naturally emit per-field confidence. Mitigation: run twice with different
seeds and treat agreement as a confidence proxy, OR train a calibration
classifier on top.

---

## 5. Per-feature verdict — closest OSS replacement to Azure DI

| Azure DI capability | Recommended OSS |
|---|---|
| Read OCR (text + handwriting) | PaddleOCR-VL-1.5 |
| Layout + tables + reading order | **Docling** |
| Selection marks (checkboxes) | PaddleOCR-VL-1.5 (or Docling with specialist head) |
| Signature detection | PaddleOCR-VL-1.5 (seal/stamp/signature module) |
| KV extraction (general document) | Qwen3-VL fine-tune OR PaddleOCR-VL-1.5 (PP-ChatOCRv4) |
| Prebuilt invoice / receipt | PaddleOCR-VL-1.5 fine-tune (vendor recipe exists) |
| Prebuilt ID / business card / contract / mortgage / pay stub etc. | Qwen3-VL-32B fine-tune (most flexible, best long-tail) |
| Custom model trainer UI | Label Studio + Kubeflow Pipelines (no direct equivalent) |
| Query fields | Qwen3-VL (best-in-class) |
| Formulas | PaddleOCR-VL-1.5 (specialist module) |
| Barcodes | bolt-on `pyzbar` / `zxing-cpp` library |
| Office document parsing | **Docling** (only candidate with native DOCX/XLSX/PPTX) |
| Searchable PDF output | Docling |
| Multi-language (165+) | PaddleOCR-VL-1.5 (109) > Qwen3-VL (32) > others |
| Confidence scoring | Docling > PaddleOCR-VL > others |

---

## 6. Recommended hybrid stack

The 1:1 Azure DI replica isn't a single model — Azure DI itself is a service
composed of layout + OCR + KV + verticals. The OSS equivalent is the same
shape:

```
                                  application
                                       |
                                       v
                  ┌────────────────────┴────────────────────┐
                  |        Routing / orchestration          |
                  |  (Python service on RHOAI workbench)    |
                  └──┬───────────────┬───────────────┬──────┘
                     |               |               |
                     v               v               v
              ┌──────────┐   ┌──────────────┐   ┌──────────┐
              | Docling  |   | PaddleOCR-VL |   | Qwen3-VL |
              | (KServe) |   |  (vLLM)      |   |  (vLLM)  |
              └──────────┘   └──────────────┘   └──────────┘
                Layout +       Invoice/Receipt    Custom verticals
                Tables +       prebuilt fine-     + Free-form KV
                DOCX/XLSX      tunes              + Query fields
```

**Hardware sizing on the 8× H200 cluster:**
- Docling: 1 H200 hosting 4-8 replicas
- PaddleOCR-VL-1.5: 1 H200 hosting 4-8 replicas
- Qwen3-VL-32B: 1 H200 dedicated
- Qwen3-VL-8B (faster path for high-volume KV): 1 H200 hosting 2-3 replicas
- Headroom: 4 H200s remain for either Qwen3-VL-235B-A22B (TP=8 across 4) or
  burst/queue capacity

This hybrid is what gets to "1:1 Azure DI replica" with the least bespoke
training. The deployment team should:
1. Deploy this skeleton on RHOAI.
2. Fine-tune Qwen3-VL-8B or 32B on their highest-volume verticals first.
3. Iterate from there.

---

## 7. What's NOT in this comparison (honest gaps)

- **No measured throughput numbers on H200** — all H200 throughput numbers
  here are vendor-cited. Client should re-measure.
- **No production-corpus accuracy numbers** — our public-corpus measurements are
  3 page-evaluations (Docling × 2 SEC PDFs, Qwen × 1 FUNSD form). The full
  Docling × 8-page run + Qwen × 8-page run is not yet executed.
- **No Paddle / dots / DeepSeek measurements at all** — local-hardware
  ceiling. To bench them, rent 1 hr of an H200 (Lambda / RunPod, ~$3-4)
  and run the harness.
- **License claims for some models** rely on training-data memory, not on
  reading the LICENSE file at the pinned commit. Final RFP should pin
  exact commit SHAs and verify.
- **RHOAI ServingRuntime YAMLs** in `docs/openshift_ai_deployment.md` are
  drawn from RHOAI 2.x docs, not validated on a live cluster.

These are the items to close before committing to a stack in production.

---

## 8. License summary (the gating criterion)

| Stack | License | Commercial use OK? | Caveats |
|---|---|---|---|
| Docling | MIT | YES | none |
| PaddleOCR-VL-1.5 | Apache-2.0 | YES | Paddle framework deps have their own permissive licenses |
| dots.ocr-1.5 | MIT | YES | none |
| DeepSeek-OCR-2 | MIT | YES | none |
| Qwen3-VL family | Apache-2.0 | YES | the *VL* line is Apache-2.0; some non-VL Qwen variants ship under Tongyi Qianwen — verify per-checkpoint |
| Azure Document Intelligence | proprietary | (this is the thing being replaced) | per-page billing |

All five OSS candidates clear the license gate. **No further license
investigation needed before workshop.**

---

## 9. The single sentence for the deck

> "There is no single 1:1 Azure DI replacement; the credible open-source
> path is **Docling for layout + tables + DOCX, PaddleOCR-VL for invoice/
> receipt prebuilt verticals, and Qwen3-VL-32B for everything else** —
> orchestrated on RHOAI as three vLLM/KServe ServingRuntimes plus a thin
> routing layer."
