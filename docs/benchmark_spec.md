# Benchmark Spec

## Hard filter — RHOAI deployability
A stack is in the bench-off only if it deploys on Red Hat OpenShift AI as a stock vLLM ServingRuntime or KServe InferenceService — no custom non-PyTorch base image. Stacks failing this gate are dropped on principle, not benchmarked.

| Stack | RHOAI deployable | Decision |
|-------|------------------|----------|
| Qwen3-VL | vLLM ServingRuntime native | KEEP |
| Docling + docling-serve | KServe InferenceService, Helm chart | KEEP |
| dots.ocr-1.5 | vLLM-compatible | KEEP |
| DeepSeek-OCR-2 | vLLM-compatible | KEEP |
| PaddleOCR-VL-1.5 | vLLM recipe published | KEEP |
| PP-StructureV3 + PP-ChatOCRv4 | Paddle framework + ERNIE deps, no clean ServingRuntime | DROP |
| Azure Document Intelligence | (baseline only, not OSS) | DROP — workshop is OSS-replacement-focused |

## Stacks under test
| ID | Stack | Params | Where we run it | License |
|----|-------|--------|-----------------|---------|
| docling | Docling + docling-serve | ~1B (layout+table) | Local RTX 3050 | MIT |
| paddle-vl | PaddleOCR-VL-1.5 | 0.9B | Local RTX 3050 | Apache-2.0 |
| dots | dots.ocr-1.5 | 1.7B | Local RTX 3050 (BF16/INT8) or Replicate fallback | MIT |
| deepseek | DeepSeek-OCR-2 | ~3B (MoE active ~570M) | Replicate (local needs 4-bit) | MIT |
| qwen-32b | Qwen3-VL-32B-Instruct | 32B | Replicate (won't fit 3050) | Apache-2.0 |

Single API platform: **Replicate**. $5 minimum deposit. One token, pay-per-second, covers Qwen3-VL + DeepSeek-OCR + dots.ocr fallback. Verify model availability on replicate.com before deposit.

## Mini-corpus — 8–10 pages total per model
| # | Source | Page count | Tests |
|---|--------|------------|-------|
| 1–2 | SEC EDGAR 10-K (one tech, one bank) | 2 | dense financial tables, multi-column, footnotes |
| 3 | DocILE invoice sample | 1 | invoice KV + line items |
| 4 | FUNSD form sample | 1 | form KV + relations |
| 5 | CORD receipt sample | 1 | receipt KV |
| 6 | FinTabNet table sample | 1 | table-only TEDS sanity check |
| 7 | IAM handwriting line | 1 | handwriting CER |
| 8 | OmniDocBench multilingual page | 1 | multilingual + layout |

Each page hand-picked, ground truth verified or hand-corrected. Per-page evaluation, not aggregated dataset metrics — at this volume, statistical aggregates aren't meaningful, but per-page side-by-side is the perfect workshop visual.

## Axes per page
- Layout fidelity (visual diff vs ground truth, manual 1–5 rubric)
- Table structure (TEDS where ground truth is available, manual rubric otherwise)
- KV extraction (F1 against hand-labeled fields)
- Raw OCR (CER on the body text)
- Latency per page on each runtime (informational only, not headline number)

## Cited from vendor sources (NOT measured)
- H200 throughput pages/sec/GPU
- VRAM peak at production batch size
- Latency p50/p95 in production deployment

## Reproducibility
- Pinned model revision SHAs in adapters
- Pinned page set in `corpus/` with ground-truth JSONs alongside PDFs
- Per-run parquet in `results/` with: stack_id, model_revision, page_id, axis, score, raw_output_path, run_timestamp_utc

## Cost target
Single Replicate $5 deposit covers all hosted runs. Local runs: free. Total out-of-pocket: $5.

## Workshop deliverables
1. `results/benchmark_matrix.parquet` — per-page scores across all stacks
2. `docs/findings.md` — opinionated narrative for the workshop
3. `docs/openshift_ai_deployment.md` — ServingRuntime / InferenceService YAML for the recommended stack(s), drawn from RHOAI docs (untested without a cluster)
4. Side-by-side rendered output gallery — the strongest visual for the demo
