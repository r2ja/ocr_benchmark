# docintel-benchmark

Benchmark harness for the Azure Document Intelligence open-source replica workshop.
Target deployment: Red Hat OpenShift AI on an 8× H200 cluster.

## Stacks measured first-hand
- Docling + docling-serve — local RTX 3050 (PyTorch)
- Qwen3-VL family — OpenRouter (8B / 30B-A3B / 32B / 235B-A22B)

## Stacks documented from vendor publications (4 GB ceiling)
- PaddleOCR-VL-1.5 — needs Paddle Python orchestration; not feasible on Py3.13/Win
- dots.ocr-1.5 — vision-encoder compute buffer (3.65 GB) exceeds free VRAM
- DeepSeek-OCR-2 — Q8 weights alone (3 GB) plus compute buffer = OOM

All three trivially fit on a single H200 in BF16. See `docs/findings.md`.

Single API platform: OpenRouter. $5 deposit covers the full bench-off ~600x over.

## Axes measured first-hand
- Raw OCR (CER / WER) — OmniDocBench, IAM
- Layout mAP — DocLayNet
- Table TEDS — FinTabNet, PubTables-1M
- KV F1 — FUNSD, CORD, SROIE, DocILE
- Document QA — DUDE
- Real-world financials — curated SEC EDGAR 10-K pages

## Axes cited from vendor sources
- Throughput (pages/sec/GPU on H200)
- VRAM peak
- Latency p50/p95

## Layout
```
adapters/    # one Python module per stack (input PDF -> normalized JSON)
corpus/      # local copies of sampled datasets + curated SEC 10-K set
corpus_meta/ # mini-corpus manifest (page_ids, axes, gold-truth flags)
metrics/    # TEDS, KV-F1, CER/WER, mAP wrappers
results/    # parquet outputs per (stack, dataset) pair
scripts/    # CLI entrypoints (run_eval.py, build_corpus.py)
docs/       # benchmark spec, workshop deck source
```

## Setup
```
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env  # fill in keys you have
```
