# OpenShift AI deployment notes

DRAFT — to be finalized after benchmark runs select the recommended stack(s).

This doc is the workshop section that answers "OK, how does the deployment
actually look on RHOAI?". YAMLs here are drawn from RHOAI 2.x docs and per-model
vendor guides; **none have been run on a live cluster** in this study and that
caveat is noted explicitly in the deck.

## Target architecture (one-page diagram in the deck)

```
application
   |
   v
+--------------------+      +------------------------------+
|  KServe ingress    | <--> | InferenceService: <stack>    |
|  (RHOAI 2.x)       |      |   ServingRuntime: vllm-cuda  |
+--------------------+      |   ModelStorage: PVC (HF)     |
                            |   Resources: H200 x N        |
                            +------------------------------+
                                          ^
                                          |
                            +------------------------------+
                            | GPU operator + NFD           |
                            | Driver / toolkit / device-   |
                            | plugin daemon sets           |
                            +------------------------------+
```

## Per-stack deployment notes

### Qwen3-VL-32B-Instruct on RHOAI
- ServingRuntime: stock vLLM ServingRuntime CR (RHOAI 2.x ships one).
- InferenceService points at HF model `Qwen/Qwen3-VL-32B-Instruct`, pinned to a
  revision SHA.
- Resource request: 1× H200 (~64 GB BF16 weights + KV cache headroom).
- Tensor-parallel for the 235B-A22B MoE: TP=8, EP=8 across the full cluster.
- Image preprocessing: vLLM handles via the multimodal API; clients send images
  as base64 or URLs.

### Docling on RHOAI
- ServingRuntime: KServe InferenceService using the official `docling-serve`
  container image (Helm chart published).
- Resource request: 1× GPU sufficient (multi-component models, all small).
- Best deployed as the *layout + table* layer in front of a VLM, not standalone
  for KV.

### PaddleOCR-VL-1.5 on RHOAI
- ServingRuntime: vLLM (recipe published by the model vendor).
- Resource request: 1× GPU, can co-locate replicas for throughput.
- Strongest single-component option for invoice/receipt/form prebuilt verticals
  once fine-tuned.

### dots.ocr-1.5 on RHOAI
- ServingRuntime: vLLM-compatible.
- Resource request: 1× GPU (BF16 ~3.4 GB).

### DeepSeek-OCR-2 on RHOAI
- ServingRuntime: vLLM-compatible.
- Resource request: 1× GPU.
- Best fit for the high-throughput Markdown-extraction layer, not the KV layer.

## What's NOT covered here (workshop disclosure)

- Custom-model fine-tuning UI — Azure DI Studio has no clean OSS equivalent.
  The recommendation is **Label Studio + a Kubeflow Pipeline DAG** for the
  training loop, with the schema registry living in a Postgres/MinIO bucket.
  Specs for this go in `docs/finetune_pipeline.md` (TODO).
- Production observability — out of scope for the workshop.
- Multi-tenancy + cost attribution — out of scope.

## To finalize before workshop

- Pin exact RHOAI version of the target environment.
- Pin exact ServingRuntime CR YAMLs from that RHOAI version's docs.
- Add storage / image-pull secrets sections.
- Add a one-page "minimum viable production deploy" YAML for the recommended
  stack.
