#!/usr/bin/env bash
# H200 one-shot benchmark runbook (RunPod, 1× H200 SXM, ~$4/hr).
#
# Designed for the "hybrid" driving mode: you SSH into the pod and run
# phases interactively, pasting key outputs back to Claude. Each phase is a
# bash function so you can re-run any of them independently after a failure.
#
# Pod prerequisites:
#   - 1× H200 SXM
#   - PyTorch 2.x + CUDA 12.x base image (RunPod's "PyTorch 2.4.0 + CUDA 12.4"
#     template is fine; ships Python 3.10/3.11)
#   - Network volume mounted at /workspace/hf_cache (persists model weights
#     across pod restarts so we don't re-pay download cost)
#
# Usage:
#   chmod +x scripts/h200_runbook.sh
#   source scripts/h200_runbook.sh        # load all functions
#   phase1_install                          # one-time per pod
#   phase2_docling
#   phase3_dots
#   phase4_deepseek
#   phase5_paddle_vl
#   phase6_qwen_openrouter                  # cheap, runs from anywhere
#   phase7_sync_back                        # rsync results back to laptop
# OR run end-to-end:
#   bash scripts/h200_runbook.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"

VLLM_LOG="$REPO_ROOT/results/_vllm_server.log"
VLLM_PORT=8000
mkdir -p "$REPO_ROOT/results"

_log()   { printf '\n=== [runbook] %s ===\n' "$*"; }
_fatal() { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

# Wait until vLLM is responding to /v1/models, with a hard timeout.
_wait_for_vllm() {
  local max_wait="${1:-180}"
  local waited=0
  while ! curl -fsS "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if [[ $waited -ge $max_wait ]]; then
      _fatal "vLLM did not come up within ${max_wait}s — check $VLLM_LOG"
    fi
    printf '.'
  done
  echo
  _log "vLLM ready at localhost:${VLLM_PORT} (waited ${waited}s)"
}

_kill_vllm() {
  pkill -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -f 'vllm serve'       2>/dev/null || true
  sleep 5
  _log "vLLM stopped"
}

_serve() {
  local model_id="$1"; shift
  _log "Starting vLLM: $model_id"
  : > "$VLLM_LOG"
  nohup vllm serve "$model_id" \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --served-model-name "$model_id" \
    "$@" >> "$VLLM_LOG" 2>&1 &
  _wait_for_vllm 600
}

# ---------------------------------------------------------------------------
# Phase 1: install (one-time per pod)
# ---------------------------------------------------------------------------
phase1_install() {
  _log "Phase 1: install"
  cd "$REPO_ROOT"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  # vLLM is heavy (~3 min); pin to a recent, multimodal-capable release
  python -m pip install "vllm>=0.6.3"
  python -c "import vllm, torch; print('vllm', vllm.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  _log "Phase 1 done"
}

# ---------------------------------------------------------------------------
# Phase 2: Docling — local PyTorch, no vLLM. Validates the H200 latency
# baseline against the laptop run.
# ---------------------------------------------------------------------------
phase2_docling() {
  _log "Phase 2: Docling"
  cd "$REPO_ROOT"
  python -m scripts.run_eval --stack docling
  _log "Phase 2 done — see results/docling_run_*.parquet"
}

# ---------------------------------------------------------------------------
# Phase 3: dots.ocr-1.5 via vLLM
# ---------------------------------------------------------------------------
phase3_dots() {
  _log "Phase 3: dots.ocr"
  cd "$REPO_ROOT"
  _serve "rednote-hilab/dots.ocr"
  export DOTS_BASE_URL="http://localhost:${VLLM_PORT}/v1"
  export DOTS_API_KEY="EMPTY"
  python -m scripts.run_eval --stack dots
  _kill_vllm
  _log "Phase 3 done"
}

# ---------------------------------------------------------------------------
# Phase 4: DeepSeek-OCR via vLLM (requires --trust-remote-code)
# ---------------------------------------------------------------------------
phase4_deepseek() {
  _log "Phase 4: DeepSeek-OCR"
  cd "$REPO_ROOT"
  _serve "deepseek-ai/DeepSeek-OCR" --trust-remote-code
  export DEEPSEEK_BASE_URL="http://localhost:${VLLM_PORT}/v1"
  export DEEPSEEK_API_KEY="EMPTY"
  python -m scripts.run_eval --stack deepseek
  _kill_vllm
  _log "Phase 4 done"
}

# ---------------------------------------------------------------------------
# Phase 5: PaddleOCR-VL — two-tier orchestrator.
# This is the install-heavy phase. Budget 30 min.
#
# We use a separate Python 3.11 venv because paddlepaddle-gpu does not
# install on Python 3.13. Then paddleocr's PaddleOCRVL pipeline does the
# layout-detect + region-cropping and points its VLM tier at vLLM serving
# the same PaddleOCR-VL weights via OpenAI-compatible API.
# ---------------------------------------------------------------------------
phase5_paddle_vl() {
  _log "Phase 5: PaddleOCR-VL"
  cd "$REPO_ROOT"

  # 5a) bring up the VLM tier on vLLM
  _serve "PaddlePaddle/PaddleOCR-VL"

  # 5b) set up paddleocr orchestrator venv (one-time per pod)
  if [[ ! -d .venv-paddle ]]; then
    _log "Creating Python 3.11 venv for paddleocr orchestrator"
    python3.11 -m venv .venv-paddle || _fatal \
      "python3.11 not found on this image. Install via: apt-get install -y python3.11 python3.11-venv"
    .venv-paddle/bin/pip install --upgrade pip
    .venv-paddle/bin/pip install paddlepaddle-gpu paddleocr requests pypdfium2 pillow pandas pyarrow tqdm python-dotenv jiwer rapidfuzz huggingface_hub tenacity
  fi

  # 5c) run eval inside the paddle venv, pointed at vLLM
  export PADDLE_VL_MODE="orchestrator"
  export PADDLE_VLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
  export PADDLE_VLM_MODEL="PaddlePaddle/PaddleOCR-VL"
  .venv-paddle/bin/python -m scripts.run_eval --stack paddle-vl

  _kill_vllm
  _log "Phase 5 done"
}

# Optional sanity-check: confirm full-page direct VLM call returns garbage.
# Demonstrates why the orchestrator tier is load-bearing.
phase5_paddle_vl_fullpage_demo() {
  _log "Phase 5b: PaddleOCR-VL fullpage bypass (expected to produce garbage)"
  cd "$REPO_ROOT"
  _serve "PaddlePaddle/PaddleOCR-VL"
  export PADDLE_VL_MODE="fullpage"
  export PADDLE_VL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
  export PADDLE_VL_API_KEY="EMPTY"
  python -m scripts.run_eval --stack paddle-vl --pages funsd-form-01
  _kill_vllm
}

# ---------------------------------------------------------------------------
# Phase 6: Qwen3-VL size sweep. Stays on OpenRouter — cheaper than swapping
# 4 models through vLLM, and the Qwen latency story is already validated.
# Skip this on the H200 if the laptop already ran it.
# ---------------------------------------------------------------------------
phase6_qwen_openrouter() {
  _log "Phase 6: Qwen size sweep (OpenRouter)"
  cd "$REPO_ROOT"
  for stack in qwen-8b qwen-30b-a3b qwen-32b qwen-235b-a22b; do
    _log "  → $stack"
    python -m scripts.run_eval --stack "$stack"
  done
  _log "Phase 6 done"
}

# ---------------------------------------------------------------------------
# Phase 7: score everything, sync results back.
# ---------------------------------------------------------------------------
phase7_score_and_sync() {
  _log "Phase 7: score + sync"
  cd "$REPO_ROOT"
  python -m scripts.score_results
  _log "Results in $REPO_ROOT/results/ — rsync them back to your laptop:"
  echo "  rsync -avz pod:$REPO_ROOT/results/ ./results-h200/"
}

# ---------------------------------------------------------------------------
# Default: end-to-end. Only the three stacks blocked by the 4 GB laptop
# ceiling. Docling and the Qwen size sweep already ran on the laptop and on
# OpenRouter respectively — those are NOT re-run on H200 unless explicitly
# called (phase2_docling / phase6_qwen_openrouter), to save pod-time.
# ---------------------------------------------------------------------------
main() {
  phase1_install
  phase3_dots
  phase4_deepseek
  phase5_paddle_vl
  phase7_score_and_sync
}

# Only run main() if invoked as a script, not when sourced.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
