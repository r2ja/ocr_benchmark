"""Diagnostic smoke test for PaddleOCRVL on Windows.

Enables faulthandler to catch native crashes (otherwise we get exit code 5
with no traceback). Logs each phase so we can isolate which step crashes.
"""
import faulthandler
import sys
import time
import traceback

faulthandler.enable()

print(f"python: {sys.version}", flush=True)


def step(label: str, fn):
    print(f"\n=== STEP: {label} ===", flush=True)
    t0 = time.time()
    try:
        out = fn()
        print(f"  OK in {time.time()-t0:.1f}s", flush=True)
        return out
    except Exception:
        print(f"  EXCEPTION in {time.time()-t0:.1f}s", flush=True)
        traceback.print_exc()
        sys.exit(1)


def import_paddle():
    import paddle
    print(f"  paddle {paddle.__version__}, cuda={paddle.is_compiled_with_cuda()}", flush=True)
    return paddle


def import_pipeline():
    from paddleocr import PaddleOCRVL
    return PaddleOCRVL


def init_pipeline(cls):
    # Try forcing the HuggingFace transformers backend to bypass paddle's
    # native safetensors loader (which crashes with access violation on Windows).
    return cls(vl_rec_backend="transformers")


def predict_one(pipe):
    return pipe.predict("corpus/funsd/form_01.png")


paddle = step("import paddle", import_paddle)
PaddleOCRVL = step("import PaddleOCRVL class", import_pipeline)
pipe = step("init PaddleOCRVL pipeline (loads weights)", lambda: init_pipeline(PaddleOCRVL))
out = step("predict on funsd/form_01.png", lambda: predict_one(pipe))

print("\n=== RESULT ===", flush=True)
print(f"  type: {type(out)}", flush=True)
if hasattr(out, "__len__"):
    print(f"  len: {len(out)}", flush=True)
if out:
    print(f"  first item type: {type(out[0])}", flush=True)
    print(f"  first item preview: {str(out[0])[:600]}", flush=True)
