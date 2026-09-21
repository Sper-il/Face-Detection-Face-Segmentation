# Deployment

This directory contains everything needed to take a trained model into
production.

## ONNX export

```bash
python -m deploy.export_onnx \
    --detector-weights models/retinaface_best.pth \
    --segmentor-weights models/unet_best.pth \
    --output-dir models/
```

Writes:

- `models/retinaface.onnx` — input `[1, 3, 640, 640]`, three outputs
  (`cls_logits`, `box_deltas`, `lmk_deltas`).
- `models/unet.onnx` — input `[1, 3, 512, 512]`, single output (`prob`).

By default the script also runs an `onnxruntime` smoke test on each exported
model to catch shape / dtype mismatches. ONNX opset is 17 (the highest stable
opset that covers all of ResNet / U-Net / interpolate ops used here).

## TensorRT (optional, GPU server)

```bash
trtexec \
    --onnx=models/retinaface.onnx \
    --saveEngine=models/retinaface.plan \
    --fp16 --workspace=4096
```

Same pattern for the U-Net segmentor (`models/unet.plan`).

## Inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("models/retinaface.onnx", providers=["CPUExecutionProvider"])
x = np.random.rand(1, 3, 640, 640).astype(np.float32)
cls, box, lmk = sess.run(None, {"input": x})
# → run your NMS + decode against the anchor generator
```

## FastAPI serving (optional)

`deploy/server.py` exposes a single `POST /segment` endpoint that takes a
multipart image and returns a JSON payload with bboxes + base64-encoded
masks + an overlay PNG. Run with:

```bash
uvicorn deploy.server:app --host 0.0.0.0 --port 8000
```

## Edge deployment

- **Jetson Nano / Orin:** TensorRT FP16 engine, batch=1.
- **CPU-only:** ONNX Runtime with `CPUExecutionProvider`.
- Always include a fallback that returns the original image if no face is
  found (the pipeline already does this — see `src/pipeline/orchestrator.py`).

## Privacy

Face data is sensitive. Keep all inference **on-prem**. Do not upload the
checkpoints or datasets to a third-party model registry without scrubbing
metadata.
