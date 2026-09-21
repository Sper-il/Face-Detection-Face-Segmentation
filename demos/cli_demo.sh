#!/usr/bin/env bash
#
# CLI demo — runs the pipeline on every image in `demos/sample_images/`
# and writes overlays to `demos/output/`.
#
# Usage:
#   bash demos/cli_demo.sh                # uses synthetic images
#   bash demos/cli_demo.sh path/to/img.jpg
#
# Requires:
#   - src/pipeline/run.py
#   - data/output/ to be writable
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="${REPO_ROOT}/demos/output"
mkdir -p "${OUTPUT_DIR}"

if [ "$#" -gt 0 ]; then
    INPUT="$1"
else
    # Build a tiny synthetic sample directory on the fly.
    SAMPLE_DIR="${REPO_ROOT}/demos/sample_images"
    mkdir -p "${SAMPLE_DIR}"
    python -c "
import cv2, numpy as np, os
out = r'${SAMPLE_DIR}'
for i in range(3):
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 80, (180, 180, 180), -1)
    cv2.imwrite(os.path.join(out, f'sample_{i}.png'), img)
print('Created 3 sample images in', out)
"
    INPUT="${SAMPLE_DIR}"
fi

echo "[demo] Running pipeline on: ${INPUT}"
echo "[demo] Output directory    : ${OUTPUT_DIR}"

python -m src.pipeline.run \
    --image "${INPUT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cpu \
    --conf-threshold 0.7 \
    --nms-iou 0.5

echo "[demo] Done. Open ${OUTPUT_DIR}/vis/ to see overlays."
