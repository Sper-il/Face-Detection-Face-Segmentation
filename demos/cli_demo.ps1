# CLI demo (PowerShell) — runs the pipeline on every image in
# `demos/sample_images/` (created on the fly) and writes overlays
# to `demos/output/`.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File demos/cli_demo.ps1
#   powershell -ExecutionPolicy Bypass -File demos/cli_demo.ps1 -Input path\to\img.jpg
[CmdletBinding()]
param(
    [string]$Input
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$OutputDir = Join-Path $RepoRoot "demos\output"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not $Input) {
    $SampleDir = Join-Path $RepoRoot "demos\sample_images"
    New-Item -ItemType Directory -Force -Path $SampleDir | Out-Null
    Write-Host "[demo] Synthesising sample images into $SampleDir"
    python -c @"
import cv2, numpy as np, os
out = r'$SampleDir'
for i in range(3):
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 80, (180, 180, 180), -1)
    cv2.imwrite(os.path.join(out, f'sample_{i}.png'), img)
print('Created 3 sample images in', out)
"@
    $Input = $SampleDir
}

Write-Host "[demo] Running pipeline on: $Input"
Write-Host "[demo] Output directory   : $OutputDir"

python -m src.pipeline.run `
    --image "$Input" `
    --output-dir "$OutputDir" `
    --device cpu `
    --conf-threshold 0.7 `
    --nms-iou 0.5

Write-Host "[demo] Done. Open $OutputDir\vis to see overlays."
