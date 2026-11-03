#!/usr/bin/env bash
# Second correction: Grounding DINO with token-to-class scoring (scripts/detectors.py).
# Reruns its three evaluations, the post-processor comparison, and latency for all six
# models on the same machine. Other models' accuracy results are kept as they are.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
log() { echo "+ $*" | tee -a results/logs/commands.log; "$@" 2>&1 | tee "results/logs/rerun_gdino_$(date +%H%M%S).log"; }

GDINO="gdino:IDEA-Research/grounding-dino-tiny"
ALL="torchvision:fasterrcnn_resnet50_fpn_v2 yolo:yolo11s.pt yolo:yolo11x.pt hf_detr:PekingU/rtdetr_r50vd yolo:yolov8s-worldv2.pt $GDINO"

uv run scripts/test_fixes.py --gdino
log uv run scripts/compare_gdino_postprocess.py --dataset coco --max-images 500
log uv run scripts/compare_gdino_postprocess.py --dataset kitti --max-images 500
log uv run scripts/eval.py --exp driving3 --taxonomy driving3 --max-images 0 --models $GDINO --datasets coco kitti
log uv run scripts/eval.py --exp driving3_bare --taxonomy driving3_bare --max-images 0 --models $GDINO --datasets kitti
log uv run scripts/latency.py --models $ALL --accel fp32 fp16
