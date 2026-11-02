#!/usr/bin/env bash
# Corrected reruns after review (see scripts/detectors.py): the six affected
# accuracy evaluations (Faster R-CNN at a true 0.001 floor, Grounding DINO with the
# fixed phrase mapping) and latency for all six models on the same machine.
# Results for the other models are left as they are; originals are in results/v1_original/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
log() { echo "+ $*" | tee -a results/logs/commands.log; "$@" 2>&1 | tee "results/logs/rerun_$(date +%H%M%S).log"; }

FRCNN="torchvision:fasterrcnn_resnet50_fpn_v2"
GDINO="gdino:IDEA-Research/grounding-dino-tiny"
ALL="$FRCNN yolo:yolo11s.pt yolo:yolo11x.pt hf_detr:PekingU/rtdetr_r50vd yolo:yolov8s-worldv2.pt $GDINO"

uv run scripts/test_fixes.py --gdino
log uv run scripts/eval.py --exp repro_coco80 --taxonomy coco80 --max-images 0 --models $FRCNN --datasets coco
log uv run scripts/eval.py --exp driving3 --taxonomy driving3 --max-images 0 --models $FRCNN $GDINO --datasets coco kitti
log uv run scripts/eval.py --exp driving3_bare --taxonomy driving3_bare --max-images 0 --models $GDINO --datasets kitti
log uv run scripts/latency.py --models $ALL --accel fp32 fp16
