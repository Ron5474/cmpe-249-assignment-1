#!/usr/bin/env bash
# Every experiment in the report, in order. Output + logs go to results/.
#   bash scripts/run_all.sh                          # report settings
#   REPRO_MAX=20 MAX_IMAGES=20 bash scripts/run_all.sh   # quick smoke test
# Reproduction check uses all 5000 COCO val images (needed to compare with published mAP).
# Cross-dataset runs use the first MAX_IMAGES images per dataset (default 2000) to keep
# Grounding DINO's runtime manageable; 0 = all.
set -euo pipefail
cd "$(dirname "$0")/.."
MAX="${MAX_IMAGES:-2000}"
REPRO_MAX="${REPRO_MAX:-0}"
mkdir -p results/logs
log() { echo "+ $*" | tee -a results/logs/commands.log; "$@" 2>&1 | tee "results/logs/run_$(date +%H%M%S).log"; }

CLOSED="torchvision:fasterrcnn_resnet50_fpn_v2 yolo:yolo11s.pt yolo:yolo11x.pt hf_detr:PekingU/rtdetr_r50vd"
OPEN="yolo:yolov8s-worldv2.pt gdino:IDEA-Research/grounding-dino-tiny"

# 1. Reproduction check: closed-set models, all 80 COCO classes, COCO val2017.
log uv run scripts/eval.py --exp repro_coco80 --taxonomy coco80 --max-images "$REPRO_MAX" \
    --models $CLOSED --datasets coco

# 2. Cross-dataset generalization: all six models, same 3 classes on every dataset.
log uv run scripts/eval.py --exp driving3 --taxonomy driving3 --max-images "$MAX" \
    --models $CLOSED $OPEN --datasets coco kitti

# 3. Prompt ablation: Grounding DINO with bare class names (curated terms are in driving3).
log uv run scripts/eval.py --exp driving3_bare --taxonomy driving3_bare --max-images "$MAX" \
    --models gdino:IDEA-Research/grounding-dino-tiny --datasets kitti

# 4. Speed / memory.
log uv run scripts/latency.py --models $CLOSED $OPEN --accel fp32 fp16
