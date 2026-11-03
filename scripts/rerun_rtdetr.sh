#!/usr/bin/env bash
# Third correction after review: the COCO-80 taxonomy had no aliases for the six
# VOC-style class names in the PekingU/rtdetr_r50vd checkpoint (motorbike, aeroplane,
# sofa, pottedplant, diningtable, tvmonitor), so those detections were discarded and
# the six classes scored AP 0. Only RT-DETR's COCO-80 reproduction is affected
# (driving3 already maps "motorbike" and ignores the other five), so only it is rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
log() { echo "+ $*" | tee -a results/logs/commands.log; "$@" 2>&1 | tee "results/logs/rerun_$(date +%H%M%S).log"; }

uv run scripts/test_fixes.py
log uv run scripts/eval.py --exp repro_coco80 --taxonomy coco80 --max-images 0 --models hf_detr:PekingU/rtdetr_r50vd --datasets coco
