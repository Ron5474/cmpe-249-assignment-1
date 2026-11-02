#!/usr/bin/env bash
# Download COCO val2017 and KITTI 2D object (training split) into $DATA_ROOT.
# Layout produced:
#   $DATA_ROOT/coco/val2017/*.jpg
#   $DATA_ROOT/coco/annotations/instances_val2017.json
#   $DATA_ROOT/kitti/training/{image_2,label_2}/
#   $DATA_ROOT/kitti/ImageSets/train.txt   (all 7481 ids, so ngdet evaluates the full split)
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"
mkdir -p "$DATA_ROOT/coco" "$DATA_ROOT/kitti"
cd "$DATA_ROOT"

fetch() {  # fetch <url> <dest>; resumable
  [ -f "$2" ] && return
  curl -fL -C - -o "$2.part" "$1"
  mv "$2.part" "$2"
}

# --- COCO val2017 (~1 GB images + 241 MB annotations) ---
if [ ! -d coco/val2017 ]; then
  fetch http://images.cocodataset.org/zips/val2017.zip coco/val2017.zip
  unzip -q coco/val2017.zip -d coco && rm coco/val2017.zip
fi
if [ ! -f coco/annotations/instances_val2017.json ]; then
  fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip coco/ann.zip
  unzip -q -o coco/ann.zip 'annotations/instances_val2017.json' -d coco && rm coco/ann.zip
fi

# --- KITTI 2D object (~12 GB left color images + 5 MB labels) ---
[ "${SKIP_KITTI:-0}" = 1 ] && { echo "COCO val images: $(ls coco/val2017 | wc -l)"; exit 0; }
KITTI_URL=https://s3.eu-central-1.amazonaws.com/avg-kitti
if [ ! -d kitti/training/image_2 ]; then
  fetch "$KITTI_URL/data_object_image_2.zip" kitti/img.zip
  unzip -q kitti/img.zip 'training/*' -d kitti && rm kitti/img.zip
fi
if [ ! -d kitti/training/label_2 ]; then
  fetch "$KITTI_URL/data_object_label_2.zip" kitti/lbl.zip
  unzip -q kitti/lbl.zip -d kitti && rm kitti/lbl.zip
fi
# ngdet's KITTI loader reads ImageSets/<split>.txt and would otherwise create a
# random 80/20 split; list every training id so the whole labelled split is used.
mkdir -p kitti/ImageSets
ls kitti/training/label_2 | sed 's/\.txt$//' | sort > kitti/ImageSets/train.txt

echo "COCO val images: $(ls coco/val2017 | wc -l)  KITTI ids: $(wc -l < kitti/ImageSets/train.txt)"
