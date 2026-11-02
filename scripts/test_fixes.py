"""Checks for the corrections in scripts/detectors.py. Run before spending GPU time:

  uv run scripts/test_fixes.py            # phrase mapping + Faster R-CNN floor (CPU ok)
  uv run scripts/test_fixes.py --gdino    # also runs Grounding DINO on a few COCO images
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "DeepDataMiningLearning"))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from DeepDataMiningLearning.ngdet.taxonomy import TAXONOMY_PRESETS, Taxonomy  # noqa: E402
from detectors import build, phrase_to_class  # noqa: E402

DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "data"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_phrase_mapping():
    curated = Taxonomy.from_preset("driving3")
    bare = Taxonomy.from_synonyms("driving3_bare", TAXONOMY_PRESETS["driving3"])
    V, P, C = 0, 1, 2
    cases = {
        "": None, "   ": None, None: None,              # empty phrase: the original bug
        "car": V, "Truck ": V, "pedestrian": P, "bicycle": C, "cyclist": C,
        "car truck": V, "bus van car": V,                # same-class multi-term
        "person bicycle": None, "car pedestrian": None,  # mixed-class
        "dog": None, "car dog": None, "##ist": None,     # unknown words
    }
    for phrase, want in cases.items():
        got = phrase_to_class(phrase, curated)
        assert got == want, f"curated {phrase!r}: got {got}, want {want}"
    for phrase, want in {"": None, "vehicle": V, "person": P, "cyclist": C, "vehicle person": None}.items():
        got = phrase_to_class(phrase, bare)
        assert got == want, f"bare {phrase!r}: got {got}, want {want}"
    print(f"phrase mapping: {len(cases) + 5} cases ok")


def coco_images(n):
    anns = json.load(open(DATA_ROOT / "coco/annotations/instances_val2017.json"))
    names = sorted(i["file_name"] for i in anns["images"])[:n]
    return [Image.open(DATA_ROOT / "coco/val2017" / f).convert("RGB") for f in names]


def test_frcnn_floor():
    tax = Taxonomy.from_preset("driving3")
    det = build("torchvision:fasterrcnn_resnet50_fpn_v2", tax, device=DEVICE, score_thr=0.001)
    assert det.model.roi_heads.score_thresh == 0.001, det.model.roi_heads.score_thresh
    scores = torch.cat([torch.as_tensor(det.predict(im).scores) for im in coco_images(5)])
    assert scores.numel() and scores.min() < 0.05, f"no box below old 0.05 floor (min {scores.min():.4f})"
    print(f"faster r-cnn: score_thresh=0.001, {scores.numel()} boxes, min score {scores.min():.4f}")


def test_gdino(n=8):
    for preset in ("driving3", "driving3_bare"):
        tax = (Taxonomy.from_preset("driving3") if preset == "driving3"
               else Taxonomy.from_synonyms(preset, TAXONOMY_PRESETS["driving3"]))
        det = build("gdino:IDEA-Research/grounding-dino-tiny", tax, device=DEVICE, score_thr=0.001)
        seen = Counter()
        orig = det._label_to_unified
        det._label_to_unified = lambda t: (seen.update([str(t).strip().lower() or "<empty>"]), orig(t))[1]
        dets = [det.predict(im) for im in coco_images(n)]
        labels = Counter(int(l) for d in dets for l in d.labels)
        lo = sum(int((d.scores < 0.25).sum()) for d in dets)
        assert lo == 0, f"{lo} kept boxes below text_threshold 0.25 (empty phrases leaking through)"
        assert labels[1] > 0, "no person boxes on COCO images"
        dropped = {p: c for p, c in seen.items() if orig(p if p != "<empty>" else "") is None}
        print(f"gdino {preset}: kept per class {dict(sorted(labels.items()))}, "
              f"dropped phrases {dict(Counter(dropped).most_common(6))}")
        del det


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdino", action="store_true")
    a = ap.parse_args()
    test_phrase_mapping()
    test_frcnn_floor()
    if a.gdino:
        test_gdino()
    print("all checks passed")
