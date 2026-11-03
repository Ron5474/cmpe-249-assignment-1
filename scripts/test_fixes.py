"""Checks for the corrections in scripts/detectors.py. Run before spending GPU time:

  uv run scripts/test_fixes.py            # Faster R-CNN floor, COCO-80 aliases, token scoring checks (CPU ok)
  uv run scripts/test_fixes.py --gdino    # also runs Grounding DINO on a few COCO images
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import weakref
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "DeepDataMiningLearning"))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from DeepDataMiningLearning.ngdet.taxonomy import TAXONOMY_PRESETS, Taxonomy  # noqa: E402
from detectors import build, class_probs  # noqa: E402

DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "data"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def taxonomies():
    return {"driving3": Taxonomy.from_preset("driving3"),
            "driving3_bare": Taxonomy.from_synonyms("driving3_bare", TAXONOMY_PRESETS["driving3"])}


def test_class_probs():
    # tokens: 0,1 -> term "car"(vehicle); 2 -> "truck"(vehicle); 3 -> "person"(person)
    spans, uids = [[0, 1], [2], [3]], [0, 0, 1]
    p = torch.tensor([[0.2, 0.4, 0.9, 0.1],    # vehicle = max(mean(.2,.4)=.3, .9) = .9
                      [0.6, 0.6, 0.0, 0.7]])   # vehicle = max(.6, 0) = .6, person = .7
    got = class_probs(p, spans, uids, 3)
    want = torch.tensor([[0.9, 0.1, 0.0], [0.6, 0.7, 0.0]])
    assert torch.allclose(got, want), got
    print("class_probs: mean within term, max within class, ok")


def coco_images(n):
    anns = json.load(open(DATA_ROOT / "coco/annotations/instances_val2017.json"))
    names = sorted(i["file_name"] for i in anns["images"])[:n]
    return [Image.open(DATA_ROOT / "coco/val2017" / f).convert("RGB") for f in names]


def test_frcnn_floor():
    det = build("torchvision:fasterrcnn_resnet50_fpn_v2", taxonomies()["driving3"], device=DEVICE, score_thr=0.001)
    assert det.model.roi_heads.score_thresh == 0.001, det.model.roi_heads.score_thresh
    scores = torch.cat([torch.as_tensor(det.predict(im).scores) for im in coco_images(5)])
    assert scores.numel() and scores.min() < 0.05, f"no box below old 0.05 floor (min {scores.min():.4f})"
    print(f"faster r-cnn: score_thresh=0.001, {scores.numel()} boxes, min score {scores.min():.4f}")


def test_coco80_aliases():
    # The PekingU/rtdetr_r50vd checkpoint names six COCO classes VOC-style; the
    # coco80 taxonomy must map every checkpoint name or those detections vanish.
    from transformers import AutoConfig
    from eval import make_taxonomy
    tax = make_taxonomy("coco80")
    for alias, coco in [("motorbike", "motorcycle"), ("aeroplane", "airplane"), ("sofa", "couch"),
                        ("pottedplant", "potted plant"), ("diningtable", "dining table"), ("tvmonitor", "tv")]:
        assert tax.name_to_id(alias) is not None and tax.name_to_id(alias) == tax.name_to_id(coco), alias
    names = list(AutoConfig.from_pretrained("PekingU/rtdetr_r50vd").id2label.values())
    unmapped = [n for n in names if tax.name_to_id(n) is None]
    assert len(names) == 80 and not unmapped, unmapped
    print("coco80 taxonomy: all 80 RT-DETR checkpoint names map (6 via aliases)")


def test_gdino(n=6):
    for name, tax in taxonomies().items():
        det = build("gdino:IDEA-Research/grounding-dino-tiny", tax, device=DEVICE, score_thr=0.001)
        images = coco_images(n)
        dets = [det.predict(im) for im in images]

        # token spans decode back to exactly the prompt terms, with no punctuation/special tokens
        ids = det.processor(images=images[0], text=det.text_prompt, return_tensors="pt")["input_ids"][0].tolist()
        decoded = [det.processor.tokenizer.decode([ids[i] for i in span]).replace(" ", "") for span in det._gdino_spans]
        assert decoded == [t.replace(" ", "") for t in det._term_list], (decoded, det._term_list)

        # top-k over query-class pairs: <= 300 boxes, some query predicts >1 class, scores below 0.25 kept
        assert all(len(d) <= 300 for d in dets)
        assert any(len(d) == 300 for d in dets), "expected full top-300 at score floor 0.001"
        assert min(float(d.scores.min()) for d in dets if len(d)) < 0.25
        multi = sum(len({tuple(b) for b in d.boxes}) < len(d) for d in dets)
        assert multi > 0, "no query produced more than one class"

        # boxes are absolute xyxy within (slightly beyond) the image, matching HF's own box conversion
        im, d = images[0], dets[0]
        inputs = det.processor(images=im, text=det.text_prompt, return_tensors="pt").to(det.device)
        with torch.no_grad():
            out = det.model(**inputs)
        ref = det.processor.post_process_grounded_object_detection(
            out, threshold=0.0, text_threshold=0.0, target_sizes=[im.size[::-1]])[0]
        top_ref = ref["boxes"][ref["scores"].argmax()].cpu()
        assert any(torch.allclose(torch.as_tensor(b), top_ref, atol=0.5) for b in d.boxes[:10]), "box conversion mismatch"

        n_cls = {int(u) for d in dets for u in d.labels}
        print(f"gdino {name}: {sum(len(d) for d in dets)} boxes on {n} images, classes {sorted(n_cls)}, "
              f"min score {min(float(d.scores.min()) for d in dets):.4f}, {multi} images with multi-class queries; "
              f"spans {decoded}")

        r = weakref.ref(det)
        del det, dets
        gc.collect()
        assert r() is None, "detector not freed"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdino", action="store_true")
    a = ap.parse_args()
    test_class_probs()
    test_coco80_aliases()
    test_frcnn_floor()
    if a.gdino:
        test_gdino()
    print("all checks passed")
