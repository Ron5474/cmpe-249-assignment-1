"""Diagnostic: score the same Grounding DINO forward passes with three post-processors.

  uv run scripts/compare_gdino_postprocess.py --dataset coco --max-images 1000

  ngdet_phrase   ngdet as shipped: phrases above text_threshold=0.25, empty phrase -> "vehicle"
  phrase_parser  the first correction (code 0f9be9f): empty and mixed-class phrases dropped
  token_scoring  current scripts/detectors.py: token-to-class scores, top-300 pairs, no text gate
Writes results/gdino_postprocess/<dataset>.json. Diagnostic only, not a benchmark result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "DeepDataMiningLearning"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from DeepDataMiningLearning.ngdet.datasets import EvalDataset  # noqa: E402
from DeepDataMiningLearning.ngdet.detectors.base import Detection  # noqa: E402
from DeepDataMiningLearning.ngdet.detectors.grounding_dino import HFZeroShotDetector  # noqa: E402
from DeepDataMiningLearning.ngdet.evaluator import COCOUnifiedEvaluator  # noqa: E402
from DeepDataMiningLearning.ngdet.taxonomy import Taxonomy  # noqa: E402
from detectors import GDINO_TOP_K, build, class_probs, term_token_spans  # noqa: E402

DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "data"))


def phrase_parser(phrase, taxonomy):
    """The first correction's mapping (code 0f9be9f), kept here only for this comparison."""
    s = str(phrase).strip().lower() if phrase is not None else ""
    if not s:
        return None
    uid = taxonomy.name_to_id(s)
    if uid is not None:
        return uid
    ids = {taxonomy.name_to_id(w) for w in s.split()}
    return ids.pop() if None not in ids and len(ids) == 1 else None


def from_phrases(res, mapper, score_thr):
    keep = [(b, s, mapper(l)) for b, s, l in zip(res["boxes"].tolist(), res["scores"].tolist(), res["text_labels"])
            if s >= score_thr]
    keep = [(b, s, u) for b, s, u in keep if u is not None]
    if not keep:
        return Detection()
    b, s, u = zip(*keep)
    return Detection(boxes=np.asarray(b, np.float32), scores=np.asarray(s, np.float32), labels=np.asarray(u, np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="coco", choices=["coco", "kitti"])
    ap.add_argument("--max-images", type=int, default=1000)
    a = ap.parse_args()

    tax = Taxonomy.from_preset("driving3")
    det = build("gdino:IDEA-Research/grounding-dino-tiny", tax, device="cuda", score_thr=0.001)
    kw = {"ann_file": str(DATA_ROOT / "coco/annotations/instances_val2017.json")} if a.dataset == "coco" else {}
    root = DATA_ROOT / ("coco/val2017" if a.dataset == "coco" else "kitti")
    ds = EvalDataset(a.dataset, str(root), tax, max_images=a.max_images, **kw)
    names = ["ngdet_phrase", "phrase_parser", "token_scoring"]
    evals = {n: COCOUnifiedEvaluator(tax) for n in names}
    stock_map = lambda t: HFZeroShotDetector._label_to_unified(det, t)  # ngdet's original mapping

    for sample in ds:
        im = sample.image
        inputs = det.processor(images=im, text=det.text_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = det.model(**inputs)
        res = det.processor.post_process_grounded_object_detection(
            out, threshold=0.0, text_threshold=0.25, target_sizes=[im.size[::-1]])[0]
        evals["ngdet_phrase"].add(sample, from_phrases(res, stock_map, det.score_thr))
        evals["phrase_parser"].add(sample, from_phrases(res, lambda t: phrase_parser(t, tax), det.score_thr))

        ids = inputs["input_ids"][0].tolist()
        if det._gdino_spans is None:
            det._gdino_spans = term_token_spans(det.text_prompt, det._term_list, det.processor.tokenizer, ids)
        cls = class_probs(out.logits[0, :, : len(ids)].sigmoid(), det._gdino_spans, det._term_uids, tax.num_classes)
        w, h = im.size
        scores, flat = cls.flatten().topk(GDINO_TOP_K)
        q, labels = flat // tax.num_classes, flat % tax.num_classes
        cx, cy, bw, bh = out.pred_boxes[0, q].unbind(-1)
        boxes = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
        k = scores >= det.score_thr
        evals["token_scoring"].add(sample, Detection(boxes=boxes[k].cpu().numpy(), scores=scores[k].cpu().numpy(),
                                                     labels=labels[k].cpu().numpy().astype(np.int64)))

    out = {}
    for n in names:
        m = evals[n].summarize(verbose=False)
        out[n] = {k: m[k] for k in ("mAP", "AP50", "AP75", "AR_100", "per_class") if k in m}
        print(f"{n:14s} mAP={m['mAP']:.3f} AR100={m.get('AR_100', 0):.3f} " +
              " ".join(f"{c}={v:.3f}" for c, v in m["per_class"].items()))
    dest = REPO / "results/gdino_postprocess"
    dest.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": a.dataset, "num_images": len(ds), **out}, open(dest / f"{a.dataset}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
