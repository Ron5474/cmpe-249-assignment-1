"""Corrected detector construction shared by eval.py and latency.py.

Wraps ngdet's build_detector (ngdet itself stays unmodified, ADR-0002) and fixes
two behaviours found in review:

* Faster R-CNN: torchvision discards boxes below its internal
  roi_heads.score_thresh (0.05) before ngdet's own threshold is applied, so the
  requested score threshold is pushed down into the model.
* Grounding DINO: ngdet's phrase -> class fallback uses `s in name`, and since
  "" is a substring of every string, boxes with an empty phrase (every box whose
  best token score is below text_threshold) were all labelled the first class,
  "vehicle". Multi-term phrases such as "car truck" matched nothing and were
  dropped. `phrase_to_class` replaces it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "DeepDataMiningLearning"))

from DeepDataMiningLearning.ngdet.detectors.base import build_detector  # noqa: E402


def phrase_to_class(phrase, taxonomy) -> Optional[int]:
    """Map a Grounding DINO phrase to one unified class id, or None.

    Empty phrase -> None. Exact synonym -> its class. Otherwise every word must be
    a known synonym and all words must agree on one class ("car truck" ->
    vehicle); mixed-class ("person bicycle") or unknown words -> None.
    """
    if phrase is None:
        return None
    s = str(phrase).strip().lower()
    if not s:
        return None
    uid = taxonomy.name_to_id(s)
    if uid is not None:
        return uid
    ids = {taxonomy.name_to_id(w) for w in s.split()}
    if None in ids or len(ids) != 1:
        return None
    return ids.pop()


def effective_thresholds(det) -> dict:
    """Score floors actually in force for a built detector (for result metadata)."""
    out = {"score_thr": det.score_thr}
    model = getattr(getattr(det, "model", None), "_orig_mod", getattr(det, "model", None))
    if hasattr(model, "roi_heads"):
        out["roi_heads_score_thresh"] = model.roi_heads.score_thresh
    if hasattr(det, "text_threshold"):
        out["text_threshold"] = det.text_threshold
    return out


def build(spec: str, taxonomy, device: str = "cuda", score_thr: float = 0.001, accel: str = "fp32"):
    det = build_detector(spec, taxonomy, device=device, score_thr=score_thr, accel=accel)
    key = spec.split(":", 1)[0]
    if key == "torchvision":
        model = getattr(det.model, "_orig_mod", det.model)  # unwrap torch.compile
        model.roi_heads.score_thresh = min(model.roi_heads.score_thresh, det.score_thr)
    if key == "gdino":
        # bind the taxonomy, not det, so the detector is not kept alive by a reference cycle
        det._label_to_unified = lambda text, tax=det.taxonomy: phrase_to_class(text, tax)
    return det
