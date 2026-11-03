"""Corrected detector construction shared by eval.py and latency.py.

Wraps ngdet's build_detector (ngdet itself stays unmodified, ADR-0002) and fixes
two behaviours found in review:

* Faster R-CNN: torchvision discards boxes below its internal
  roi_heads.score_thresh (0.05) before ngdet's own threshold is applied, so the
  requested score threshold is pushed down into the model.
* Grounding DINO: ngdet derives each box's class from the decoded phrase, which
  (a) labelled every empty phrase "vehicle" because "" is a substring of every
  string, and (b) can only exist for tokens above text_threshold (0.25), which
  acts as a score floor. `token_postprocess` replaces phrase decoding with
  token-to-class scoring in the style of the official Grounding DINO COCO
  evaluator (PostProcessCocoGrounding): each class's probability is the mean of
  its tokens' sigmoid probabilities, and the top 300 query-class pairs are kept,
  so one query may predict several classes. Adaptation for our curated synonyms:
  token probabilities are averaged within each prompt term, then the maximum
  term score is taken within each unified class (pooled before top-k, so "car"
  and "truck" cannot produce duplicate vehicle boxes from one query).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "DeepDataMiningLearning"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from DeepDataMiningLearning.ngdet.detectors.base import Detection, build_detector  # noqa: E402

GDINO_TOP_K = 300  # as in the official COCO evaluator


def term_token_spans(caption: str, terms, tokenizer, input_ids) -> list:
    """Token indices of each prompt term inside the tokenized caption.

    Uses character offsets, so punctuation, [CLS]/[SEP] and padding never belong
    to a term. Raises if the re-tokenized caption differs from the model input.
    """
    enc = tokenizer(caption, return_offsets_mapping=True)
    if list(enc["input_ids"]) != list(input_ids):
        raise ValueError("caption tokenization does not match the model input")
    spans, pos = [], 0
    for term in terms:
        start = caption.index(term, pos)
        end = start + len(term)
        pos = end
        idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a >= start and b <= end]
        if not idx:
            raise ValueError(f"no tokens for term {term!r}")
        spans.append(idx)
    return spans


def class_probs(token_probs: torch.Tensor, spans, term_uids, num_classes: int) -> torch.Tensor:
    """[Q, T] token probabilities -> [Q, C] class probabilities.

    Mean over each term's tokens, then max over the terms of each class.
    """
    term_scores = torch.stack([token_probs[:, s].mean(-1) for s in spans], -1)  # [Q, terms]
    out = torch.zeros(token_probs.shape[0], num_classes, device=token_probs.device)
    for uid in range(num_classes):
        cols = [k for k, u in enumerate(term_uids) if u == uid]
        if cols:
            out[:, uid] = term_scores[:, cols].max(-1).values
    return out


def token_postprocess(det, image) -> Detection:
    """Grounding DINO predict() with token-to-class scoring instead of phrase decoding."""
    w, h = image.size
    inputs = det.processor(images=image, text=det.text_prompt, return_tensors="pt").to(det.device)
    with torch.no_grad():
        out = det.model(**inputs)
    ids = inputs["input_ids"][0].tolist()
    if det._gdino_spans is None:  # prompt is fixed per detector, so compute once
        det._gdino_spans = term_token_spans(det.text_prompt, det._term_list, det.processor.tokenizer, ids)
    probs = out.logits[0, :, : len(ids)].sigmoid()                              # [900, T]
    cls = class_probs(probs, det._gdino_spans, det._term_uids, det.taxonomy.num_classes)  # [900, C]
    C = cls.shape[1]
    scores, flat = cls.flatten().topk(min(GDINO_TOP_K, cls.numel()))
    q, labels = flat // C, flat % C
    cx, cy, bw, bh = out.pred_boxes[0, q].unbind(-1)                             # normalized cxcywh
    boxes = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
    keep = scores >= det.score_thr
    labels = labels[keep].cpu().numpy().astype(np.int64)
    return Detection(boxes=boxes[keep].cpu().numpy().astype(np.float32),
                     scores=scores[keep].cpu().numpy().astype(np.float32),
                     labels=labels, names=[det.taxonomy.classes[int(u)] for u in labels])


def effective_thresholds(det) -> dict:
    """Score floors actually in force for a built detector (for result metadata)."""
    out = {"score_thr": det.score_thr}
    model = getattr(getattr(det, "model", None), "_orig_mod", getattr(det, "model", None))
    if hasattr(model, "roi_heads"):
        out["roi_heads_score_thresh"] = model.roi_heads.score_thresh
    if getattr(det, "_gdino_token_scoring", False):
        out["gdino_scoring"] = f"token-to-class, top-{GDINO_TOP_K} query-class pairs, no text_threshold"
    return out


def build(spec: str, taxonomy, device: str = "cuda", score_thr: float = 0.001, accel: str = "fp32"):
    det = build_detector(spec, taxonomy, device=device, score_thr=score_thr, accel=accel)
    key = spec.split(":", 1)[0]
    if key == "torchvision":
        model = getattr(det.model, "_orig_mod", det.model)  # unwrap torch.compile
        model.roi_heads.score_thresh = min(model.roi_heads.score_thresh, det.score_thr)
    if key == "gdino":
        det._gdino_spans = None
        det._gdino_token_scoring = True
        # swap in a subclass rather than storing a bound method on det (which would be a reference cycle)
        det.__class__ = type("TokenScoredGroundingDino", (type(det),), {"predict": token_postprocess})
    return det
