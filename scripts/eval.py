"""Evaluate detectors on datasets with the stock ngdet harness.

Mirrors ngdet.run_eval.evaluate_one, plus three things the harness lacks:
  * a `coco80` taxonomy (identity over the 80 COCO names) for the reproduction check
  * a `driving3_bare` taxonomy (driving3 with bare class-name prompts) for the
    open-vocab prompt ablation
  * every per-image prediction (unrounded) + GT saved for failure analysis and
    rescoring, and peak GPU memory / wall-clock throughput per run
Detectors are built through scripts/detectors.py, which corrects Faster R-CNN's
internal score floor and Grounding DINO's phrase -> class mapping.

Usage (see scripts/run_all.sh for the exact commands used in the report):
  uv run scripts/eval.py --exp driving3 --taxonomy driving3 \
      --models yolo:yolo11x.pt --datasets coco kitti
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "DeepDataMiningLearning"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from DeepDataMiningLearning.ngdet import taxonomy as ngtax  # noqa: E402
from DeepDataMiningLearning.ngdet.datasets import EvalDataset  # noqa: E402
from DeepDataMiningLearning.ngdet.evaluator import COCOUnifiedEvaluator  # noqa: E402
from DeepDataMiningLearning.ngdet.report import plot_heatmap, write_markdown_report  # noqa: E402

from detectors import build, effective_thresholds  # noqa: E402

DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "data"))
COCO_ANN = DATA_ROOT / "coco/annotations/instances_val2017.json"
ROOTS = {"coco": DATA_ROOT / "coco/val2017", "kitti": DATA_ROOT / "kitti"}


def make_taxonomy(name: str) -> ngtax.Taxonomy:
    if name == "coco80":
        cats = json.load(open(COCO_ANN))["categories"]
        return ngtax.Taxonomy.from_synonyms("coco80", {c["name"]: [] for c in cats})
    if name == "driving3_bare":
        # Same classes/synonyms as driving3, but no curated prompt terms, so
        # open-vocab models are prompted with "vehicle . person . cyclist".
        return ngtax.Taxonomy.from_synonyms("driving3_bare",
                                            ngtax.TAXONOMY_PRESETS["driving3"])
    return ngtax.Taxonomy.from_preset(name)


def code_version() -> str:
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD") + ("-dirty" if run("status", "--porcelain", "scripts") else "")


RUN_INFO = {
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "code_version": code_version(),
    "run_started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}


def evaluate_one(spec: str, ds_name: str, tax, score_thr: float, max_images,
                 out_dir: Path) -> dict:
    print(f"\n{'=' * 70}\n[{spec}] x [{ds_name}] taxonomy={tax.name}\n{'=' * 70}", flush=True)
    ds_kwargs = {"ann_file": str(COCO_ANN)} if ds_name == "coco" else {}
    dataset = EvalDataset(ds_name, str(ROOTS[ds_name]), tax, max_images=max_images, **ds_kwargs)

    torch.cuda.reset_peak_memory_stats()
    detector = build(spec, tax, device="cuda", score_thr=score_thr)
    evaluator = COCOUnifiedEvaluator(tax)
    label = spec.split("@")[0]
    tag = label.replace(":", "_").replace("/", "_")

    t0 = time.time()
    infer_s = 0.0
    boxes_path = out_dir / f"boxes_{tag}__{ds_name}.jsonl.gz"
    with gzip.open(boxes_path, "wt") as fbox:
        for i, sample in enumerate(dataset):
            torch.cuda.synchronize()
            ti = time.perf_counter()
            det = detector.predict(sample.image)
            torch.cuda.synchronize()
            infer_s += time.perf_counter() - ti
            evaluator.add(sample, det)
            fbox.write(json.dumps({
                "image_id": int(sample.image_id),
                "pred_boxes": np.asarray(det.boxes, np.float64).tolist(),
                "pred_scores": np.asarray(det.scores, np.float64).tolist(),
                "pred_labels": np.asarray(det.labels).tolist(),
                "gt_boxes": np.asarray(sample.gt_boxes, np.float64).tolist(),
                "gt_labels": sample.gt_labels.tolist(),
            }) + "\n")
            if i % 500 == 0:
                print(f"  {i}/{len(dataset)}", flush=True)

    metrics = evaluator.summarize()
    n = len(dataset)
    metrics["_meta"] = {
        "model": label, "dataset": ds_name, "taxonomy": tax.name,
        "num_images": n, "open_vocab": detector.is_open_vocab,
        "score_thr": detector.score_thr,
        "effective_thresholds": effective_thresholds(detector),
        "prompt": getattr(detector, "text_prompt", None),
        "seconds": round(time.time() - t0, 1),
        "mean_predict_ms": round(1000 * infer_s / max(n, 1), 2),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        **RUN_INFO,
    }
    del detector
    gc.collect()  # HF detectors hold internal reference cycles; free before the next peak-memory reading
    torch.cuda.empty_cache()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="results/<exp>/ output folder")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True, choices=list(ROOTS))
    ap.add_argument("--taxonomy", default="driving3")
    ap.add_argument("--score-thr", type=float, default=0.001)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    out_dir = REPO / "results" / a.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    tax = make_taxonomy(a.taxonomy)
    metrics_path = out_dir / "metrics.json"
    all_metrics = json.load(open(metrics_path)) if metrics_path.exists() else []

    for spec in a.models:
        for ds in a.datasets:
            m = evaluate_one(spec, ds, tax, a.score_thr, a.max_images or None, out_dir)
            # replace any earlier result for the same (model, dataset)
            all_metrics = [x for x in all_metrics
                           if (x["_meta"]["model"], x["_meta"]["dataset"]) != (m["_meta"]["model"], ds)]
            all_metrics.append(m)
            json.dump(all_metrics, open(metrics_path, "w"), indent=2)  # save after every pair

    env = {
        "argv": sys.argv, **RUN_INFO,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(), "platform": platform.platform(),
    }
    # one env file per invocation, so reruns don't overwrite earlier provenance
    json.dump(env, open(out_dir / f"env_{time.strftime('%Y%m%d-%H%M%S')}.json", "w"), indent=2)
    write_markdown_report(all_metrics, str(out_dir / "report.md"), tax)
    plot_heatmap(all_metrics, str(out_dir / "heatmap_mAP.png"))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
