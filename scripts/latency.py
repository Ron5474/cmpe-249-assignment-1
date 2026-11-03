"""Batch-1 latency, FPS and peak GPU memory per detector and precision.

Times ngdet's full `predict()` (preprocess + forward + postprocess + taxonomy fold)
on one fixed KITTI frame, after warmup, using the same corrected detector
construction as eval.py (scripts/detectors.py). ngdet.latency has no memory
measurement, hence this script.

  uv run scripts/latency.py --models yolo:yolo11x.pt --accel fp32 fp16
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "DeepDataMiningLearning"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from DeepDataMiningLearning.ngdet.taxonomy import Taxonomy  # noqa: E402

from detectors import build, effective_thresholds  # noqa: E402
from eval import RUN_INFO  # noqa: E402

DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--accel", nargs="+", default=["fp32", "fp16"])
    ap.add_argument("--image", default=str(DATA_ROOT / "kitti/training/image_2/000008.png"))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()

    tax = Taxonomy.from_preset("driving3")
    image = Image.open(a.image).convert("RGB")
    rows = []
    for spec in a.models:
        for accel in a.accel:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                det = build(spec, tax, device="cuda", score_thr=0.001, accel=accel)
            except Exception as e:  # unsupported model/precision pair
                print(f"[skip] {spec} {accel}: {e}")
                continue
            for _ in range(a.warmup):
                det.predict(image)
            times = []
            for _ in range(a.iters):
                torch.cuda.synchronize()
                t = time.perf_counter()
                det.predict(image)
                torch.cuda.synchronize()
                times.append(1000 * (time.perf_counter() - t))
            row = {
                "model": spec, "accel": accel,
                "median_ms": round(statistics.median(times), 2),
                "p90_ms": round(sorted(times)[int(0.9 * len(times))], 2),
                "fps": round(1000 / statistics.median(times), 1),
                "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "effective_thresholds": effective_thresholds(det),
            }
            print(row, flush=True)
            rows.append(row)
            del det
            gc.collect()  # HF detectors hold internal reference cycles; free before the next peak-memory reading

    out = REPO / "results" / "latency"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**RUN_INFO, "torch": torch.__version__,
               "image": a.image, "image_size": image.size, "warmup": a.warmup,
               "iters": a.iters, "rows": rows},
              open(out / "latency.json", "w"), indent=2)


if __name__ == "__main__":
    main()
