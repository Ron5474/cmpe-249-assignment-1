# CMPE 249 HW: 2D Object Detection, Track A

Zero-shot evaluation of six pretrained detector configurations (4 families: four closed-set COCO models and two open-vocabulary models) on COCO val2017 and KITTI, using the course's `ngdet` harness (pinned submodule in `external/`).

| Configuration | Family |
|---|---|
| `torchvision:fasterrcnn_resnet50_fpn_v2` | two-stage, anchor-based |
| `yolo:yolo11s.pt`, `yolo:yolo11x.pt` | one-stage, anchor-free |
| `hf_detr:PekingU/rtdetr_r50vd` | query-based (DETR) |
| `yolo:yolov8s-worldv2.pt`, `gdino:IDEA-Research/grounding-dino-tiny` | open-vocabulary |

## Run

```bash
git clone --recurse-submodules https://github.com/Ron5474/cmpe-249-assignment-1.git
cd cmpe-249-assignment-1
uv sync                                  # pinned env (torch 2.8 + CUDA 12.8)
bash scripts/download_data.sh            # COCO val2017 + KITTI -> ./data (or set DATA_ROOT)
bash scripts/run_all.sh                  # all experiments -> results/
```

`results/logs/commands.log` lists every command run; each `results/<exp>/` has `metrics.json`, an environment file per run (`env.json` for the first run, timestamped `env_<YYYYMMDD-HHMMSS>.json` for each rerun: GPU, versions, code version, argv), `report.md`, and per-image predicted/GT boxes.

Report: `report/report.pdf` (source `report/report.md`). Regenerate tables, figures and the PDF:

```bash
uv run scripts/analyze.py                                         # report/tables.md + report/figures/
uv run --with markdown --with weasyprint scripts/build_report.py  # report/report.pdf
```

Per-image box files (`results/**/boxes_*.jsonl.gz`, about 410 MB across all experiments) are not committed; `run_all.sh` regenerates them. The failure figure also needs five KITTI frames in `data/kitti_figs/` (chosen by `scripts/pick_failures.py`).

Corrections after review: `scripts/detectors.py` (Faster R-CNN's internal 0.001 floor, Grounding DINO token-to-class scoring) and `scripts/eval.py` (COCO-80 aliases for the six VOC-style class names in the RT-DETR checkpoint), tested by `scripts/test_fixes.py` and rerun with `scripts/rerun.sh`, `scripts/rerun_gdino.sh` and `scripts/rerun_rtdetr.sh`. Superseded results are kept in `results/v1_original/` (first run) and `results/v2_phrase_parser/` (first Grounding DINO fix).

Evaluation protocol and its deviations from the tutorial defaults: `docs/adr/`.
