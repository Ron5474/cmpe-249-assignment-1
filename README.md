# CMPE 249 HW: 2D Object Detection, Track A

Zero-shot evaluation of six COCO-pretrained detectors (4 families) on COCO val2017 and KITTI, using the course's `ngdet` harness (pinned submodule in `external/`).

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

`results/logs/commands.log` lists every command run; each `results/<exp>/` has `metrics.json`, `env.json` (GPU, versions, argv), `report.md`, and per-image predicted/GT boxes.

Evaluation protocol and its deviations from the tutorial defaults: `docs/adr/`.
