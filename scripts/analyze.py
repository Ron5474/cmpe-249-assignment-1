"""Turn results/ into the report's tables (report/tables.md) and figures (report/figures/).

  uv run scripts/analyze.py

Failure-case figure needs the five KITTI frames + label files in data/kitti_figs/.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "report"
FIG = OUT / "figures"

# Display names and fixed categorical colours (validated palette, slot order).
MODELS = {
    "torchvision:fasterrcnn_resnet50_fpn_v2": ("Faster R-CNN v2", "#2a78d6"),
    "yolo:yolo11s.pt": ("YOLO11s", "#eb6834"),
    "yolo:yolo11x.pt": ("YOLO11x", "#1baf7a"),
    "hf_detr:PekingU/rtdetr_r50vd": ("RT-DETR R50", "#eda100"),
    "yolo:yolov8s-worldv2.pt": ("YOLO-World v2-s", "#e87ba4"),
    "gdino:IDEA-Research/grounding-dino-tiny": ("Grounding DINO-T", "#008300"),
}
# Published COCO val2017 box mAP (sources cited in the report).
PUBLISHED = {
    "torchvision:fasterrcnn_resnet50_fpn_v2": 46.7,
    "yolo:yolo11s.pt": 47.0,
    "yolo:yolo11x.pt": 54.7,
    "hf_detr:PekingU/rtdetr_r50vd": 53.1,
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def load(exp):
    return {(m["_meta"]["model"], m["_meta"]["dataset"]): m
            for m in json.load(open(RES / exp / "metrics.json"))}


def pct(x):
    return f"{100 * x:.1f}"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def tables(repro, d3, bare, lat):
    L = []
    L.append("## Reproduction check (COCO val2017, 80 classes, 5000 images)\n")
    L.append("| Model | Ours mAP | Published mAP | Δ | AP50 | AP75 | AP_S | AP_M | AP_L | AR100 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, pub in PUBLISHED.items():
        m = repro[(k, "coco")]
        L.append(f"| {MODELS[k][0]} | {pct(m['mAP'])} | {pub:.1f} | {100 * m['mAP'] - pub:+.1f} | "
                 f"{pct(m['AP50'])} | {pct(m['AP75'])} | {pct(m['AP_small'])} | {pct(m['AP_medium'])} | "
                 f"{pct(m['AP_large'])} | {pct(m['AR_100'])} |")

    L.append("\n## Cross-dataset (driving3 taxonomy: vehicle / person / cyclist)\n")
    L.append("| Model | Dataset | mAP | AP50 | AP75 | AP_S | AP_M | AP_L | AR100 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in MODELS:
        for ds in ("coco", "kitti"):
            m = d3[(k, ds)]
            L.append(f"| {MODELS[k][0]} | {ds.upper()} | {pct(m['mAP'])} | {pct(m['AP50'])} | {pct(m['AP75'])} | "
                     f"{pct(m['AP_small'])} | {pct(m['AP_medium'])} | {pct(m['AP_large'])} | {pct(m['AR_100'])} |")

    L.append("\n## Per-class AP@[.5:.95], COCO → KITTI\n")
    L.append("| Model | vehicle | person | cyclist |")
    L.append("|---|---|---|---|")
    for k in MODELS:
        c, t = d3[(k, "coco")]["per_class"], d3[(k, "kitti")]["per_class"]
        L.append(f"| {MODELS[k][0]} | " + " | ".join(
            f"{pct(c[n])} → {pct(t[n])}" for n in ("vehicle", "person", "cyclist")) + " |")

    L.append("\n## Prompt ablation, Grounding DINO-T on KITTI\n")
    L.append("| Prompt | mAP | AR100 | vehicle | person | cyclist |")
    L.append("|---|---|---|---|---|---|")
    g = "gdino:IDEA-Research/grounding-dino-tiny"
    for name, m in (("curated", d3[(g, "kitti")]), ("bare", bare[(g, "kitti")])):
        pc = m["per_class"]
        L.append(f"| {name}: `{m['_meta']['prompt']}` | {pct(m['mAP'])} | {pct(m['AR_100'])} | "
                 f"{pct(pc['vehicle'])} | {pct(pc['person'])} | {pct(pc['cyclist'])} |")

    L.append(f"\n## Latency (batch 1, KITTI frame {lat['image_size'][0]}×{lat['image_size'][1]}, "
             f"{lat['warmup']} warmup + {lat['iters']} timed, {lat['gpu']})\n")
    L.append("| Model | Precision | Median ms | p90 ms | FPS | Peak GPU MB |")
    L.append("|---|---|---|---|---|---|")
    for r in lat["rows"]:
        note = " (adapter ignores fp16)" if r["model"].startswith("gdino") and r["accel"] == "fp16" else ""
        L.append(f"| {MODELS[r['model']][0]} | {r['accel']}{note} | {r['median_ms']} | {r['p90_ms']} | "
                 f"{r['fps']} | {r['peak_gpu_mem_mb']} |")
    v1 = RES / "v1_original/latency/latency.json"
    if v1.exists():
        old = {(r["model"], r["accel"]): r for r in json.load(open(v1))["rows"]}
        # only models whose detector code did not change between the two runs
        same_code = ("yolo:yolo11s.pt", "yolo:yolo11x.pt", "hf_detr:PekingU/rtdetr_r50vd", "yolo:yolov8s-worldv2.pt")
        L.append("\n## Host comparison: same GPU model, different host (fp32 median ms)\n")
        L.append("| Model | Instance 1 | Instance 2 | Ratio (2 / 1) |")
        L.append("|---|---|---|---|")
        for r in lat["rows"]:
            if r["accel"] == "fp32" and r["model"] in same_code and (r["model"], "fp32") in old:
                o = old[(r["model"], "fp32")]["median_ms"]
                L.append(f"| {MODELS[r['model']][0]} | {o} | {r['median_ms']} | {r['median_ms'] / o:.2f}× |")

    for ds in ("coco", "kitti"):
        f = RES / f"gdino_postprocess/{ds}.json"
        if f.exists():
            d = json.load(open(f))
            L.append(f"\n## Grounding DINO post-processor comparison ({ds.upper()}, first {d['num_images']} images, "
                     "same forward passes)\n")
            L.append("| Post-processor | mAP | AR100 | vehicle | person | cyclist |")
            L.append("|---|---|---|---|---|---|")
            for n, lbl in (("ngdet_phrase", "ngdet phrase mapping (empty → vehicle)"),
                           ("phrase_parser", "phrase parser, mixed/empty dropped"),
                           ("token_scoring", "token-to-class scoring (used)")):
                m = d[n]
                pc = m["per_class"]
                L.append(f"| {lbl} | {pct(m['mAP'])} | {pct(m.get('AR_100', 0))} | {pct(pc['vehicle'])} | "
                         f"{pct(pc['person'])} | {pct(pc['cyclist'])} |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")


def fig_speed_accuracy(d3, lat):
    ms = {r["model"]: r["median_ms"] for r in lat["rows"] if r["accel"] == "fp32"}
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    for k, (name, col) in MODELS.items():
        x = ms[k]
        yc, yk = 100 * d3[(k, "coco")]["mAP"], 100 * d3[(k, "kitti")]["mAP"]
        ax.plot([x, x], [yc, yk], color=col, linewidth=2, alpha=0.5, zorder=1)
        ax.scatter([x], [yc], s=64, facecolor="white", edgecolor=col, linewidth=2, zorder=3)
        ax.scatter([x], [yk], s=64, color=col, edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(name, (x, yc), xytext=(7, 2), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(9, 170)
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xticklabels(["10", "20", "50", "100"])
    ax.set_xlabel("Median latency per image, fp32, batch 1 (ms, log scale)", color=MUTED, fontsize=9)
    ax.set_ylabel("mAP@[.5:.95], driving3 (%)", color=MUTED, fontsize=9)
    ax.scatter([], [], s=50, facecolor="white", edgecolor=MUTED, linewidth=2, label="COCO val2017")
    ax.scatter([], [], s=50, color=MUTED, label="KITTI train (zero-shot)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.6, 0.47))
    style(ax)
    ax.set_title("Speed vs accuracy, and the drop from COCO to KITTI", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "speed_accuracy.png")
    plt.close(fig)


def fig_scale_retention(d3):
    bins = [("AP_small", "small"), ("AP_medium", "medium"), ("AP_large", "large")]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    n = len(MODELS)
    w = 0.8 / n
    for i, (k, (name, col)) in enumerate(MODELS.items()):
        vals = [100 * d3[(k, "kitti")][b] / d3[(k, "coco")][b] for b, _ in bins]
        xs = np.arange(3) + (i - (n - 1) / 2) * w
        ax.bar(xs, vals, width=w - 0.02, color=col, label=name, edgecolor="white", linewidth=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{lbl} objects" for _, lbl in bins], color=INK, fontsize=9)
    ax.set_ylabel("KITTI AP as % of COCO AP", color=MUTED, fontsize=9)
    ax.axhline(100, color=MUTED, linewidth=0.8, linestyle=":")
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    ax.set_ylim(0, 115)
    ax.set_title("Accuracy retained after domain shift, by object size", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "scale_retention.png")
    plt.close(fig)


def read_boxes(exp, model, ids):
    tag = model.replace(":", "_").replace("/", "_")
    out = {}
    with gzip.open(RES / exp / f"boxes_{tag}__kitti.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["image_id"] in ids:
                out[r["image_id"]] = r
    return out


def fig_failures():
    frames = [
        (3554, "(1) small distant objects"),
        (2920, "(2) parked bikes predicted as 'cyclist'"),
        (4139, "(3) crowded, occluded pedestrians"),
        (2526, "(4) parked bikes + DontCare 'false positives'"),
        (273, "(5) Grounding DINO-T recovers missed cars"),
    ]
    ids = {i for i, _ in frames}
    yolo = read_boxes("driving3", "yolo:yolo11x.pt", ids)
    gd = read_boxes("driving3", "gdino:IDEA-Research/grounding-dino-tiny", ids)
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 3.9), dpi=220)
    axes = axes.ravel()
    axes[-1].axis("off")
    for ax, (i, title) in zip(axes, frames):
        img = Image.open(REPO / f"data/kitti_figs/{i:06d}.png")
        ax.imshow(img)
        for line in open(REPO / f"data/kitti_figs/{i:06d}.txt"):
            p = line.split()
            if p[0] == "DontCare":
                x1, y1, x2, y2 = map(float, p[4:8])
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True, facecolor="#ffffff40",
                                               edgecolor="#ffffff", linestyle="--", linewidth=0.7))
        r = yolo[i]
        for b in r["gt_boxes"]:
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                           edgecolor="#1baf7a", linewidth=1.0))
        dets = [(r, "#2a78d6")] + ([(gd[i], "#eb6834")] if i == 273 else [])
        for rr, col in dets:
            for b, s in zip(rr["pred_boxes"], rr["pred_scores"]):
                if s >= 0.3:
                    ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                                   edgecolor=col, linewidth=0.8, linestyle="-"))
        ax.set_title(f"{i:06d} {title}", fontsize=6.5, color=INK, loc="left", pad=2)
        ax.axis("off")
    handles = [patches.Patch(edgecolor="#1baf7a", fill=False, label="ground truth (driving3)"),
               patches.Patch(edgecolor="#2a78d6", fill=False, label="YOLO11x, score ≥ 0.3"),
               patches.Patch(edgecolor="#eb6834", fill=False, label="Grounding DINO-T, score ≥ 0.3"),
               patches.Patch(edgecolor="#888888", facecolor="#dddddd", linestyle="--", label="DontCare region")]
    axes[-1].legend(handles=handles, loc="center", frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.4)
    fig.savefig(FIG / "failures.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    repro, d3, bare = load("repro_coco80"), load("driving3"), load("driving3_bare")
    lat = json.load(open(RES / "latency/latency.json"))
    tables(repro, d3, bare, lat)
    fig_speed_accuracy(d3, lat)
    fig_scale_retention(d3)
    if (REPO / "data/kitti_figs").exists():
        fig_failures()
    print(f"wrote {OUT / 'tables.md'} and {sorted(p.name for p in FIG.iterdir())}")


if __name__ == "__main__":
    main()
