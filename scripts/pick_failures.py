"""Rank KITTI frames by failure type (YOLO11x, score >= 0.3, IoU 0.5) to pick the failure-analysis figure frames.

  uv run scripts/pick_failures.py
"""
import gzip,json,numpy as np
def load(p): return [json.loads(l) for l in gzip.open(p,"rt")]
def iou(a,b):
    a=np.array(a).reshape(-1,4);b=np.array(b).reshape(-1,4)
    if not len(a) or not len(b): return np.zeros((len(a),len(b)))
    x1=np.maximum(a[:,None,0],b[None,:,0]);y1=np.maximum(a[:,None,1],b[None,:,1]);x2=np.minimum(a[:,None,2],b[None,:,2]);y2=np.minimum(a[:,None,3],b[None,:,3])
    inter=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None)
    ar=lambda z:(z[:,2]-z[:,0])*(z[:,3]-z[:,1])
    return inter/(ar(a)[:,None]+ar(b)[None,:]-inter+1e-9)
from pathlib import Path
R=str(Path(__file__).resolve().parents[1] / "results/driving3") + "/"
y=load(R+"boxes_yolo_yolo11x.pt__kitti.jsonl.gz"); g=load(R+"boxes_gdino_IDEA-Research_grounding-dino-tiny__kitti.jsonl.gz")
gd={r["image_id"]:r for r in g}
rows=[]
for r in y:
    gt=np.array(r["gt_boxes"]).reshape(-1,4); gl=np.array(r["gt_labels"])
    if not len(gt): continue
    keep=np.array(r["pred_scores"])>=0.3
    pb=np.array(r["pred_boxes"]).reshape(-1,4)[keep]
    M=iou(gt,pb); hit=(M>=0.5).any(1) if M.size else np.zeros(len(gt),bool)
    h=gt[:,3]-gt[:,1]
    fp=int((~(M>=0.5).any(0)).sum()) if M.size else len(pb)
    rg=gd[r["image_id"]]; kg=np.array(rg["pred_scores"])>=0.3
    Mg=iou(gt,np.array(rg["pred_boxes"]).reshape(-1,4)[kg]); hitg=(Mg>=0.5).any(1) if Mg.size else np.zeros(len(gt),bool)
    rows.append(dict(id=r["image_id"],n=len(gt),miss_small=int(((~hit)&(h<25)).sum()),cyc=int((gl==2).sum()),cyc_hit=int(((gl==2)&hit).sum()),
                     ped_miss=int(((gl==1)&~hit).sum()),fp=fp,g_only=int((hitg&~hit).sum())))
def top(key,k=3,f=lambda r:True): return [(r["id"],r[key],r["n"]) for r in sorted(filter(f,rows),key=lambda r:-r[key])[:k]]
print("miss_small",top("miss_small")); print("cyclists",top("cyc",f=lambda r:r["cyc_hit"]==0))
print("ped_miss",top("ped_miss")); print("fp",top("fp")); print("gdino_only",top("g_only"))
